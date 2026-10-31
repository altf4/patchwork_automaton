"""Update-local numerical packing without changing batch padding or record order.

Backing bytes live in an anonymous temporary file, mapped read-only after the
initial scan. This trades temporary disk/page-cache use for repeated Python
record decoding. The explicit in-memory rollout cache remains separately bounded.
"""
import mmap
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, wait
from types import MappingProxyType
import numpy as np


class NumericRollout:
    """Append frozen records once; gather the same locally padded batches."""
    def __init__(self, max_bytes=None, *, read_mode='auto', read_workers=4,
                 read_buffer_bytes=64*1024**2):
        if read_mode == 'auto':
            read_mode = 'coalesced' if hasattr(os, 'pread') else 'mmap'
        if read_mode not in ('coalesced', 'mmap'):
            raise ValueError('Unknown numerical rollout read mode')
        if type(read_workers) is not int or read_workers < 1:
            raise ValueError('Record reader workers must be positive')
        if type(read_buffer_bytes) is not int or read_buffer_bytes < 2:
            raise ValueError('Record reader buffer must be at least two bytes')
        if read_mode == 'coalesced' and not hasattr(os, 'pread'):
            raise ValueError('Coalesced rollout reads require positional file I/O')
        self.file = tempfile.TemporaryFile()
        self.max_bytes = max_bytes
        self.read_mode, self.read_workers = read_mode, read_workers
        self.read_buffer_bytes = read_buffer_bytes
        self.read_pool = None
        self.read_calls = self.read_bytes = self.read_spans = self.maximum_span_bytes = 0
        self.read_seconds = self.copy_seconds = self.gather_seconds = 0.
        self.schema = None
        self.offsets = []; self.shapes = []
        self.mapping = None
        self.count = self.disk_bytes = self.gathers = 0

    def append(self, sample):
        arrays = dict(sample.arrays, memory_in=np.asarray(sample.memory),
                      advance_memory=np.asarray([float(sample.advance)], np.float32),
                      _old=np.asarray(sample.log_probs, np.float32),
                      _action=np.asarray([sample.action], np.int32))
        schema = [(key, value.dtype.str, value.ndim) for key, value in arrays.items()]
        if self.max_bytes is not None and self.file.tell()+sum(v.nbytes+8 for v in arrays.values())>self.max_bytes:
            raise ValueError('Numerical rollout exceeds declared temporary-disk budget')
        if self.schema is None: self.schema = schema
        if schema != self.schema: raise ValueError('Numerical rollout schema changed')
        row_offsets = []; row_shapes = []
        for value in arrays.values():
            value = np.ascontiguousarray(value)
            pad = -self.file.tell() % 8
            if pad: self.file.write(b'\0' * pad)
            row_offsets.append(self.file.tell())
            row_shapes.extend(value.shape)
            if value.nbytes: self.file.write(memoryview(value).cast('B'))
        self.offsets.append(row_offsets); self.shapes.append(row_shapes); self.count += 1

    def finish(self):
        if not self.count: raise ValueError('Empty numerical rollout')
        pad = -self.file.tell() % 8
        if pad: self.file.write(b'\0' * pad)
        self.file.flush(); self.disk_bytes = self.file.tell()
        self.offsets = np.asarray(self.offsets, np.int64)
        self.shapes = np.asarray(self.shapes, np.int64)
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        if self.read_mode == 'coalesced' and hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(self.file.fileno(), 0, 0, os.POSIX_FADV_RANDOM)

    def gather(self, indices):
        if self.mapping is None: raise ValueError('Numerical rollout is not finalized')
        indices = np.asarray(indices, np.int64)
        if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= self.count):
            raise ValueError('Invalid numerical rollout indices')
        start = time.monotonic()
        result = (self._gather_coalesced(indices) if self.read_mode == 'coalesced'
                  else self._gather_mapped(indices))
        if result['_old'].shape != result['candidate_mask'].shape:
            raise ValueError('Behavior and candidate extents disagree')
        self.gathers += 1
        self.gather_seconds += time.monotonic()-start
        return result

    def _gather_mapped(self, indices):
        """Reference reader retained for numerical and throughput comparisons."""
        result = {}; column = 0
        for field, (key, dtype, rank) in enumerate(self.schema):
            shapes = self.shapes[indices, column:column+rank]; column += rank
            target = tuple(map(int, shapes.max(axis=0)))
            output = np.full((len(indices), *target), -1.e9 if key == '_old' else 0, dtype=dtype)
            # Direct contiguous row copies avoid an int64 index grid as large
            # as every padded feature tensor. Frozen values are never changed.
            for row, index in enumerate(indices):
                shape = tuple(map(int, shapes[row]))
                view = np.ndarray(shape, dtype=dtype, buffer=self.mapping,
                                  offset=int(self.offsets[index, field]))
                output[(row, *[slice(0, d) for d in shape])] = view
            result[key] = output
        return result

    def _read_exact(self, start, size):
        """One extent per request, with explicit short-read/EOF handling."""
        buffer = bytearray(size)
        view = memoryview(buffer)
        position = calls = 0
        while position < size:
            if hasattr(os, 'preadv'):
                count = os.preadv(self.file.fileno(), [view[position:]], start+position)
            else:
                data = os.pread(self.file.fileno(), min(size-position, 65536), start+position)
                count = len(data)
                view[position:position+count] = data
            calls += 1
            if count == 0:
                raise EOFError('Numerical rollout ended inside a record')
            position += count
        return buffer, calls

    def _gather_coalesced(self, indices):
        # Sort only physical reads. Every destination retains its original
        # shuffled row; repeated IDs are fetched once and copied to each row.
        identities, inverse = np.unique(indices, return_inverse=True)
        destinations = [[] for _ in identities]
        for row, identity in enumerate(inverse):
            destinations[int(identity)].append(row)
        starts = self.offsets[identities, 0]
        ends = np.asarray([self.offsets[i+1, 0] if i+1 < self.count else self.disk_bytes
                           for i in identities], np.int64)
        largest = int((ends-starts).max())
        if largest*2 > self.read_buffer_bytes:
            raise ValueError('Numerical record exceeds declared reader-buffer budget')
        workers = min(self.read_workers, max(1, self.read_buffer_bytes//(2*largest)))
        # The second half also bounds a positional-read fallback's copy and
        # any short-read bookkeeping. Padded output tensors are separate.
        limit = self.read_buffer_bytes//(2*workers)
        spans = []
        for identity, start, end, rows in zip(identities, starts, ends, destinations):
            item = (int(identity), rows)
            if spans and start == spans[-1][1] and end-spans[-1][0] <= limit:
                spans[-1][1] = int(end); spans[-1][2].append(item)
            else:
                spans.append([int(start), int(end), [item]])
        fields = []; result = {}; column = 0
        for field, (key, dtype, rank) in enumerate(self.schema):
            shapes = self.shapes[indices, column:column+rank]; column += rank
            target = tuple(map(int, shapes.max(axis=0)))
            output = np.full((len(indices), *target), -1.e9 if key == '_old' else 0, dtype=dtype)
            fields.append((field, dtype, shapes, output)); result[key] = output

        def consume(group):
            calls = byte_count = 0; read_seconds = copy_seconds = 0.
            for start, end, records in group:
                before = time.monotonic()
                buffer, reads = self._read_exact(start, end-start)
                read_seconds += time.monotonic()-before
                calls += reads; byte_count += end-start
                before = time.monotonic()
                for identity, rows in records:
                    for field, dtype, shapes, output in fields:
                        shape = tuple(map(int, shapes[rows[0]]))
                        value = np.ndarray(shape, dtype=dtype, buffer=buffer,
                                           offset=int(self.offsets[identity, field])-start)
                        for row in rows:
                            output[(row, *[slice(0, d) for d in shape])] = value
                copy_seconds += time.monotonic()-before
                # Release views before reading the next extent.
                del value, buffer
            return calls, byte_count, read_seconds, copy_seconds

        workers = min(workers, len(spans))
        if workers == 1:
            statistics = [consume(spans)]
        else:
            if self.read_pool is None:
                self.read_pool = ThreadPoolExecutor(max_workers=self.read_workers,
                                                    thread_name_prefix='ppo-record-read')
            futures = [self.read_pool.submit(consume, spans[i::workers]) for i in range(workers)]
            try:
                statistics = [future.result() for future in futures]
            except BaseException:
                for future in futures: future.cancel()
                wait(futures)
                raise
        self.read_spans += len(spans)
        self.maximum_span_bytes = max(self.maximum_span_bytes, max(end-start for start,end,_ in spans))
        for calls, byte_count, read_seconds, copy_seconds in statistics:
            self.read_calls += calls; self.read_bytes += byte_count
            self.read_seconds += read_seconds; self.copy_seconds += copy_seconds
        return result

    def stats(self):
        return dict(mode=self.read_mode, workers=self.read_workers,
            buffer_budget_bytes=self.read_buffer_bytes, calls=self.read_calls,
            spans=self.read_spans, bytes=self.read_bytes, maximum_span_bytes=self.maximum_span_bytes,
            gathers=self.gathers, gather_seconds=self.gather_seconds,
            worker_read_seconds=self.read_seconds, worker_copy_seconds=self.copy_seconds)

    def close(self):
        if self.read_pool is not None:
            self.read_pool.shutdown(wait=True, cancel_futures=True)
            self.read_pool = None
        if self.mapping is not None:
            self.mapping.close(); self.mapping = None
        self.file.close()


class PolicyBatchCache:
    """Read-only mapped canonical inputs; model outputs are never cached."""
    def __init__(self, max_bytes=None):
        self.file = tempfile.TemporaryFile(); self.entries = {}; self.mapping = None
        self.max_bytes = max_bytes
        self.hits = self.disk_bytes = 0

    def get(self, key):
        if self.mapping is None or key not in self.entries: return None
        keys, descriptors = self.entries[key]
        arrays = [np.ndarray(shape, dtype=dtype, buffer=self.mapping, offset=offset)
                  for offset, shape, dtype in descriptors]
        self.hits += 1
        return (dict(zip(keys, arrays[:len(keys)])), *arrays[len(keys):])

    def put(self, key, batch):
        if self.mapping is not None or key in self.entries: return
        inputs, *rest = batch
        arrays = [*inputs.values(), *rest]; descriptors = []
        if self.max_bytes is not None and self.file.tell()+sum(a.nbytes+8 for a in arrays)>self.max_bytes:
            return  # Retain the bounded prefix; uncached inputs are recomputed.
        for array in arrays:
            array = np.ascontiguousarray(array)
            pad = -self.file.tell() % 8
            if pad: self.file.write(b'\0' * pad)
            descriptors.append((self.file.tell(), array.shape, array.dtype.str))
            if array.nbytes: self.file.write(memoryview(array).cast('B'))
        self.entries[key] = (list(inputs), descriptors)
        self.disk_bytes = self.file.tell()

    def finish(self):
        self.file.flush()
        if self.disk_bytes: self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self):
        if self.mapping is not None:
            self.mapping.close(); self.mapping = None
        self.file.close()


class DevicePolicyBatchCache:
    """Bounded, update-local tensors for canonical read-only policy scans.

    Admission must be called in canonical order by one preparation thread.
    The first batch that does not fit ends admission for this update, making
    the retained prefix independent of reader scheduling. Only frozen inputs
    and labels are retained; policy outputs and drift are always recomputed.
    The byte limit counts tensor payloads, not TensorFlow allocator overhead.
    Returned tensors and their input dictionaries are caller-read-only.
    """
    def __init__(self, *, device, max_bytes, compact=False):
        if type(compact) is not bool:
            raise ValueError('Compact input cache flag must be boolean')
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('Device policy cache budget must be nonnegative integer bytes')
        if not isinstance(device, str) or not device:
            raise ValueError('Expected an explicit CPU or GPU cache device')
        # TensorFlow remains optional until the device cache is requested.
        import tensorflow as tf
        try:
            spec = tf.DeviceSpec.from_string(device)
        except (TypeError, ValueError) as error:
            raise ValueError('Expected an explicit CPU or GPU cache device') from error
        if (spec.device_type not in ('CPU', 'GPU') or spec.device_index is None
                or spec.device_index < 0 or spec.job is not None
                or spec.replica is not None or spec.task is not None):
            raise ValueError('Expected an explicit CPU or GPU cache device')
        self.device = f'/{spec.device_type}:{spec.device_index}'
        available = [tf.DeviceSpec.from_string(d.name)
                     for d in tf.config.list_logical_devices(spec.device_type)]
        if not any(d.device_index == spec.device_index for d in available):
            raise ValueError(f'Requested policy cache device is unavailable: {self.device}')
        self._tf = tf
        self.compact = compact
        self.max_bytes = max_bytes
        self.entries = {}; self._counts = {}; self._fallback = set()
        self.bytes = self.uploads = self.hits = self.uploaded_tensors = 0
        self.dense_bytes = self.compact_arrays = 0
        self.encode_seconds = self.decode_seconds = 0.
        self.fallback_batches = self.fallback_bytes = 0
        self.admission_closed = self.finished = self.closed = False

    def get(self, key):
        if self.closed: raise ValueError('Device policy cache is closed')
        batch = self.entries.get(key)
        if batch is not None: self.hits += 1
        return self._materialize(batch) if batch is not None else None

    def _materialize(self,batch):
        if not self.compact: return batch
        from .compact_inputs import DeviceSparseInput
        started=time.monotonic()
        with self._tf.device(self.device):
            values=[v.materialize(self._tf) if isinstance(v,DeviceSparseInput) else v
                    for v in [*batch[0].values(),*batch[1:]]]
        self.decode_seconds+=time.monotonic()-started
        return (dict(zip(batch[0],values[:len(batch[0])])),*values[len(batch[0]):])

    def counts(self, key):
        """Return frozen CPU reduction metadata without reading GPU tensors."""
        if self.closed: raise ValueError('Device policy cache is closed')
        return self._counts[key]

    def put(self, key, batch):
        if self.closed: raise ValueError('Device policy cache is closed')
        if key in self.entries: return self._materialize(self.entries[key])
        if key in self._fallback: return None
        if self.finished: raise ValueError('Device policy cache admission is finalized')
        inputs, *rest = batch
        # Evaluate the exact original CPU count expressions once, including
        # floating-point summation of advance_memory, before tensor upload.
        self._counts[key] = (int((inputs['candidate_mask'].sum(axis=1) > 1).sum()),
                             float(inputs['advance_memory'].sum()))
        arrays = [*inputs.values(), *rest]
        dense_size = sum(array.nbytes for array in arrays)
        from .compact_inputs import compact_array, SparseInput, DeviceSparseInput
        started=time.monotonic()
        storage=[compact_array(a) for a in arrays] if self.compact and not self.admission_closed else arrays
        self.encode_seconds+=time.monotonic()-started
        size = sum(array.nbytes for array in storage)
        if self.admission_closed or self.bytes + size > self.max_bytes:
            self.admission_closed = True
            self._fallback.add(key)
            self.fallback_batches += 1; self.fallback_bytes += dense_size
            return None
        # No OOM catch/replay: failures remain visible and unwind the update.
        # Explicit placement validation rejects TensorFlow's soft CPU fallback.
        with self._tf.device(self.device):
            tensors = [DeviceSparseInput(a,self._tf) if isinstance(a,SparseInput)
                       else self._tf.convert_to_tensor(a) for a in storage]
        placed=[t for value in tensors for t in
                ((value.indices,value.values) if isinstance(value,DeviceSparseInput) else (value,))]
        requested = self._tf.DeviceSpec.from_string(self.device)
        for tensor in placed:
            actual = self._tf.DeviceSpec.from_string(tensor.device)
            if (actual.device_type != requested.device_type
                    or actual.device_index != requested.device_index):
                raise RuntimeError(f'Policy cache tensor placement {tensor.device} != {self.device}')
        packed = (dict(zip(inputs, tensors[:len(inputs)])), *tensors[len(inputs):])
        self.entries[key] = packed
        self.bytes += size; self.dense_bytes += dense_size
        self.compact_arrays += sum(isinstance(v,DeviceSparseInput) for v in tensors)
        self.uploads += 1; self.uploaded_tensors += len(placed)
        return self._materialize(packed)

    def finish(self):
        if self.closed: raise ValueError('Device policy cache is closed')
        self.entries = MappingProxyType(self.entries)
        self._counts = MappingProxyType(self._counts)
        self._fallback = frozenset(self._fallback)
        self.finished = True

    def close(self):
        self.entries = {}; self._counts = {}; self._fallback = set()
        self.bytes = 0; self.closed = True
