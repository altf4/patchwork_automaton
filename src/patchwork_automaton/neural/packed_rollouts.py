"""Update-local numerical packing without changing batch padding or record order.

Backing bytes live in an anonymous temporary file, mapped read-only after the
initial scan. This trades temporary disk/page-cache use for repeated Python
record decoding. The explicit in-memory rollout cache remains separately bounded.
"""
import mmap
import tempfile
from types import MappingProxyType
import numpy as np


class NumericRollout:
    """Append frozen records once; gather the same locally padded batches."""
    def __init__(self):
        self.file = tempfile.TemporaryFile()
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

    def gather(self, indices):
        if self.mapping is None: raise ValueError('Numerical rollout is not finalized')
        indices = np.asarray(indices, np.int64)
        if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= self.count):
            raise ValueError('Invalid numerical rollout indices')
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
        # The feature schema pads paired axes together. Frozen production rows
        # use matching extents; reject malformed input rather than alter shapes.
        if result['_old'].shape != result['candidate_mask'].shape:
            raise ValueError('Behavior and candidate extents disagree')
        self.gathers += 1
        return result

    def close(self):
        if self.mapping is not None:
            self.mapping.close(); self.mapping = None
        self.file.close()


class PolicyBatchCache:
    """Read-only mapped canonical inputs; model outputs are never cached."""
    def __init__(self):
        self.file = tempfile.TemporaryFile(); self.entries = {}; self.mapping = None
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
    def __init__(self, *, device, max_bytes):
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
        self.max_bytes = max_bytes
        self.entries = {}; self._counts = {}; self._fallback = set()
        self.bytes = self.uploads = self.hits = self.uploaded_tensors = 0
        self.fallback_batches = self.fallback_bytes = 0
        self.admission_closed = self.finished = self.closed = False

    def get(self, key):
        if self.closed: raise ValueError('Device policy cache is closed')
        batch = self.entries.get(key)
        if batch is not None: self.hits += 1
        return batch

    def counts(self, key):
        """Return frozen CPU reduction metadata without reading GPU tensors."""
        if self.closed: raise ValueError('Device policy cache is closed')
        return self._counts[key]

    def put(self, key, batch):
        if self.closed: raise ValueError('Device policy cache is closed')
        if key in self.entries: return self.entries[key]
        if key in self._fallback: return None
        if self.finished: raise ValueError('Device policy cache admission is finalized')
        inputs, *rest = batch
        # Evaluate the exact original CPU count expressions once, including
        # floating-point summation of advance_memory, before tensor upload.
        self._counts[key] = (int((inputs['candidate_mask'].sum(axis=1) > 1).sum()),
                             float(inputs['advance_memory'].sum()))
        arrays = [*inputs.values(), *rest]
        size = sum(array.nbytes for array in arrays)
        if self.admission_closed or self.bytes + size > self.max_bytes:
            self.admission_closed = True
            self._fallback.add(key)
            self.fallback_batches += 1; self.fallback_bytes += size
            return None
        # No OOM catch/replay: failures remain visible and unwind the update.
        # Explicit placement validation rejects TensorFlow's soft CPU fallback.
        with self._tf.device(self.device):
            tensors = [self._tf.convert_to_tensor(array) for array in arrays]
        requested = self._tf.DeviceSpec.from_string(self.device)
        for tensor in tensors:
            actual = self._tf.DeviceSpec.from_string(tensor.device)
            if (actual.device_type != requested.device_type
                    or actual.device_index != requested.device_index):
                raise RuntimeError(f'Policy cache tensor placement {tensor.device} != {self.device}')
        packed = (dict(zip(inputs, tensors[:len(inputs)])), *tensors[len(inputs):])
        self.entries[key] = packed
        self.bytes += size; self.uploads += 1; self.uploaded_tensors += len(tensors)
        return packed

    def finish(self):
        if self.closed: raise ValueError('Device policy cache is closed')
        self.entries = MappingProxyType(self.entries)
        self._counts = MappingProxyType(self._counts)
        self._fallback = frozenset(self._fallback)
        self.finished = True

    def close(self):
        self.entries = {}; self._counts = {}; self._fallback = set()
        self.bytes = 0; self.closed = True
