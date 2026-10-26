"""Attribute host memory across the unchanged five-round actor-cache replay.

The existing soak driver and ML implementation are called without edits. Natural
update boundaries receive passive Linux/glibc/Python metadata only. After all five
updates, while the exact model, both Adams and PCG remain alive, one bounded idle
control, one GC and one malloc_trim probe run. Those interventions never occur
between updates and cannot be used to pass the original lifecycle gates.

No tracemalloc, new games, checkpoint writes, clear_session, allocation retry,
model/RL override or production allocator policy change is performed.
"""
import argparse
import copy
import ctypes
import gc
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import weakref
import xml.etree.ElementTree as ET

from examples import benchmark_actor_cache_soak as soak
from examples.profile_ppo_update import sha, write
from examples.profile_packed_update import counters


class Mallinfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in (
        'arena', 'ordblks', 'smblks', 'hblks', 'hblkhd', 'usmblks',
        'fsmblks', 'uordblks', 'fordblks', 'keepcost')]


class LibcAllocator:
    """glibc-only observations; other native allocators are explicitly outside it."""
    def __init__(self):
        self.libc = ctypes.CDLL('libc.so.6', use_errno=True)
        self.libc.mallinfo2.argtypes = []
        self.libc.mallinfo2.restype = Mallinfo2
        self.libc.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
        self.libc.malloc_info.restype = ctypes.c_int
        self.libc.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.libc.fopen.restype = ctypes.c_void_p
        self.libc.fclose.argtypes = [ctypes.c_void_p]
        self.libc.fclose.restype = ctypes.c_int
        self.libc.malloc_trim.argtypes = [ctypes.c_size_t]
        self.libc.malloc_trim.restype = ctypes.c_int

    def mallinfo(self):
        info = self.libc.mallinfo2()
        return {name: int(getattr(info, name)) for name, _ in Mallinfo2._fields_}

    def xml(self, path):
        path = Path(path)
        handle = self.libc.fopen(bytes(path), b'w')
        if not handle:
            raise OSError(ctypes.get_errno(), 'fopen failed for malloc_info')
        try:
            if self.libc.malloc_info(0, handle):
                raise RuntimeError('glibc malloc_info failed')
        finally:
            if self.libc.fclose(handle):
                raise OSError(ctypes.get_errno(), 'fclose failed for malloc_info')
        root = ET.parse(path).getroot()
        if root.tag != 'malloc':
            raise ValueError('Unexpected malloc_info root')
        # The root totals already include all arenas; do not add them to each
        # heap's totals and accidentally count the same memory twice.
        return dict(heaps=len(root.findall('heap')),
            totals={x.attrib['type']: {k: int(v) for k, v in x.attrib.items() if k != 'type'}
                    for x in root.findall('total')},
            system={x.attrib['type']: int(x.attrib['size']) for x in root.findall('system')},
            xml=path.name, xml_sha256=sha(path))

    def trim(self):
        return int(self.libc.malloc_trim(0))


def retained_python_size(value):
    """Shallow-container recursive estimate; never follows Tensor/weakref targets."""
    seen = set()

    def size(item):
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        total = sys.getsizeof(item)
        if isinstance(item, dict):
            total += sum(size(k) + size(v) for k, v in item.items())
        elif isinstance(item, (list, tuple, set, frozenset)):
            total += sum(size(v) for v in item)
        return total

    return size(value)


def smaps_summary(source, destination):
    """Save complete maps and aggregate byte counts without retaining map text."""
    shutil.copyfile(source, destination)
    groups = {}
    current = None
    with Path(destination).open() as file:
        for line in file:
            fields = line.split()
            if fields and '-' in fields[0] and len(fields) >= 5:
                path = ' '.join(fields[5:])
                current = ('anonymous' if not path else 'heap' if path == '[heap]'
                    else 'stack' if path.startswith('[stack')
                    else 'special' if path.startswith('[') else 'file')
                groups.setdefault(current, {})
            elif current and len(fields) == 3 and fields[2] == 'kB':
                name = fields[0].rstrip(':')
                groups[current][name] = groups[current].get(name, 0) + int(fields[1])
    return dict(groups_kib=groups, file=Path(destination).name,
                sha256=sha(destination))


class HostMemoryObserver:
    def __init__(self, output, *, idle_seconds=2.):
        self.output = Path(output)
        self.driver_sha256 = sha(__file__)
        self.directory = self.output/'host-memory-raw'
        self.allocator = LibcAllocator()
        self.idle_seconds = idle_seconds
        self.boundaries = []
        self.phases = []
        self.model_ref = self.trainer_ref = None
        self.state_function = None
        self.latest_state = None
        self.probe = self.lifetimes = None
        self.completed = False
        self.original_snapshot = soak.os_snapshot

    def bind(self, model, trainer, state, function):
        self.model_ref, self.trainer_ref = weakref.ref(model), weakref.ref(trainer)
        self.state_function = function
        self.latest_state = copy.deepcopy(state)

    def metadata(self):
        probe = self.probe
        lifetimes = self.lifetimes
        # Snapshot lists/dicts are scalar metadata, not Tensor payloads. Their
        # recursive Python sizes are estimates, not an allocator accounting API.
        with probe.event_lock if probe is not None else threading.Lock():
            probe_data = dict(samples=probe.samples, boundaries=probe.boundaries,
                sampling_errors=probe.sampling_errors, events=probe.events) if probe else {}
            probe_bytes = retained_python_size(probe_data)
        if lifetimes:
            with lifetimes.lock:
                lifetime_bytes = retained_python_size(dict(rows=lifetimes.rows,
                    tensor_refs=lifetimes.tensor_refs, originals=lifetimes.originals))
                weakrefs = len(lifetimes.tensor_refs)
                rows = len(lifetimes.rows)
        else:
            lifetime_bytes = weakrefs = rows = 0
        return dict(sampler_samples=len(probe.samples) if probe else 0,
            sampler_python_bytes_estimate=probe_bytes,
            lifetime_python_bytes_estimate=lifetime_bytes,
            lifetime_rows=rows, lifetime_cached_tensor_weakrefs=weakrefs,
            diagnostic_python_bytes_estimate=retained_python_size(dict(
                boundaries=self.boundaries, phases=self.phases)),
            limitations='Recursive sys.getsizeof over explicitly named observer containers; excludes allocator overhead and native allocations. Weakrefs/callables are sized but their targets/closures are not traversed.')

    def observe(self, label):
        self.directory.mkdir(parents=True, exist_ok=True)
        number = len(self.boundaries)
        prefix = f'{number:02d}-{label}'
        # Capture RSS before allocating XML/smaps/metadata diagnostics. The
        # second rollup quantifies each diagnostic boundary's own net effect.
        original = self.original_snapshot()
        row = dict(index=number, label=label, perf_ns=time.perf_counter_ns(),
            process_before_diagnostics_kib=original['process_kib'],
            mallinfo2_bytes=self.allocator.mallinfo(),
            gc_count=list(gc.get_count()), gc_stats=gc.get_stats(),
            ppo_threads=soak.ppo_threads())
        row['malloc_info'] = self.allocator.xml(self.directory/(prefix+'-malloc.xml'))
        row['smaps'] = smaps_summary('/proc/self/smaps', self.directory/(prefix+'-smaps.txt'))
        row['observer_metadata'] = self.metadata()
        row['process_after_diagnostics_kib'] = counters('/proc/self/smaps_rollup')
        self.boundaries.append(row)
        with (self.output/'host-memory-boundaries.jsonl').open('a') as file:
            file.write(json.dumps(row, allow_nan=False)+'\n')
        write(self.output/'host-memory-diagnostics.json', self.report('running'))
        original['host_memory_diagnostic_boundary'] = dict(index=number, label=label)
        return original

    def natural_boundary(self):
        index = sum(x['label'].startswith('natural-') for x in self.boundaries)
        if index == 0:
            label = 'natural-before-series'
        elif 1 <= index <= 10:
            round_id = soak.ROUNDS[(index-1)//2]
            label = f'natural-round-{round_id:04d}-' + ('before' if index % 2 else 'after')
        elif index == 11:
            label = 'natural-after-series'
        else:
            raise RuntimeError('Unexpected natural boundary count: original driver changed')
        return self.observe(label)

    def current_state(self):
        model, trainer = self.model_ref(), self.trainer_ref()
        if model is None or trainer is None:
            raise RuntimeError('Model/trainer expired before final allocator diagnostic')
        return self.state_function(model, trainer)

    def interventions(self):
        if self.completed:
            raise RuntimeError('Post-series diagnostic repeated')
        natural = [x['label'] for x in self.boundaries if x['label'].startswith('natural-')]
        expected = ['natural-before-series'] + [
            f'natural-round-{r:04d}-{side}' for r in soak.ROUNDS for side in ('before', 'after')]
        if natural != expected:
            raise RuntimeError('All five unmodified update boundaries must precede probes')
        expected_state = copy.deepcopy(self.latest_state)
        for name, operation in (
            ('idle-control', lambda: time.sleep(self.idle_seconds)),
            ('gc-collect', gc.collect),
            ('malloc-trim', self.allocator.trim),
        ):
            before = self.current_state()
            if before != expected_state:
                raise RuntimeError('Learning state changed before post-series probe')
            before_snapshot = self.observe(name+'-before')['host_memory_diagnostic_boundary']['index']
            started = time.perf_counter_ns()
            value = operation()
            elapsed = time.perf_counter_ns()-started
            after_snapshot = self.observe(name+'-after')['host_memory_diagnostic_boundary']['index']
            after = self.current_state()
            if after != before:
                raise RuntimeError('Learning state changed during post-series probe')
            phase = dict(name=name, before_boundary=before_snapshot,
                after_boundary=after_snapshot, operation_elapsed_ns=elapsed,
                operation_result=value, state_before=before, state_after=after,
                model_both_adams_pcg_unchanged=True,
                used_to_pass_normal_lifecycle_gates=False)
            self.phases.append(phase)
            write(self.output/'host-memory-diagnostics.json', self.report('running'))
        self.completed = True

    def report(self, status):
        return dict(status=status, driver_sha256=self.driver_sha256,
            no_tracemalloc=True, no_between_update_gc_or_trim=True,
            no_clear_session=True, no_new_learning_settings=True,
            idle_seconds=self.idle_seconds, boundaries=self.boundaries,
            post_series_phases=self.phases,
            all_three_post_series_probes_completed=self.completed,
            allocator_limitations='mallinfo2 and malloc_info describe glibc arenas and direct mmap allocations known to glibc, not all TensorFlow/CUDA/custom native allocators. Free malloc arena space is not necessarily fully resident or fully releasable. Only smaps RSS before/after trim measures observed resident return. malloc_trim success alone does not measure bytes.',
            intervention_limitations='Post-series only; interventions do not establish a natural long-run RSS plateau or justify per-update production trimming. State hashing/materialization and passive observation can allocate; an idle control and before/after-diagnostic RSS expose some of that overhead.')


def run(args):
    if args.idle_seconds != 2.:
        raise ValueError('The single idle control is fixed at two seconds')
    if args.final_gc_diagnostic:
        raise ValueError('Use the separately recorded host-memory post-series probes only')
    observer = HostMemoryObserver(args.output, idle_seconds=args.idle_seconds)
    original_state_summary = soak.state_summary
    original_lifetimes = soak.LifetimeObserver
    original_probe = soak.StorageProbe
    original_snapshot = soak.os_snapshot

    def state_summary(model, trainer):
        result = original_state_summary(model, trainer)
        observer.bind(model, trainer, result, original_state_summary)
        return result

    class ObservedProbe(original_probe):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            observer.probe = self

    class ObservedLifetimes(original_lifetimes):
        def __enter__(self):
            result = super().__enter__()
            observer.lifetimes = self
            return result

        def __exit__(self, error_type, error, traceback):
            try:
                if error_type is None:
                    observer.interventions()
            finally:
                super().__exit__(error_type, error, traceback)

    soak.state_summary = state_summary
    soak.LifetimeObserver = ObservedLifetimes
    soak.StorageProbe = ObservedProbe
    soak.os_snapshot = observer.natural_boundary
    succeeded = False
    try:
        soak.run(args)
        if not observer.completed:
            raise RuntimeError('Post-series host-memory probes did not complete')
        if sha(__file__) != observer.driver_sha256:
            raise RuntimeError('Host-memory driver changed during diagnostic')
        succeeded = True
    finally:
        soak.state_summary = original_state_summary
        soak.LifetimeObserver = original_lifetimes
        soak.StorageProbe = original_probe
        soak.os_snapshot = original_snapshot
        if args.output.exists():
            diagnostic = observer.report('passed' if succeeded else 'failed')
            write(args.output/'host-memory-diagnostics.json', diagnostic)
            path = args.output/'report.json'
            if path.exists():
                report = json.loads(path.read_text())
                report['host_memory_diagnostic'] = dict(status=diagnostic['status'],
                    path='host-memory-diagnostics.json', sha256=sha(args.output/'host-memory-diagnostics.json'),
                    driver_sha256=observer.driver_sha256, natural_update_boundaries_unchanged=True,
                    post_series_probes_completed=observer.completed,
                    no_tracemalloc=True, no_between_update_gc_or_trim=True)
                write(path, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy-cache-bytes', type=int, default=soak.CACHE_BYTES)
    parser.add_argument('--actor-cache-bytes', type=int, default=soak.CACHE_BYTES)
    parser.add_argument('--final-gc-diagnostic', action='store_true')
    parser.add_argument('--idle-seconds', type=float, default=2.)
    run(parser.parse_args())
