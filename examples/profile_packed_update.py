"""Observe an unchanged packed update: host spans, page faults and Linux I/O.

Replays saved training trajectories only. OS sampling is passive; it never
flushes caches, alters memory limits or changes the training configuration.
"""
import argparse
from contextlib import contextmanager
import functools
import json
from pathlib import Path
import resource
import threading
import time

from examples import benchmark_packed_update as replay
from examples.profile_ppo_update import Probe, instrument, write


def counters(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        fields = line.replace(':', '').split()
        if len(fields) >= 2:
            try: result[fields[0]] = int(fields[1])
            except ValueError: pass
    return result


def os_snapshot():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    row = dict(perf_ns=time.perf_counter_ns(), wall_ns=time.time_ns(),
               minor_faults=usage.ru_minflt, major_faults=usage.ru_majflt,
               maxrss_kib=usage.ru_maxrss, user_seconds=usage.ru_utime,
               system_seconds=usage.ru_stime,
               memory_kib=counters('/proc/meminfo'), vmstat=counters('/proc/vmstat'))
    # Some WSL kernels omit CONFIG_TASK_IO_ACCOUNTING. Missing is not zero I/O.
    try: row['io'] = counters('/proc/self/io')
    except OSError as error: row['io'] = None; row['io_unavailable'] = str(error)
    for kind in ('memory', 'io', 'cpu'):
        row[kind+'_pressure'] = Path('/proc/pressure/'+kind).read_text()
    row['process_kib'] = counters('/proc/self/smaps_rollup')
    row['diskstats'] = Path('/proc/diskstats').read_text()
    return row


class StorageProbe(Probe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.samples = []; self.boundaries = {}; self.sampling_errors = []
        self.done = threading.Event()
        self.sampler = threading.Thread(target=self.sample, name='passive-os-sampler', daemon=True)
        self.sampler.start()

    def sample(self):
        while not self.done.is_set():
            try: self.samples.append(os_snapshot())
            except Exception as error: self.sampling_errors.append(repr(error))
            self.done.wait(1)

    @contextmanager
    def span(self, name):
        if name == 'full_update': self.boundaries['before'] = os_snapshot()
        try:
            with super().span(name): yield
        finally:
            if name == 'full_update': self.boundaries['after'] = os_snapshot()

    def finish(self, output):
        self.done.set(); self.sampler.join(timeout=10)
        if self.sampler.is_alive(): raise RuntimeError('Passive sampler did not exit')
        write(Path(output)/'os-samples.json', dict(samples=self.samples,
              boundaries=self.boundaries, errors=self.sampling_errors,
              units='meminfo/smaps: KiB; proc io: bytes; vmstat per Linux definitions',
              limitations='Linux counters; Windows host paging is not measured. System counters may include unrelated work.'))
        return super().finish(output)


def instrument_storage(trainer, probe):
    transformed = instrument(trainer, probe)
    from patchwork_automaton.neural.packed_rollouts import NumericRollout, PolicyBatchCache
    for cls, prefix, methods in (
        (NumericRollout, 'numeric', ('gather', 'finish')),
        (PolicyBatchCache, 'policy_cache', ('get', 'put', 'finish')),
    ):
        for name in methods:
            original = getattr(cls, name)
            @functools.wraps(original)
            def wrapped(*args, _original=original, _label=prefix+'_'+name, **kwargs):
                with probe.span(_label): return _original(*args, **kwargs)
            setattr(cls, name, wrapped)
    return transformed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--round', type=int, default=40)
    parser.add_argument('--instrumented', action='store_true')
    parser.add_argument('--trace-seconds', type=float, default=30)
    args = parser.parse_args()
    args.device = 'gpu'; args.gru_backend = 'cudnn-memory'; args.packing_variant = 'both'
    replay.Probe = StorageProbe; replay.instrument = instrument_storage
    replay.run(args)
