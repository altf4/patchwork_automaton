"""Return unused host allocator pages at quiescent pipeline boundaries.

This never changes tensors, neural configuration or optimizer/RNG state. It
must run after readers/optimizers/validators have joined, not during a batch.
"""
import ctypes
import gc
from pathlib import Path
import time


class _Mallinfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in (
        'arena', 'ordblks', 'smblks', 'hblks', 'hblkhd', 'usmblks',
        'fsmblks', 'uordblks', 'fordblks', 'keepcost')]


def _process_memory():
    result = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'RssAnon', 'VmSwap'):
            result[key] = int(value.split()[0]) * 1024
    return result


class HostMemoryReclaimer:
    """Explicit Linux/glibc runtime policy; unsupported runtimes fail early.

    Freed arrays can leave resident pages in glibc's thread arenas. Collection
    drops unreachable Python cycles, then malloc_trim returns free pages from
    those arenas. No session reset, allocation retry or cache-budget change.
    """
    def __init__(self):
        try:
            self._libc = ctypes.CDLL('libc.so.6', use_errno=True)
            self._trim = self._libc.malloc_trim
            self._info = self._libc.mallinfo2
        except (OSError, AttributeError) as error:
            raise RuntimeError('Host memory release requires Linux/glibc malloc_trim and mallinfo2') from error
        self._trim.argtypes = [ctypes.c_size_t]
        self._trim.restype = ctypes.c_int
        self._info.argtypes = []
        self._info.restype = _Mallinfo2
        _process_memory()  # Validate passive instrumentation before doing work.

    def snapshot(self):
        info = self._info()
        return dict(process_bytes=_process_memory(),
                    glibc_bytes={name: int(getattr(info, name)) for name, _ in _Mallinfo2._fields_})

    def release(self, stage):
        if not isinstance(stage, str) or not stage:
            raise ValueError('Host memory boundary must have a name')
        before = self.snapshot()
        started = time.monotonic()
        collected = gc.collect()
        after_gc = self.snapshot()
        trimmed = int(self._trim(0))
        after = self.snapshot()
        return dict(stage=stage, before=before, after_gc=after_gc, after=after,
                    gc_collected=collected, trim_return=trimmed,
                    seconds=time.monotonic()-started,
                    rss_released_bytes=before['process_bytes']['VmRSS']-after['process_bytes']['VmRSS'])
