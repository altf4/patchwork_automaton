"""Conservative byte accounting for update-local numerical rollout caches."""
import sys
from pathlib import Path
from collections import OrderedDict
import pickle
import numpy as np


def retained_bytes(value):
    """Count owned buffers and container overhead; shared values overcount safely.

    Views charge their buffer too. Rollout records and collated batches contain
    only acyclic numeric arrays and ordinary containers. This bounds retained
    cache data, not TensorFlow allocations or transient minibatch workspaces.
    """
    native_allocations = set()
    def count(item):
        size = sys.getsizeof(item)
        if isinstance(item, np.ndarray):
            return size + (0 if item.flags.owndata else
                           count(item.base) if item.base is not None else item.nbytes)
        if isinstance(item, memoryview):
            owner = getattr(item.obj, '_allocation_owner', None)
            if owner is not None and hasattr(owner, 'native_nbytes'):
                # A slice can retain the entire Rust bundle. Charge that bundle
                # once within this cache entry, even when all 14 arrays share it.
                if id(owner) in native_allocations: return size
                native_allocations.add(id(owner))
                return size + sys.getsizeof(item.obj) + sys.getsizeof(owner) + owner.native_nbytes
            return size + max(item.nbytes, count(item.obj))
        if isinstance(item, dict):
            return size + sum(count(k) + count(v) for k, v in item.items())
        if isinstance(item, (list, tuple)):
            return size + sum(count(v) for v in item)
        if hasattr(item, '__dict__'):
            return size + count(vars(item))
        return size
    return count(value)


class RolloutCache:
    """Admission-only cache: avoids eviction churn during repeated full scans."""
    def __init__(self, limit):
        self.limit = limit
        self.bytes = self.hits = 0
        self.entries = {}

    def get(self, key):
        value = self.entries.get(key)
        if value is not None:
            self.hits += 1
        return value

    def put(self, key, value, *, size=None):
        # Include a conservative allowance for the outer dictionary entry.
        size = (retained_bytes(value) if size is None else size) + retained_bytes(key) + 128
        if self.bytes + size > self.limit:
            return False
        self.entries[key] = value
        self.bytes += size
        return True


class IndexedTrajectories:
    """Random access without loading uncached games into RAM.

    Cached trajectories are indexed directly. Numeric .rlt files support direct
    reads with at most 64 open readers. Each uncached gzip stream is copied once
    to a temporary seekable file, with one byte offset per sample.
    Only trusted, locally generated trajectory pickles are accepted. The caller
    owns and closes the file, including on update failure.
    """
    def __init__(self, trajectories, cache, reader, file):
        self.trajectories, self.cache, self.reader, self.file = trajectories, cache, reader, file
        self.offsets = {}
        self.disk_bytes = 0
        self.numeric_readers = OrderedDict()
        self.numeric_opens = 0

    def get(self, index, position):
        records = self.cache.get(index)
        if records is not None:
            return records[position]
        if Path(self.trajectories[index]['path']).suffix == '.rlt':
            reader = self.numeric_readers.get(index)
            if reader is None:
                from .indexed_rollouts import IndexedTrajectoryReader
                reader = IndexedTrajectoryReader(self.trajectories[index]['path'])
                if len(reader) != self.trajectories[index]['samples']:
                    reader.close(); raise ValueError('Incomplete trajectory file')
                self.numeric_readers[index] = reader
                self.numeric_opens += 1
                if len(self.numeric_readers) > 64:
                    self.numeric_readers.popitem(last=False)[1].close()
            self.numeric_readers.move_to_end(index)
            return reader.get(position)
        if index not in self.offsets:
            trajectory = self.trajectories[index]
            offsets = np.empty(trajectory['samples'], dtype=np.int64)
            self.file.seek(0, 2)
            seen = 0
            for record in self.reader(trajectory['path']):
                if seen >= len(offsets): raise ValueError('Incomplete trajectory file')
                offsets[seen] = self.file.tell()
                pickle.dump(record, self.file, protocol=5)
                seen += 1
            if seen != len(offsets): raise ValueError('Incomplete trajectory file')
            self.offsets[index] = offsets
            self.disk_bytes = self.file.tell()
        self.file.seek(int(self.offsets[index][position]))
        return pickle.load(self.file)

    def close(self):
        for reader in self.numeric_readers.values(): reader.close()
        self.numeric_readers.clear()
