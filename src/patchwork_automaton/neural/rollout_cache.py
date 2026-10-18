"""Conservative byte accounting for update-local numerical rollout caches."""
import sys
import numpy as np


def retained_bytes(value):
    """Count owned buffers and container overhead; shared values overcount safely.

    Views charge their buffer too. Rollout records and collated batches contain
    only acyclic numeric arrays and ordinary containers. This bounds retained
    cache data, not TensorFlow allocations or transient minibatch workspaces.
    """
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size + (0 if value.flags.owndata else
                       retained_bytes(value.base) if value.base is not None else value.nbytes)
    if isinstance(value, dict):
        return size + sum(retained_bytes(k) + retained_bytes(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return size + sum(retained_bytes(v) for v in value)
    if hasattr(value, '__dict__'):
        return size + retained_bytes(vars(value))
    return size


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
