"""Trusted PPO record I/O usable by game workers without TensorFlow."""
from dataclasses import dataclass
import gzip
import os
from pathlib import Path
import pickle
import numpy as np


@dataclass
class Sample:
    arrays: dict
    memory: np.ndarray
    advance: bool
    action: int
    value: float
    log_probs: np.ndarray | None = None


class TrajectoryWriter:
    def __init__(self, path):
        self.value_sum = self.value_square_sum = 0.
        storage = os.environ.get('PATCHWORK_ROLLOUT_STORAGE', 'pickle')
        if storage not in ('pickle', 'indexed'):
            raise ValueError('Unknown rollout storage')
        self.numeric = None
        if storage == 'indexed':
            from .indexed_rollouts import IndexedTrajectoryWriter
            self.numeric = IndexedTrajectoryWriter(path)
            self.path, self.file, self.count = self.numeric.path, self.numeric.file, 0
        else:
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = gzip.open(self.path, 'wb', compresslevel=1)
            self.count = 0

    def write(self, sample):
        self.value_sum += sample.value
        self.value_square_sum += sample.value * sample.value
        if self.numeric is not None:
            self.numeric.write(sample)
            self.count = self.numeric.count
        else:
            pickle.dump(sample, self.file, protocol=5)
            self.count += 1

    def close(self):
        if self.numeric is not None:
            self.numeric.close()
        else:
            self.file.close()


def read_trajectory(path):
    if Path(path).suffix == '.rlt':
        from .indexed_rollouts import IndexedTrajectoryReader
        with IndexedTrajectoryReader(path) as reader:
            yield from reader
        return
    with gzip.open(path, 'rb') as stream:
        while True:
            try:
                yield pickle.load(stream)
            except EOFError:
                return
