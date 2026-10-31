"""Two frozen CPU inference paths for Standard PPO learner and opponent.

The opponent is a pinned checkpoint for the duration of each collection round.
Only the learner path is synchronized after a PPO update. Game workers hold no
TensorFlow model and route the two seats through separate private pipes.
"""
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import time
from types import SimpleNamespace

from ..neural.inference_processes import InferenceProcessGroup
from .rustmage_processes import _RemoteInference


def _initialize_worker(connections, counter, metadata):
    global _learner_model, _opponent_model, _learner_batcher, _opponent_batcher
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    if index >= len(connections[0]):
        raise RuntimeError('Unexpected replacement Standard self-play worker')
    _learner_model = SimpleNamespace(**metadata[0])
    _opponent_model = SimpleNamespace(**metadata[1])
    _learner_batcher = _RemoteInference(connections[0][index])
    _opponent_batcher = _RemoteInference(connections[1][index])
    for role in connections:
        for i, connection in enumerate(role):
            if i != index:
                connection.close()


def _run_game(options):
    from .standard_ppo import play_ppo_game
    options = dict(options)
    if options.pop('game_runner', None) != 'standard_ppo':
        raise ValueError('Standard self-play collector accepts only PPO games')
    for batcher in (_learner_batcher, _opponent_batcher):
        batcher.wait_seconds = 0.
        batcher.requests = 0
    started = time.process_time()
    row = play_ppo_game(model=_learner_model, batcher=_learner_batcher,
        opponent_model=_opponent_model, opponent_batcher=_opponent_batcher, **options)
    row.update(worker_cpu_seconds=time.process_time()-started, worker_pid=os.getpid(),
        worker_inference_seconds=row.get('game_inference_seconds',_learner_batcher.wait_seconds),
        worker_inference_requests=row.get('game_inference_requests',_learner_batcher.requests),
        opponent_inference_seconds=row.get('opponent_game_inference_seconds',_opponent_batcher.wait_seconds),
        opponent_inference_requests=row.get('opponent_game_inference_requests',_opponent_batcher.requests))
    return row


class StandardSelfplayCollector:
    """Persistent engines with independently versioned learner/opponent groups."""
    def __init__(self, model, opponent_model, *, workers=24,
                 inference_processes=1, opponent_inference_processes=1):
        if type(workers) is not int or workers < 1:
            raise ValueError('Positive worker count required')
        for count in (inference_processes, opponent_inference_processes):
            if type(count) is not int or not 1 <= count <= workers:
                raise ValueError('Inference replicas must be between one and workers')
        self.model, self.opponent_model, self.workers = model, opponent_model, workers
        context = mp.get_context('spawn')
        pairs = [[context.Pipe() for _ in range(workers)] for _ in range(2)]
        self.connections = [[pair[0] for pair in role] for role in pairs]
        self.worker_connections = [[pair[1] for pair in role] for role in pairs]
        self.replicas = (
            InferenceProcessGroup(self.connections[0], replicas=inference_processes, device='cpu'),
            InferenceProcessGroup(self.connections[1], replicas=opponent_inference_processes, device='cpu'),
        )
        metadata = [dict(card_count=m.card_count, memory_size=m.memory_size,
                         feature_spec=dict(m.feature_spec)) for m in (model, opponent_model)]
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
            initializer=_initialize_worker,
            initargs=(self.worker_connections, context.Value('i', 0), metadata))
        self.closed = self.active = self.broken = False
        self.policy_digest = self.opponent_digest = None

    @property
    def version(self):
        return self.replicas[0].version

    @property
    def opponent_version(self):
        return self.replicas[1].version

    async def start(self, *, version=0, opponent_version=0):
        if self.closed or self.broken or self.active:
            raise RuntimeError('Collector is unavailable or active')
        try:
            for group, model, expected in zip(self.replicas,
                    (self.model, self.opponent_model), (version, opponent_version)):
                if not group.started:
                    await group.start(model, version=expected)
                elif group.version != expected:
                    raise ValueError('Use sync methods to change initialized policy versions')
            self.policy_digest = self.replicas[0].digest
            self.opponent_digest = self.replicas[1].digest
        except BaseException:
            self.broken = True
            raise
        return self

    async def sync_weights(self, model, version):
        if self.closed or self.broken or self.active:
            raise RuntimeError('Cannot synchronize unavailable or active collector')
        try:
            await self.replicas[0].sync(model, version)
            self.policy_digest = self.replicas[0].digest
        except BaseException:
            self.broken = True
            raise

    async def sync_opponent(self, model, version):
        if self.closed or self.broken or self.active:
            raise RuntimeError('Cannot synchronize unavailable or active opponent')
        try:
            await self.replicas[1].sync(model, version)
            self.opponent_digest = self.replicas[1].digest
        except BaseException:
            self.broken = True
            raise

    def stats(self):
        return dict(learner=self.replicas[0].stats(), opponent=self.replicas[1].stats())

    async def run_schedule(self, jobs, *, progress=None, **options):
        if self.closed or self.broken or self.active:
            raise RuntimeError('Collector is unavailable or active')
        if not all(group.started for group in self.replicas):
            raise RuntimeError('Both inference groups must be started')
        options.setdefault('version', self.version)
        options.setdefault('opponent_version', self.opponent_version)
        if options['version'] != self.version or options['opponent_version'] != self.opponent_version:
            raise ValueError('Stale learner or opponent version')
        loop = asyncio.get_running_loop()
        self.active = True
        async def one(job):
            row = await loop.run_in_executor(self.pool, _run_game, dict(options, **job))
            if progress:
                progress(row)
            return row
        try:
            rows = await asyncio.gather(*(one(job) for job in jobs), return_exceptions=True)
            for row in rows:
                if isinstance(row, BaseException):
                    raise row
            for group in self.replicas:
                await group.refresh_stats()
            return rows
        except BaseException:
            self.broken = True
            raise
        finally:
            self.active = False

    async def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            await asyncio.to_thread(self.pool.shutdown, wait=True, cancel_futures=True)
        finally:
            for group in self.replicas:
                await group.close()
            for role in self.connections + self.worker_connections:
                for connection in role:
                    connection.close()
