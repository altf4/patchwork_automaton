"""Frozen neural-versus-neural evaluation with separate CPU inference pools.

Each engine owns two private inference pipes, one per policy. Models are never
trained, synchronized or sent to an opponent's inference service during a duel.
The focal checkpoint occupies index % 2; its outcome defines win/loss in rows.
"""
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import time
from types import SimpleNamespace

from ..neural.inference_processes import InferenceProcessGroup
from .rustmage_processes import _RemoteInference


def _initialize_duel_worker(connections, counter, metadata):
    global _duel_models, _duel_batchers
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    if index >= len(connections[0]):
        raise RuntimeError('Unexpected replacement duel worker')
    _duel_models = [SimpleNamespace(**item) for item in metadata]
    _duel_batchers = [_RemoteInference(role[index]) for role in connections]
    for role in connections:
        for i, connection in enumerate(role):
            if i != index:
                connection.close()


def _run_duel_game(options):
    from .rustmage_experiment import play_game
    for batcher in _duel_batchers:
        batcher.wait_seconds = 0.
        batcher.requests = 0
    started = time.process_time()
    row = play_game(mode='versus_neural', model=_duel_models[0], batcher=_duel_batchers[0],
        opponent_model=_duel_models[1], opponent_batcher=_duel_batchers[1], **options)
    row.update(worker_cpu_seconds=time.process_time()-started, worker_pid=os.getpid(),
        focal_inference_requests=_duel_batchers[0].requests,
        opponent_inference_requests=_duel_batchers[1].requests)
    return row


class DuelProcessCollector:
    """A fixed pair of policies, balanced by the ordinary seeded game scheduler."""
    def __init__(self, model, opponent_model, *, workers=48, replicas_per_policy=2):
        if type(workers) is not int or workers < 1:
            raise ValueError('Positive integer game-worker count required')
        if type(replicas_per_policy) is not int or not 1 <= replicas_per_policy <= workers:
            raise ValueError('Inference replicas per policy must be between one and workers')
        self.models = (model, opponent_model)
        self.workers = workers
        self.started = self.closed = self.active = self.broken = False
        context = mp.get_context('spawn')
        pairs = [[context.Pipe() for _ in range(workers)] for _ in self.models]
        self.connections = [[pair[0] for pair in role] for role in pairs]
        self.worker_connections = [[pair[1] for pair in role] for role in pairs]
        self.replicas = [InferenceProcessGroup(role, replicas=replicas_per_policy,
            wait_ms=0, device='cpu') for role in self.connections]
        metadata = [dict(card_count=m.card_count, memory_size=m.memory_size,
                         feature_spec=dict(m.feature_spec)) for m in self.models]
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
            initializer=_initialize_duel_worker,
            initargs=(self.worker_connections, context.Value('i', 0), metadata))

    async def start(self):
        if self.started or self.closed or self.broken:
            raise RuntimeError('Duel collector cannot start')
        try:
            for group, model in zip(self.replicas, self.models):
                await group.start(model, version=0)
            self.started = True
        except BaseException:
            self.broken = True
            await self.close()
            raise
        return self

    def stats(self):
        return dict(game_workers=self.workers, focal=self.replicas[0].stats(),
                    opponent=self.replicas[1].stats())

    async def run_games(self, *, games, seed_base, index_base=0, progress=None, **limits):
        if not self.started or self.closed or self.active or self.broken:
            raise RuntimeError('Duel collector is unavailable or active')
        if type(games) is not int or games < 1:
            raise ValueError('Positive integer game count required')
        allowed = {'max_decisions', 'max_own_turns', 'repeat_limit', 'greedy'}
        if set(limits)-allowed:
            raise ValueError('Only evaluation limits may be passed to a duel')
        loop = asyncio.get_running_loop()
        self.active = True
        async def one(i):
            row = await loop.run_in_executor(self.pool, _run_duel_game,
                dict(seed=seed_base+i, index=index_base+i, **limits))
            if progress:
                progress(row)
            return row
        try:
            rows = await asyncio.gather(*(one(i) for i in range(games)), return_exceptions=True)
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
            # Finish submitted games before closing their inference pipes.
            await asyncio.to_thread(self.pool.shutdown, wait=True, cancel_futures=True)
        finally:
            try:
                results = await asyncio.gather(*(group.close() for group in self.replicas),
                                               return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
            finally:
                for role in self.connections + self.worker_connections:
                    for connection in role:
                        connection.close()
