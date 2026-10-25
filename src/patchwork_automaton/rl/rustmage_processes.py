"""Persistent game engines with local or replicated frozen neural inference.

Each worker has one synchronous, private pipe. Bridge threads submit concurrent
requests to the existing batcher without copying TensorFlow models into workers.
Only already-redacted numerical observations cross the inference boundary.
"""
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import os
from threading import Event
import time
from types import SimpleNamespace
from ..neural.inference_processes import InferenceProcessGroup, bridge_requests, weights_digest


class _RemoteInference:
    def __init__(self, connection):
        self.connection = connection
        self.wait_seconds = 0.
        self.requests = 0

    def predict(self, batch, *, version, identity, unbatched=False):
        started = time.perf_counter()
        self.connection.send((batch, version, identity, unbatched))
        if not self.connection.poll(120): raise TimeoutError('Process inference timed out')
        ok, result = self.connection.recv()
        self.wait_seconds += time.perf_counter() - started
        self.requests += 1
        if not ok: raise RuntimeError(result)
        return result


def _initialize_worker(connections, counter, metadata):
    global _worker_model, _worker_batcher
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    if index >= len(connections): raise RuntimeError('Unexpected replacement game worker')
    _worker_model = SimpleNamespace(**metadata)
    _worker_batcher = _RemoteInference(connections[index])
    for i, connection in enumerate(connections):
        if i != index: connection.close()


def _run_game(options):
    from .rustmage_experiment import play_game
    _worker_batcher.wait_seconds = 0.
    _worker_batcher.requests = 0
    started = time.process_time()
    result = play_game(model=_worker_model, batcher=_worker_batcher, **options)
    result['worker_cpu_seconds'] = time.process_time() - started
    result['worker_inference_seconds'] = _worker_batcher.wait_seconds
    result['worker_inference_requests'] = _worker_batcher.requests
    result['worker_pid'] = os.getpid()
    return result


class ProcessGameCollector:
    def __init__(self, model, batcher=None, *, workers=4, inference_processes=2, wait_ms=0,
                 inference_device='cpu'):
        if workers < 1: raise ValueError('Positive worker count required')
        if batcher is None and (type(inference_processes) is not int or not 1 <= inference_processes <= workers):
            raise ValueError('Inference processes must be between one and the game-engine count')
        self.workers, self.batcher, self.model = workers, batcher, model
        self.stopping = Event()
        context = mp.get_context('spawn')  # Forking a live TensorFlow runtime is unsafe.
        pairs = [context.Pipe() for _ in range(workers)]
        self.connections = [p[0] for p in pairs]
        self.worker_connections = [p[1] for p in pairs]
        self.replicas = (InferenceProcessGroup(self.connections,replicas=inference_processes,wait_ms=wait_ms,
                                               device=inference_device)
                         if batcher is None else None)
        self.bridges = None
        if batcher is not None:
            self.bridges = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='process-inference')
            self.bridge_futures = [self.bridges.submit(bridge_requests,c,batcher,self.stopping) for c in self.connections]
        # Keras wraps dictionaries in trackers containing non-picklable callbacks.
        metadata = dict(card_count=model.card_count, memory_size=model.memory_size, feature_spec=dict(model.feature_spec))
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
            initializer=_initialize_worker, initargs=(self.worker_connections, context.Value('i', 0), metadata))
        self.closed = self.active = self.broken = False
        self.policy_digest = None

    async def start(self, *, version=0):
        if self.closed or self.broken or self.active: raise RuntimeError('Collector is unavailable or active')
        try:
            if self.replicas and not self.replicas.started:
                await self.replicas.start(self.model,version=version)
            elif self.replicas and self.version != version:
                raise ValueError('Use sync_weights to change an initialized policy version')
            elif not self.replicas:
                self.batcher.set_version(version)
        except BaseException:
            self.broken = True
            raise
        self.policy_digest = self.replicas.digest if self.replicas else weights_digest(self.model.get_weights())
        return self

    @property
    def version(self):
        return self.replicas.version if self.replicas else self.batcher.version

    async def sync_weights(self, model, version):
        if self.closed or self.broken or self.active:
            raise RuntimeError('Cannot synchronize an unavailable or active collector')
        try:
            if self.replicas:
                await self.replicas.sync(model,version)
            else:
                if model is not self.model: raise ValueError('Local inference must use the learner model')
                self.batcher.set_version(version)
        except BaseException:
            self.broken = True
            raise
        self.policy_digest = weights_digest(model.get_weights())

    def stats(self):
        if self.replicas: return self.replicas.stats()
        return dict(self.batcher.stats(),version=self.version,weights_sha256=self.policy_digest)

    async def run_games(self, *, games, seed_base, index_base=0, progress=None, **options):
        if self.closed or self.broken or self.active: raise RuntimeError('Collector is unavailable or active')
        if self.replicas:
            if not self.replicas.started: await self.start()
            options.setdefault('version',self.version)
            if options['version'] != self.version: raise ValueError('Stale model version for collection')
        loop = asyncio.get_running_loop()
        async def one(i):
            row = await loop.run_in_executor(self.pool, _run_game,
                dict(options, seed=seed_base+i, index=index_base+i))
            if progress: progress(row)
            return row
        self.active = True
        try:
            # Finish all submitted games even if a progress callback fails.
            # Ordering, seeds and seats are independent of completion order.
            results = await asyncio.gather(*(one(i) for i in range(games)),return_exceptions=True)
            for result in results:
                if isinstance(result,BaseException): raise result
            if self.replicas: await self.replicas.refresh_stats()
            return results
        except BaseException:
            self.broken = True
            raise
        finally: self.active = False

    async def close(self):
        if self.closed: return
        self.closed = True
        try:
            await asyncio.to_thread(self.pool.shutdown, wait=True, cancel_futures=True)
        finally:
            self.stopping.set()
            if self.bridges: await asyncio.to_thread(self.bridges.shutdown, wait=True)
            if self.replicas: await self.replicas.close()
            for connection in self.connections + self.worker_connections: connection.close()
