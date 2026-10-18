"""Persistent spawned game workers; the parent alone owns neural inference.

Each worker has one synchronous, private pipe. Bridge threads submit concurrent
requests to the existing batcher without copying TensorFlow models into workers.
Only already-redacted numerical observations cross the inference boundary.
"""
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from threading import Event
import time
from types import SimpleNamespace


class _RemoteInference:
    def __init__(self, connection):
        self.connection = connection

    def predict(self, batch, *, version, identity, unbatched=False):
        self.connection.send((batch, version, identity, unbatched))
        if not self.connection.poll(120): raise TimeoutError('Process inference timed out')
        ok, result = self.connection.recv()
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
    started = time.process_time()
    result = play_game(model=_worker_model, batcher=_worker_batcher, **options)
    result['worker_cpu_seconds'] = time.process_time() - started
    return result


class ProcessGameCollector:
    def __init__(self, model, batcher, *, workers=4):
        if workers < 1: raise ValueError('Positive worker count required')
        self.workers, self.batcher = workers, batcher
        self.stopping = Event()
        context = mp.get_context('spawn')  # Forking a live TensorFlow runtime is unsafe.
        pairs = [context.Pipe() for _ in range(workers)]
        self.connections = [p[0] for p in pairs]
        self.worker_connections = [p[1] for p in pairs]
        self.bridges = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='process-inference')
        self.bridge_futures = [self.bridges.submit(self._bridge, c) for c in self.connections]
        # Keras wraps dictionaries in trackers containing non-picklable callbacks.
        metadata = dict(card_count=model.card_count, memory_size=model.memory_size, feature_spec=dict(model.feature_spec))
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
            initializer=_initialize_worker, initargs=(self.worker_connections, context.Value('i', 0), metadata))
        self.closed = False

    def _bridge(self, connection):
        while not self.stopping.is_set():
            try:
                if not connection.poll(.2): continue
                batch, version, identity, unbatched = connection.recv()
                try:
                    result = self.batcher.predict(batch, version=version, identity=identity, unbatched=unbatched)
                    message = (True, result)
                except Exception as error:
                    message = (False, f'{type(error).__name__}: {error}')
                connection.send(message)
            except (EOFError, BrokenPipeError, OSError):
                return

    async def run_games(self, *, games, seed_base, index_base=0, progress=None, **options):
        if self.closed: raise RuntimeError('Collector is closed')
        loop = asyncio.get_running_loop()
        async def one(i):
            row = await loop.run_in_executor(self.pool, _run_game,
                dict(options, seed=seed_base+i, index=index_base+i))
            if progress: progress(row)
            return row
        # gather retains game ordering even when workers finish out of order.
        return await asyncio.gather(*(one(i) for i in range(games)))

    async def close(self):
        if self.closed: return
        self.closed = True
        try:
            await asyncio.to_thread(self.pool.shutdown, wait=True, cancel_futures=True)
        finally:
            self.stopping.set()
            await asyncio.to_thread(self.bridges.shutdown, wait=True)
            for connection in self.connections + self.worker_connections: connection.close()
