"""Frozen inference replicas with direct pipes from their assigned game engines.

The learner sends weights only at collection boundaries. Per-action traffic
never passes through the learner or a shared inference queue.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import multiprocessing as mp
import os
from threading import Event
import time
import traceback


async def _settle(awaitable):
    """Drain an in-flight pipe exchange before cancellation can close its pipe."""
    task = asyncio.ensure_future(awaitable)
    try: return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.shield(task)
        raise


def weights_digest(weights):
    import numpy as np
    digest = hashlib.sha256()
    for weight in weights:
        array = np.asarray(weight)
        digest.update(str((array.shape,array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def bridge_requests(connection, batcher, stopping):
    while not stopping.is_set():
        try:
            if not connection.poll(.2): continue
            batch, version, identity, unbatched = connection.recv()
            try:
                result = batcher.predict(batch,version=version,identity=identity,unbatched=unbatched)
                message = (True,result)
            except Exception as error:
                message = (False,f'{type(error).__name__}: {error}')
            connection.send(message)
        except (EOFError,BrokenPipeError,OSError):
            return


def _replica_main(control, connections, config, weights, version, wait_ms):
    async def serve():
        import keras
        from . import model as registered_models  # noqa: F401
        from .batching import InferenceBatcher, example_batch
        network = keras.saving.deserialize_keras_object(config)
        network(example_batch(network.memory_size,network.feature_spec),training=False)
        network.set_weights(weights)
        batcher = InferenceBatcher(network,max_batch=len(connections),wait_ms=wait_ms)
        bridges = None
        stopping = Event()
        def state():
            return dict(pid=os.getpid(),engines=len(connections),version=batcher.version,
                weights_sha256=weights_digest(network.get_weights()),
                process_cpu_seconds=time.process_time(),**batcher.stats())
        try:
            await batcher.start()
            batcher.set_version(version)
            bridges = ThreadPoolExecutor(max_workers=len(connections),thread_name_prefix='replica-inference')
            for connection in connections:
                bridges.submit(bridge_requests,connection,batcher,stopping)
            control.send((True,state()))
            while True:
                command,payload = await asyncio.to_thread(control.recv)
                if command == 'close': break
                if command == 'sync':
                    new_version,new_weights = payload
                    batcher.set_version(new_version)
                    network.set_weights(new_weights)
                elif command != 'stats':
                    raise ValueError(f'Unknown inference command: {command}')
                control.send((True,state()))
        finally:
            stopping.set()
            if bridges: await asyncio.to_thread(bridges.shutdown,wait=True)
            await batcher.close()
    try:
        asyncio.run(serve())
    except BaseException:
        try: control.send((False,traceback.format_exc()))
        except (BrokenPipeError,EOFError,OSError): pass
    finally:
        for connection in connections: connection.close()
        control.close()


class InferenceProcessGroup:
    def __init__(self, connections, *, replicas=2, wait_ms=0, timeout=120):
        if type(replicas) is not int or not 1 <= replicas <= len(connections):
            raise ValueError('Inference processes must be between one and the game-engine count')
        if not 0 <= wait_ms < float('inf') or not 0 < timeout < float('inf'):
            raise ValueError('Invalid inference wait or control timeout')
        self.connections,self.replicas = connections,replicas
        self.wait_ms,self.timeout = wait_ms,timeout
        self.handles = []
        self.states = []
        self.started = self.closed = self.broken = self.busy = False
        self.version = None
        self.digest = None

    def _receive(self, handle):
        connection,process = handle
        deadline = time.monotonic()+self.timeout
        while not connection.poll(.1):
            if not process.is_alive(): raise RuntimeError(f'Inference process {process.pid} exited')
            if time.monotonic() >= deadline: raise TimeoutError('Inference process response timed out')
        try: ok,result = connection.recv()
        except (EOFError,OSError) as error: raise RuntimeError('Inference process disconnected') from error
        if not ok: raise RuntimeError(f'Inference process failed:\n{result}')
        return result

    async def _receive_all(self):
        # Drain all control requests before propagating any failure, so no RPC
        # reader can race with shutdown or consume a later command's response.
        results = await _settle(asyncio.gather(*(asyncio.to_thread(self._receive,h) for h in self.handles),
                                             return_exceptions=True))
        for result in results:
            if isinstance(result,BaseException): raise result
        return results

    def _verify(self, states, version, digest):
        if any(s['version'] != version or s['weights_sha256'] != digest for s in states):
            raise RuntimeError('Inference replicas disagree on frozen policy weights or version')
        self.states,self.version,self.digest = states,version,digest

    async def start(self, model, *, version=0):
        if self.started or self.closed or self.busy: raise RuntimeError('Inference group cannot start')
        import keras
        self.busy = True
        try:
            config = keras.saving.serialize_keras_object(model)
            weights = model.get_weights()
            digest = weights_digest(weights)
            context = mp.get_context('spawn')
            for i in range(self.replicas):
                parent,child = context.Pipe()
                assigned = self.connections[i::self.replicas]
                process = context.Process(target=_replica_main,
                    args=(child,assigned,config,weights,version,self.wait_ms),name=f'inference-replica-{i}')
                try: process.start()
                except BaseException:
                    parent.close();child.close()
                    raise
                child.close()
                self.handles.append((parent,process))
            # Only the replicas retain the server ends. A dead replica then
            # causes EOF at its game engines instead of an indefinite wait.
            for connection in self.connections: connection.close()
            self._verify(await self._receive_all(),version,digest)
            self.started = True
        except BaseException:
            self.broken = True
            await self.close()
            raise
        finally: self.busy = False
        return self

    async def _command(self, command, payload=None, *, version=None, digest=None):
        if not self.started or self.closed or self.broken or self.busy:
            raise RuntimeError('Inference group is unavailable or busy')
        self.busy = True
        async def exchange():
            for connection,process in self.handles:
                if not process.is_alive(): raise RuntimeError(f'Inference process {process.pid} exited')
                await asyncio.to_thread(connection.send,(command,payload))
            return await self._receive_all()
        try:
            self._verify(await _settle(exchange()),self.version if version is None else version,
                         self.digest if digest is None else digest)
        except BaseException:
            self.broken = True
            raise
        finally: self.busy = False

    async def sync(self, model, version):
        weights = model.get_weights()
        await self._command('sync',(version,weights),version=version,digest=weights_digest(weights))

    async def refresh_stats(self):
        await self._command('stats')
        return self.stats()

    def stats(self):
        calls = sum(s['calls'] for s in self.states)
        requests = sum(s['requests'] for s in self.states)
        histogram = {}
        for state in self.states:
            for size,count in state['batch_histogram'].items():
                histogram[size] = histogram.get(size,0)+count
        return dict(calls=calls,requests=requests,mean_batch=requests/max(calls,1),
            batch_histogram=histogram,inference_seconds=sum(s['inference_seconds'] for s in self.states),
            mean_queue_ms=sum(s['mean_queue_ms']*s['requests'] for s in self.states)/max(requests,1),
            graph_traces=[s['graph_traces'] for s in self.states],
            replica_cpu_seconds=sum(s['process_cpu_seconds'] for s in self.states),
            version=self.version,weights_sha256=self.digest,replicas=list(self.states))

    async def close(self):
        if self.closed: return
        self.closed = True
        for connection,process in self.handles:
            if process.is_alive():
                try: connection.send(('close',None))
                except (BrokenPipeError,EOFError,OSError): pass
        for connection,process in self.handles:
            await asyncio.to_thread(process.join,5)
            if process.is_alive():
                process.terminate()
                await asyncio.to_thread(process.join,5)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join)
            connection.close()
        for connection in self.connections: connection.close()
