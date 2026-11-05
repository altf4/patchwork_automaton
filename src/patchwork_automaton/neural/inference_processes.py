"""Frozen inference replicas with direct pipes from their assigned game engines.

The learner sends weights only at collection boundaries. Per-action traffic
never passes through the learner or a shared inference queue.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import multiprocessing as mp
import os
import pickle
from threading import Event
import time
import traceback

CPU_INTRA_THREADS_ENV = 'PATCHWORK_CPU_INFERENCE_INTRA_THREADS'


def _cpu_intra_threads(device, requested=None):
    """Resolve a replica-only override without changing the learner's runtime."""
    from .devices import canonical_device
    if canonical_device(device) != '/CPU:0':
        if requested is not None:
            raise ValueError('CPU inference thread override requires a CPU replica')
        return None
    if requested is None:
        raw = os.environ.get(CPU_INTRA_THREADS_ENV, '1')
        try:
            requested = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f'{CPU_INTRA_THREADS_ENV} must be a positive integer') from None
    if type(requested) is not int or requested < 1:
        raise ValueError('CPU inference intra-op threads must be a positive integer')
    return requested


def _configure_replica_threads(device, cpu_intra_threads):
    """Configure only a fresh CPU replica, before TensorFlow initializes."""
    from .devices import canonical_device
    if canonical_device(device) != '/CPU:0':
        return
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(cpu_intra_threads)
    except RuntimeError as error:
        raise RuntimeError('CPU inference threads must be configured before TensorFlow runtime initialization') from error


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


def _versions_digest(digests):
    return hashlib.sha256(repr(sorted(digests.items())).encode()).hexdigest()


def bridge_requests(connection, batcher, stopping):
    from ..rl.inference_transport import send_message, recv_message
    session, sequence = None, 0
    try:
        while not stopping.is_set():
            if not connection.poll(.2): continue
            request = recv_message(connection, timeout=120)
            if (not isinstance(request, tuple) or len(request) not in (3, 6)
                    or request[0] not in ('inference-v1', 'reset-v1')):
                raise ValueError('Invalid inference request envelope')
            request_id = request[1]
            if (not isinstance(request_id, tuple) or len(request_id) != 2
                    or not isinstance(request_id[0], str) or not request_id[0]
                    or type(request_id[1]) is not int or request_id[1] != sequence + 1
                    or session is not None and request_id[0] != session):
                raise ValueError('Invalid inference request sequence')
            session, sequence = request_id
            if request[0] == 'reset-v1':
                if len(request) != 3:
                    raise ValueError('Invalid inference reset envelope')
                send_message(connection, request, timeout=120)
                continue
            if len(request) != 6:
                raise ValueError('Invalid inference prediction envelope')
            _, _, version, identity, unbatched, batch = request
            try:
                result = batcher.predict(batch,version=version,identity=identity,unbatched=unbatched)
                message = (*request[:4], True, result)
            except Exception as error:
                message = (*request[:4], False, f'{type(error).__name__}: {error}')
            send_message(connection, message, timeout=120)
    except (EOFError, OSError, TimeoutError, ValueError, pickle.UnpicklingError):
        return
    finally:
        connection.close()


class _VersionBatchers:
    """Route immutable checkpoint versions inside one inference process."""
    version = None

    def __init__(self, batchers):
        self.batchers = batchers

    def predict(self, batch, *, version, **options):
        if version not in self.batchers:
            raise RuntimeError(f'Unknown frozen opponent version: {version}')
        return self.batchers[version].predict(batch, version=version, **options)

    def stats(self):
        rows = [b.stats() for b in self.batchers.values()]
        result = dict(rows[0]);histogram = {}
        for row in rows:
            for size, count in row['batch_histogram'].items():
                histogram[size] = histogram.get(size, 0) + count
        for key in ('calls', 'requests', 'inference_seconds', 'graph_traces'):
            result[key] = sum(row[key] for row in rows)
        result.update(mean_batch=result['requests']/max(result['calls'], 1), batch_histogram=histogram,
            mean_queue_ms=sum(row['mean_queue_ms']*row['requests'] for row in rows)/max(result['requests'], 1),
            version_stats={n:b.stats() for n,b in self.batchers.items()})
        return result

    async def close(self):
        await asyncio.gather(*(b.close() for b in self.batchers.values()))


def _replica_main(control, connections, config, weights, version, wait_ms, device, cpu_intra_threads=None,
                  version_weights=None):
    async def serve():
        import keras
        from . import model as registered_models  # noqa: F401
        from . import compact_standard_model as registered_compact_models  # noqa: F401
        from . import compact_identity_model as registered_identity_models  # noqa: F401
        from .batching import InferenceBatcher, model_example_batch
        from .devices import device_scope, assert_model_device
        import tensorflow as tf
        batchers = {};networks = {}
        try:
            for number, frozen_weights in (version_weights or {version: weights}).items():
                with device_scope(device):
                    network = keras.saving.deserialize_keras_object(config)
                    network(model_example_batch(network),training=False)
                    network.set_weights(frozen_weights)
                assert_model_device(network, device)
                service = InferenceBatcher(network,max_batch=len(connections),wait_ms=wait_ms,device=device)
                networks[number] = network;batchers[number] = service
                await service.start();service.set_version(number)
        except BaseException:
            await asyncio.gather(*(b.close() for b in batchers.values()),return_exceptions=True)
            raise
        batcher = _VersionBatchers(batchers) if version_weights is not None else batchers[version]
        digests = {number: weights_digest(network.get_weights()) for number, network in networks.items()}
        digest = _versions_digest(digests) if version_weights is not None else digests[version]
        bridges = None
        stopping = Event()
        def state():
            return dict(pid=os.getpid(),engines=len(connections),version=batcher.version,
                weights_sha256=digest if version_weights is not None else weights_digest(network.get_weights()),
                versions=digests if version_weights is not None else None,
                visible_gpus=[gpu.name for gpu in tf.config.get_visible_devices('GPU')],
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                intra_op_threads=tf.config.threading.get_intra_op_parallelism_threads(),
                inter_op_threads=tf.config.threading.get_inter_op_parallelism_threads(),
                tf_num_intraop_threads_env=os.environ.get('TF_NUM_INTRAOP_THREADS'),
                tf_num_interop_threads_env=os.environ.get('TF_NUM_INTEROP_THREADS'),
                process_cpu_seconds=time.process_time(),**batcher.stats())
        try:
            bridges = ThreadPoolExecutor(max_workers=len(connections),thread_name_prefix='replica-inference')
            for connection in connections:
                bridges.submit(bridge_requests,connection,batcher,stopping)
            control.send((True,state()))
            while True:
                command,payload = await asyncio.to_thread(control.recv)
                if command == 'close': break
                if command == 'sync':
                    if version_weights is not None:raise RuntimeError('Frozen checkpoint bank cannot be mutated')
                    new_version,new_weights = payload
                    batcher.set_version(new_version)
                    with device_scope(device):
                        network.set_weights(new_weights)
                elif command != 'stats':
                    raise ValueError(f'Unknown inference command: {command}')
                control.send((True,state()))
        finally:
            stopping.set()
            if bridges: await asyncio.to_thread(bridges.shutdown,wait=True)
            await batcher.close()
    async def serve_with_timer():
        loop = asyncio.get_running_loop()
        def tick():
            if not loop.is_closed():
                loop.call_later(.002, tick)
        loop.call_later(.002, tick)
        await serve()

    try:
        from .devices import configure_runtime, canonical_device
        if canonical_device(device) == '/CPU:0':
            # Hide learner GPUs before the threading helper imports TensorFlow.
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            _configure_replica_threads(device, _cpu_intra_threads(device, cpu_intra_threads))
        configure_runtime(device, hide_gpu=canonical_device(device) == '/CPU:0')
        asyncio.run(serve_with_timer() if os.environ.get('PATCHWORK_ARENA_POLL_INFERENCE') == '1'
                    else serve())
    except BaseException:
        try: control.send((False,traceback.format_exc()))
        except (BrokenPipeError,EOFError,OSError): pass
    finally:
        for connection in connections: connection.close()
        control.close()


class InferenceProcessGroup:
    def __init__(self, connections, *, replicas=2, wait_ms=0, timeout=120, device='cpu', cpu_intra_threads=None):
        if type(replicas) is not int or not 1 <= replicas <= len(connections):
            raise ValueError('Inference processes must be between one and the game-engine count')
        if not 0 <= wait_ms < float('inf') or not 0 < timeout < float('inf'):
            raise ValueError('Invalid inference wait or control timeout')
        self.connections,self.replicas = connections,replicas
        self.wait_ms,self.timeout = wait_ms,timeout
        from .devices import canonical_device
        self.device = canonical_device(device)
        self.cpu_intra_threads = _cpu_intra_threads(self.device, cpu_intra_threads)
        self.handles = []
        self.states = []
        self.started = self.closed = self.broken = self.busy = False
        self.version = None
        self.digest = None
        self.version_digests = None

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
        if self.version_digests is not None and any(s.get('versions') != self.version_digests for s in states):
            raise RuntimeError('Inference replicas disagree on frozen checkpoint bank')
        if any(s['device'] != self.device for s in states):
            raise RuntimeError('Inference replicas disagree on requested neural device')
        if self.cpu_intra_threads is not None and any(s['intra_op_threads'] != self.cpu_intra_threads for s in states):
            raise RuntimeError('CPU inference replicas disagree on intra-op thread count')
        self.states,self.version,self.digest = states,version,digest

    async def start(self, model, *, version=0, version_weights=None):
        if self.started or self.closed or self.busy: raise RuntimeError('Inference group cannot start')
        import keras
        self.busy = True
        try:
            config = keras.saving.serialize_keras_object(model)
            weights = model.get_weights()
            digest = weights_digest(weights)
            if version_weights is not None:
                if not version_weights or len(version_weights)>4 or any(type(n) is not int or n<0 for n in version_weights):
                    raise ValueError('Frozen opponent bank requires one to four nonnegative checkpoint versions')
                self.version_digests = {n:weights_digest(w) for n,w in version_weights.items()}
                version = None;digest = _versions_digest(self.version_digests)
            context = mp.get_context('spawn')
            for i in range(self.replicas):
                parent,child = context.Pipe()
                assigned = self.connections[i::self.replicas]
                process = context.Process(target=_replica_main,
                    args=(child,assigned,config,weights,version,self.wait_ms,self.device,self.cpu_intra_threads,version_weights),name=f'inference-replica-{i}')
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
        if self.version_digests is not None:raise RuntimeError('Frozen checkpoint bank cannot be mutated')
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
            version=self.version,weights_sha256=self.digest,device=self.device,replicas=list(self.states))

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
