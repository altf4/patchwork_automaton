"""One compiled inference service shared by independent game workers."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import time
import numpy as np
import tensorflow as tf

from .features import (collate, ENTITY_WIDTH, SEMANTIC_WIDTH, CANDIDATE_WIDTH,
                       CONTEXT_WIDTH)
from .history import HISTORY_WIDTH
from .policy import NeuralPolicy
from .training import Sample
from .devices import canonical_device, device_scope, assert_model_device, assert_output_device


def example_batch(memory_size, feature_spec=None):
    spec = feature_spec or dict(entities=ENTITY_WIDTH, semantics=SEMANTIC_WIDTH,
        context=CONTEXT_WIDTH, candidates=CANDIDATE_WIDTH, history=HISTORY_WIDTH)
    arrays = dict(entities=np.zeros((2, spec['entities']), np.float32),
        card_ids=np.zeros(2, np.int32), entity_mask=np.ones(2, bool),
        semantics=np.zeros((2, 1, spec['semantics']), np.float32), semantic_mask=np.ones((2, 1), bool),
        relations=np.zeros((2, 2, 4), np.float32), context=np.zeros(spec['context'], np.float32),
        candidates=np.zeros((1, spec['candidates']), np.float32), candidate_refs=np.zeros((1, 2), np.float32),
        candidate_mask=np.ones(1, bool), selection=np.zeros(spec['candidates'], np.float32),
        history=np.zeros((1, spec['history']), np.float32), history_cards=np.zeros(1, np.int32),
        history_mask=np.zeros(1, bool))
    batch = collate([arrays])
    batch.update(memory_in=np.zeros((1, memory_size), np.float32), advance_memory=np.ones((1, 1), np.float32))
    return batch


def input_signature(batch):
    # Only record widths and recurrent state widths are static. Dynamic axes
    # share one graph across batch sizes, entity counts and sequence lengths.
    fixed_last = {'entities', 'semantics', 'relations', 'context', 'candidates',
                  'selection', 'history', 'memory_in', 'advance_memory'}
    return {key: tf.TensorSpec(tuple(size if key in fixed_last and i == value.ndim-1 else None
                                    for i, size in enumerate(value.shape)), tf.as_dtype(value.dtype), name=key)
            for key, value in batch.items()}


@dataclass
class Request:
    batch: dict
    version: int
    identity: tuple
    future: asyncio.Future
    queued: float
    unbatched: bool = False


def pack_requests(requests):
    """Accept both existing batch-of-one callers and direct numerical records."""
    items = [{k: v if r.unbatched else v[0] for k, v in r.batch.items()
              if k not in ('memory_in', 'advance_memory')} for r in requests]
    batch = collate(items)
    for key in ('memory_in', 'advance_memory'):
        batch[key] = np.stack([r.batch[key] if r.unbatched else r.batch[key][0] for r in requests])
    return items, batch


class InferenceBatcher:
    def __init__(self, model, *, max_batch=16, wait_ms=5, device=None):
        if max_batch < 1 or wait_ms < 0: raise ValueError('Invalid inference batching limits')
        self.model, self.max_batch, self.wait_seconds = model, max_batch, wait_ms / 1000
        self.device = canonical_device(device) if device is not None else None
        self.variable_placements = self.output_placements = None
        self.queue = asyncio.Queue(maxsize=256)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='inference')
        self.version = 0
        self.task = None
        self.active = 0
        self.closed = False
        self.calls = self.requests = 0
        self.batch_histogram = {}
        self.inference_seconds = self.queue_seconds = 0.

    async def start(self):
        self.loop = asyncio.get_running_loop()
        def warm():
            batch = example_batch(self.model.memory_size, getattr(self.model, 'feature_spec', None))
            with device_scope(self.device) if self.device else nullcontext():
                self.model(batch, training=False)
                self.compiled = tf.function(self.model.call, input_signature=[input_signature(batch)], autograph=False)
                outputs = self.compiled(batch)
            if self.device:
                self.variable_placements = assert_model_device(self.model, self.device)
                self.output_placements = assert_output_device(outputs, self.device)
        await self.loop.run_in_executor(self.executor, warm)
        self.task = asyncio.create_task(self._serve())
        return self

    def predict(self, batch, *, version, identity, unbatched=False):
        # Called only by a game-policy thread, never by the event loop.
        pending = asyncio.run_coroutine_threadsafe(self._request(batch, version, identity, unbatched), self.loop)
        try: return pending.result(timeout=120)
        except BaseException:
            pending.cancel()
            raise

    async def _request(self, batch, version, identity, unbatched=False):
        if self.closed or version != self.version: raise RuntimeError('Stale model version or closed batcher')
        future = self.loop.create_future()
        await self.queue.put(Request(batch, version, identity, future, time.monotonic(), unbatched))
        return await future

    def set_version(self, version):
        if self.active or not self.queue.empty(): raise RuntimeError('Cannot update model with pending inference')
        self.version = version

    def stats(self):
        return dict(calls=self.calls, requests=self.requests, mean_batch=self.requests/max(self.calls,1),
                    batch_histogram=self.batch_histogram.copy(), inference_seconds=self.inference_seconds,
                    mean_queue_ms=1000*self.queue_seconds/max(self.requests,1),
                    graph_traces=self.compiled.experimental_get_tracing_count(), device=self.device,
                    model_variable_devices=sorted(set(self.variable_placements.values())) if self.variable_placements else None,
                    output_placements=self.output_placements)

    def restore_stats(self, stats):
        self.calls, self.requests = stats['calls'], stats['requests']
        self.batch_histogram = {int(k):v for k,v in stats['batch_histogram'].items()}
        self.inference_seconds = stats['inference_seconds']
        self.queue_seconds = stats['mean_queue_ms'] * self.requests / 1000

    def _infer(self, requests):
        items, batch = pack_requests(requests)
        with device_scope(self.device) if self.device else nullcontext():
            outputs = self.compiled(batch)
        if self.device:
            self.output_placements = assert_output_device(outputs, self.device)
        output = {k: np.asarray(v) for k,v in outputs.items()}
        return [{k: (v[i:i+1, :len(items[i]['candidates'])] if k == 'logits' else v[i:i+1])
                 for k,v in output.items()} for i in range(len(requests))]

    async def _serve(self):
        while True:
            first = await self.queue.get()
            if first is None: return
            requests = [first]
            self.active = 1
            deadline = self.loop.time() + self.wait_seconds
            while len(requests) < self.max_batch:
                try:
                    if not self.queue.empty(): request = self.queue.get_nowait()
                    else:
                        remaining = deadline - self.loop.time()
                        if remaining <= 0: break
                        request = await asyncio.wait_for(self.queue.get(), remaining)
                    if request is None:
                        self.queue.put_nowait(None)
                        break
                    requests.append(request)
                except asyncio.TimeoutError: break
            requests = [r for r in requests if not r.future.cancelled()]
            try:
                if any(r.version != self.version for r in requests): raise RuntimeError('Stale queued model version')
                if requests:
                    start = time.monotonic()
                    self.queue_seconds += sum(start-r.queued for r in requests)
                    outputs = await self.loop.run_in_executor(self.executor, self._infer, requests)
                    self.inference_seconds += time.monotonic()-start
                    self.calls += 1; self.requests += len(requests)
                    self.batch_histogram[len(requests)] = self.batch_histogram.get(len(requests),0)+1
                    for request, output in zip(requests,outputs):
                        if not request.future.done(): request.future.set_result(output)
            except Exception as error:
                for request in requests:
                    if not request.future.done(): request.future.set_exception(error)
            finally: self.active = 0

    async def close(self):
        self.closed = True
        if self.task:
            # All worker actions must have been awaited before shutdown.
            await self.queue.put(None)
            await self.task
        self.executor.shutdown(wait=True)


class BatchedPolicy(NeuralPolicy):
    def __init__(self, model, batcher, *, version, seed, sink=None):
        super().__init__(model, seed=seed)
        self.batcher, self.version, self.sink = batcher, version, sink
        self.selection_details = []

    def act(self, obs):
        self.identity = (obs.decision.token.session, str(obs.game), str(obs.actor),
                         obs.decision.token.generation)
        self.selection_details = []
        return super().act(obs)

    def _forward(self, batch):
        self.last_output = self.batcher.predict(batch, version=self.version, identity=self.identity)
        return self.last_output

    def _select(self, arrays, state, *, advance, stage, records):
        result = super()._select(arrays,state,advance=advance,stage=stage,records=records)
        index, _, value = result
        logits = self.last_output['logits'][0].astype(np.float64) / self.temperature
        logits[~arrays['candidate_mask']] = -np.inf
        probabilities = np.exp(logits-np.max(logits)); probabilities /= probabilities.sum()
        self.selection_details.append(dict(stage=stage,index=index,probabilities=probabilities.tolist()))
        if self.sink:
            # Synchronous serialization precedes any mutation of argument arrays.
            behavior = np.where(arrays['candidate_mask'], logits, -1.e9)
            behavior -= np.max(behavior)
            behavior -= np.log(np.exp(behavior).sum())
            self.sink.write(Sample(arrays,state,advance,index,value,behavior.astype(np.float32)))
        return result
