"""Bounded-memory PPO over frozen full-game trajectories with terminal returns.

Recurrent states are carried during collection and detached during updates.
This is one-step truncated recurrence, not sequence backpropagation.
"""
import itertools
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import numpy as np
from .ppo import PPOTrainer, normalized_advantages, value_prediction_diagnostics
from .features import collate
from .batching import example_batch
from .experiment import read_trajectory
from .rollout_cache import IndexedTrajectories, RolloutCache, retained_bytes
from .ordered_jobs import ordered_jobs


class StreamingPPOTrainer(PPOTrainer):
    def __init__(self, model, *, seed=0, cache_bytes=256 * 1024**2, concurrent_updates=False,
                 prefetch_batches=0, measurement_workers=1, **kwargs):
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError('Rollout cache budget must be nonnegative integer bytes')
        if (type(prefetch_batches) is not int or prefetch_batches < 0
                or type(measurement_workers) is not int or measurement_workers < 1):
            raise ValueError('Invalid prefetch depth or measurement worker count')
        model(example_batch(model.memory_size, model.feature_spec), training=False)
        super().__init__(model, **kwargs)
        if concurrent_updates and not self.separate_value:
            raise ValueError('Concurrent updates require separate actor and critic networks')
        if self.separate_value:
            if not getattr(model, 'deterministic_forward', False):
                raise ValueError('Independent streaming updates require deterministic forwards')
            if {id(v) for v in model.actor.variables} & {id(v) for v in model.critic.variables}:
                raise ValueError('Actor and critic must not share variables')
        self.concurrent_updates = bool(concurrent_updates)
        if measurement_workers > 1 and not getattr(model,'deterministic_forward',False):
            raise ValueError('Parallel measurements require deterministic forwards')
        self.prefetch_batches, self.measurement_workers = prefetch_batches, measurement_workers
        self.rng = np.random.default_rng(seed)
        self.cache_bytes = cache_bytes

    def compile(self):
        batch = example_batch(self.model.memory_size, self.model.feature_spec)
        zero = np.zeros(1, np.float32)
        self.measure(batch, np.zeros(1, np.int32), zero, zero, zero)
        self.drift(batch, np.zeros((1, 1), np.float32))
        self.measure_with_drift(batch, np.zeros(1, np.int32), zero, zero, zero,
                                np.zeros((1, 1), np.float32))
        if self.separate_value:
            self.measure_actor_with_drift(batch, np.zeros(1, np.int32), zero, zero, zero,
                                          np.zeros((1, 1), np.float32))

    def update_files(self, trajectories):
        # Random access permits a genuine shuffle across games. The spill stays
        # empty when all trajectories fit in the decoded-record cache.
        with tempfile.TemporaryFile() as spill:
            return self._update_files(list(trajectories), spill)

    def _update_files(self, trajectories, spill):
        trajectories = list(trajectories)
        # Caches live for one frozen rollout only. Larger rollouts retain the
        # streaming fallback; they never force all trajectories into memory.
        records_cache = RolloutCache(self.cache_bytes * 3 // 4)
        packed_cache = RolloutCache(self.cache_bytes - records_cache.limit)
        offsets = []; values = []; returns = []; choices = []; advancing = []
        for index, trajectory in enumerate(trajectories):
            if trajectory['reward'] not in (-1., 0., 1.): raise ValueError('Expected terminal win/loss/draw')
            offsets.append(len(values)); seen = 0
            records = [] if records_cache.limit else None
            record_bytes = 128
            for sample in read_trajectory(trajectory['path']):
                logp = sample.log_probs
                mask = sample.arrays['candidate_mask']
                if (logp is None or np.shape(logp) != np.shape(mask) or not np.isfinite(logp).all()
                        or not np.isclose(np.exp(logp).sum(), 1., atol=1e-5)):
                    raise ValueError('Missing or invalid frozen behavior probabilities')
                if not 0 <= sample.action < len(mask) or not mask[sample.action]:
                    raise ValueError('Invalid rollout action')
                seen += 1; values.append(sample.value); returns.append(trajectory['reward']); choices.append(mask.sum() > 1)
                advancing.append(sample.advance)
                if records is not None:
                    record_bytes += retained_bytes(sample) + 16
                    if records_cache.bytes + record_bytes + 256 <= records_cache.limit:
                        records.append(sample)
                    else:
                        records = None
            if seen != trajectory['samples']: raise ValueError('Incomplete trajectory file')
            if records is not None:
                records_cache.put(index, tuple(records), size=record_bytes)
        if not values or not np.isfinite(values).all(): raise ValueError('Empty or invalid rollout')
        advantages = normalized_advantages(returns, values, np.asarray(choices), self.advantage_scale_floor)
        count = len(values)
        offsets = np.asarray(offsets, np.int64)
        returns = np.asarray(returns, np.float32)
        indexed = IndexedTrajectories(trajectories, records_cache, read_trajectory, spill)
        packing_lock = threading.Lock()

        def pack(chunk):
            # Both optimizer streams share the bounded cache and seekable file.
            # Hold the lock only for CPU preparation, never during TF execution.
            with packing_lock:
                key = tuple(map(int, chunk))
                packed = packed_cache.get(key)
                if packed is not None:
                    return packed
                trajectory_ids = np.searchsorted(offsets, chunk, side='right') - 1
                records = [indexed.get(int(i), int(p-offsets[i])) for i, p in zip(trajectory_ids, chunk)]
                inputs = collate([s.arrays for s in records])
                inputs['memory_in'] = np.stack([s.memory for s in records])
                inputs['advance_memory'] = np.asarray([[float(s.advance)] for s in records], np.float32)
                old = np.full(inputs['candidate_mask'].shape, -1.e9, np.float32)
                for row, sample in enumerate(records): old[row, :len(sample.log_probs)] = sample.log_probs
                actions = np.asarray([s.action for s in records], np.int32)
                chosen = old[np.arange(len(records)), actions]
                packed = (inputs, actions, returns[chunk], chosen, advantages[chunk], old)
                if packed_cache.bytes < packed_cache.limit:
                    if not packed_cache.put(key, packed):
                        # Stop admission at the first full prefix, avoiding
                        # repeated copies of batches that cannot fit the cache.
                        packed_cache.limit = packed_cache.bytes
                return packed

        def chunks(order):
            stream = iter(order)
            while chunk := list(itertools.islice(stream, self.batch_size)):
                yield chunk

        @contextmanager
        def batches(order):
            if self.prefetch_batches:
                with ordered_jobs(pack,chunks(order),workers=1,capacity=self.prefetch_batches,
                                  name='ppo-prefetch') as prepared:
                    yield prepared
            else:
                yield map(pack,chunks(order))

        def measure(*, actor_only=False):
            sums = np.zeros(5); choice_count = value_count = 0; exact_sum = worst = 0.
            forward = self.measure_actor_with_drift if actor_only else self.measure_with_drift
            def evaluate(chunk):
                inputs, actions, rewards, chosen, adv, old = pack(chunk)
                metrics, drift = map(np.asarray, forward(inputs, actions, rewards, chosen, adv, old))
                n = int((inputs['candidate_mask'].sum(axis=1) > 1).sum())
                v = float(inputs['advance_memory'].sum())
                return metrics, drift, n, v
            # The actor is frozen during actor-only checks. Both networks are
            # frozen during full scans. Join every reader before any mutation;
            # reduce in canonical order to preserve the exact acceptance test.
            jobs = (ordered_jobs(evaluate,chunks(range(count)),workers=self.measurement_workers,
                                 capacity=self.measurement_workers,name='ppo-measure')
                    if self.measurement_workers > 1 else nullcontext(map(evaluate,chunks(range(count)))))
            with jobs as results:
                for metrics, drift, n, v in results:
                    sums += metrics[1:] * np.asarray([n, v, n, n, n])
                    choice_count += n; value_count += v
                    exact_sum += float(drift[0]) * n; worst = max(worst, float(drift[1]))
            denominator = np.asarray([choice_count, value_count, choice_count, choice_count, choice_count])
            terms = np.divide(sums, denominator, out=np.zeros_like(sums), where=denominator > 0)
            loss = terms[0] + self.value_weight * terms[1] - self.entropy_weight * terms[2]
            return np.r_[loss, terms], np.asarray([exact_sum / max(choice_count, 1), worst])

        steps = rejected = 0; norm = 0.; stopped = False; largest = np.zeros(2)
        metrics, exact = measure()
        value_loss_before = float(metrics[2])
        # Draw the same critic permutations, in the same order, before the
        # actor consumes RNG. Only the calling thread ever touches self.rng.
        critic_orders = iter([self.rng.permutation(count) for _ in range(self.epochs)]) if self.separate_value else None
        def critic_batches():
            order = next(critic_orders) if critic_orders is not None else self.rng.permutation(count)
            with batches(order) as prepared:
                for inputs, _, rewards, *_ in prepared:
                    yield inputs, rewards
        # The executor joins even when either side fails, before the spill file
        # is closed or a caller can save a checkpoint. Optimizers and variables
        # are disjoint; advantages and behavior memories stay frozen.
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ppo-critic') if self.concurrent_updates else nullcontext()
        with pool as executor:
            future = executor.submit(self.fit_critic, critic_batches) if executor else None
            critic_stats = {} if future else self.fit_critic(critic_batches)
            for epoch in range(self.epochs):
                saved = self.snapshot(); rate = self.learning_rate
                order = self.rng.permutation(count); accepted = False
                for attempt in range(self.backtracks+1):
                    self.optimizer.learning_rate.assign(rate * .5**attempt)
                    epoch_steps = 0
                    with batches(order) as prepared:
                        for inputs, actions, rewards, chosen, adv, _ in prepared:
                            if future and future.done(): future.result()
                            norm = float(self.step(inputs, actions, rewards, chosen, adv)); epoch_steps += 1
                    proposed, drift = measure(actor_only=self.separate_value)
                    if (np.isfinite(proposed).all() and np.isfinite(drift).all()
                            and drift[0] <= self.target_kl and drift[1] <= self.max_state_kl):
                        metrics, exact = proposed, drift; largest = np.maximum(largest, drift)
                        steps += epoch_steps; accepted = True; break
                    self.restore(saved); rejected += 1
                if not accepted: stopped = True; break
            if future: critic_stats = future.result()
        if self.separate_value:
            metrics, exact = measure()
            if not np.isfinite(metrics).all() or not np.isfinite(exact).all():
                raise FloatingPointError('Nonfinite final actor/critic metrics')
        return dict(zip(('loss','policy_loss','value_loss','entropy','approx_kl','clip_fraction'), map(float,metrics)),
            samples=count,optimizer_steps=int(self.optimizer.iterations),accepted_steps=steps,
            rejected_epochs=rejected,early_stop=stopped,gradient_norm=norm,
            exact_kl=float(exact[0]),max_state_kl=float(exact[1]),
            largest_accepted_kl=float(largest[0]),largest_accepted_state_kl=float(largest[1]),
            reward_std=float(np.std(returns)),advantage_std=float(np.std(advantages)),
            cache=dict(budget_bytes=self.cache_bytes, retained_bytes=records_cache.bytes+packed_cache.bytes,
                trajectories=len(records_cache.entries), trajectory_hits=records_cache.hits,
                batches=len(packed_cache.entries), batch_hits=packed_cache.hits,
                spill_bytes=indexed.disk_bytes),
            sample_order='global_shuffle_v1',
            update_schedule='concurrent' if self.concurrent_updates else 'sequential',
            pipeline=dict(prefetch_batches=self.prefetch_batches,measurement_workers=self.measurement_workers),
            behavior_critic=value_prediction_diagnostics(returns, values, advancing),
            recurrent_updates='detached behavior states',terminal_rewards_only=True,
            value_loss_before=value_loss_before, **critic_stats)
