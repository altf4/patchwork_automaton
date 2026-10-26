"""Bounded-memory PPO over frozen full-game trajectories with terminal returns.

Recurrent states are carried during collection and detached during updates.
This is one-step truncated recurrence, not sequence backpropagation.
"""
import itertools
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext, ExitStack
import numpy as np
from .ppo import PPOTrainer, normalized_advantages, value_prediction_diagnostics
from .features import collate
from .batching import example_batch
from .experiment import read_trajectory
from .rollout_cache import IndexedTrajectories, RolloutCache, retained_bytes
from .ordered_jobs import ordered_jobs


def _guard_status(proposed, drift, target_kl, max_state_kl):
    """Describe the existing acceptance predicate without altering its limits."""
    finite = bool(np.isfinite(proposed).all() and np.isfinite(drift).all())
    reasons = []
    if not finite: reasons.append('nonfinite')
    if drift[0] > target_kl: reasons.append('mean_kl')
    if drift[1] > max_state_kl: reasons.append('max_state_kl')
    accepted = bool(finite and drift[0] <= target_kl and drift[1] <= max_state_kl)
    return accepted, finite, reasons


def _json_number(value):
    return float(value) if np.isfinite(value) else None


class StreamingPPOTrainer(PPOTrainer):
    def __init__(self, model, *, seed=0, cache_bytes=256 * 1024**2, concurrent_updates=False,
                 prefetch_batches=0, measurement_workers=1, guard_audit=False,
                 guard_observer=None, numerical_prepacking=False, cache_policy_batches=False,
                 policy_cache_device=None, policy_cache_device_bytes=0,
                 actor_epoch_cache_device=None, actor_epoch_cache_bytes=0, **kwargs):
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError('Rollout cache budget must be nonnegative integer bytes')
        if (type(prefetch_batches) is not int or prefetch_batches < 0
                or type(measurement_workers) is not int or measurement_workers < 1):
            raise ValueError('Invalid prefetch depth or measurement worker count')
        if guard_observer is not None and not callable(guard_observer):
            raise ValueError('Guard observer must be callable')
        if type(policy_cache_device_bytes) is not int or policy_cache_device_bytes < 0:
            raise ValueError('Device policy cache budget must be nonnegative integer bytes')
        if policy_cache_device is None and policy_cache_device_bytes:
            raise ValueError('Device policy cache budget requires an explicit device')
        if type(actor_epoch_cache_bytes) is not int or actor_epoch_cache_bytes < 0:
            raise ValueError('Actor epoch cache budget must be nonnegative integer bytes')
        if actor_epoch_cache_device is None and actor_epoch_cache_bytes:
            raise ValueError('Actor epoch cache budget requires an explicit device')
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
        self.guard_audit, self.guard_observer = bool(guard_audit), guard_observer
        self.numerical_prepacking = bool(numerical_prepacking)
        self.cache_policy_batches = bool(cache_policy_batches)
        self.policy_cache_device = policy_cache_device
        self.policy_cache_device_bytes = policy_cache_device_bytes
        self.actor_epoch_cache_device = actor_epoch_cache_device
        self.actor_epoch_cache_bytes = actor_epoch_cache_bytes

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
        with tempfile.TemporaryFile() as spill, ExitStack() as resources:
            return self._update_files(list(trajectories), spill, resources)

    def warmup_critic_files(self, trajectories, *, max_steps=None):
        """Fit fresh terminal rollouts while preserving the actor and its Adam.

        Uses the same bounded caches, global shuffle and detached behavior
        memories as PPO. The critic's shuffle advances the trainer RNG. A
        positive max_steps overrides the configured value budget for this call
        only; None retains that budget. No actor step or rollback is performed.
        """
        if not self.separate_value:
            raise ValueError('Critic warmup requires separate actor and critic networks')
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError('Critic warmup max_steps must be a positive integer')
        saved_limit = self.value_max_steps
        try:
            if max_steps is not None:
                self.value_max_steps = max_steps
            with tempfile.TemporaryFile() as spill, ExitStack() as resources:
                result = self._update_files(list(trajectories), spill, resources, critic_only=True)
            return dict(result, phase='critic_warmup', critic_only=True, accepted_actor_steps=0,
                        update_schedule='critic_only')
        finally:
            self.value_max_steps = saved_limit

    def _update_files(self, trajectories, spill, resources, *, critic_only=False):
        if self.guard_audit and not getattr(self.model, 'deterministic_forward', False):
            raise ValueError('Guard auditing requires deterministic read-only forwards')
        trajectories = list(trajectories)
        # Caches live for one frozen rollout only. Larger rollouts retain the
        # streaming fallback; they never force all trajectories into memory.
        records_cache = RolloutCache(self.cache_bytes * 3 // 4)
        packed_cache = RolloutCache(self.cache_bytes - records_cache.limit)
        from .packed_rollouts import NumericRollout, PolicyBatchCache, DevicePolicyBatchCache
        numeric = NumericRollout() if self.numerical_prepacking else None
        policy_cache = PolicyBatchCache() if self.cache_policy_batches else None
        if numeric: resources.callback(numeric.close)
        if policy_cache: resources.callback(policy_cache.close)
        device_cache = (DevicePolicyBatchCache(device=self.policy_cache_device,
                        max_bytes=self.policy_cache_device_bytes)
                        if self.policy_cache_device is not None else None)
        if device_cache: resources.callback(device_cache.close)
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
                if numeric: numeric.append(sample)
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
        actor_values = values if self.advantage_baseline == 'critic' else np.zeros(len(values), np.float32)
        advantages = normalized_advantages(returns, actor_values, np.asarray(choices), self.advantage_scale_floor)
        count = len(values)
        if numeric: numeric.finish()
        offsets = np.asarray(offsets, np.int64)
        returns = np.asarray(returns, np.float32)
        indexed = IndexedTrajectories(trajectories, records_cache, read_trajectory, spill)
        resources.callback(indexed.close)
        packing_lock = threading.Lock()

        def retain_policy(key, packed):
            if device_cache:
                converted = device_cache.put(key, packed)
                # The initial scan uses the very tensors just uploaded. Avoid
                # a second transfer or a duplicate host-cache entry for them.
                if converted is not None: return converted
            if policy_cache: policy_cache.put(key, packed)
            return packed

        def pack(chunk, *, policy=False):
            # Both optimizer streams share the bounded cache and seekable file.
            # Hold the lock only for CPU preparation, never during TF execution.
            with packing_lock:
                key = tuple(map(int, chunk))
                if policy and device_cache:
                    hit = device_cache.get(key)
                    if hit is not None: return hit
                if policy and policy_cache:
                    hit = policy_cache.get(key)
                    if hit is not None: return hit
                packed = packed_cache.get(key)
                if packed is not None:
                    return retain_policy(key, packed) if policy else packed
                if numeric:
                    inputs = numeric.gather(chunk)
                    old = inputs.pop('_old'); actions = inputs.pop('_action')[:, 0]
                    chosen = old[np.arange(len(chunk)), actions]
                    packed = (inputs, actions, returns[chunk], chosen, advantages[chunk], old)
                else:
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
                return retain_policy(key, packed) if policy else packed

        def chunks(order):
            stream = iter(order)
            while chunk := list(itertools.islice(stream, self.batch_size)):
                yield chunk

        @contextmanager
        def batches(order, *, actor_cache=None):
            def prepare(chunk):
                if actor_cache is None: return pack(chunk)
                key = tuple(map(int, chunk))
                hit = actor_cache.get(key)
                if hit is not None: return hit
                packed = pack(chunk)
                converted = actor_cache.put(key, packed)
                # The first optimizer pass immediately consumes these newly
                # uploaded tensors. Only frozen inputs are cached across the
                # same epoch's backtracks; critic and policy scans use pack.
                return converted if converted is not None else packed
            if self.prefetch_batches:
                with ordered_jobs(prepare,chunks(order),workers=1,capacity=self.prefetch_batches,
                                  name='ppo-prefetch') as prepared:
                    yield prepared
            else:
                yield map(prepare,chunks(order))

        actor_epoch_cache_stats = []

        @contextmanager
        def actor_epoch_inputs(epoch):
            if self.actor_epoch_cache_device is None:
                yield None
                return
            cache = DevicePolicyBatchCache(device=self.actor_epoch_cache_device,
                                          max_bytes=self.actor_epoch_cache_bytes)
            row = dict(epoch=epoch+1, device=cache.device, budget_bytes=cache.max_bytes)
            try:
                yield cache
            finally:
                # Every nested batches() context has already joined/cancelled
                # its prefetcher on success, early stop and exceptions. The
                # critic never references this epoch-local cache.
                row.update(logical_bytes=cache.bytes, uploads=cache.uploads,
                           tensors=cache.uploaded_tensors, hits=cache.hits,
                           fallback_batches=cache.fallback_batches,
                           fallback_bytes=cache.fallback_bytes,
                           unique_input_bytes=cache.bytes+cache.fallback_bytes)
                cache.close()
                row['released'] = cache.closed and cache.bytes == 0 and not cache.entries
                actor_epoch_cache_stats.append(row)

        def measure(*, actor_only=False):
            sums = np.zeros(5); choice_count = value_count = 0; exact_sum = worst = 0.
            forward = self.measure_actor_with_drift if actor_only else self.measure_with_drift
            def prepared_policy_batches():
                # Input iteration remains on ordered_jobs' calling thread.
                # This makes device admission a deterministic canonical prefix
                # even when forwards execute on several measurement workers.
                for chunk in chunks(range(count)):
                    packed = pack(chunk, policy=True)
                    yield packed, device_cache.counts(tuple(map(int, chunk)))

            def evaluate(item):
                if device_cache:
                    packed, (n, v) = item
                else:
                    packed = pack(item, policy=True)
                    n = int((packed[0]['candidate_mask'].sum(axis=1) > 1).sum())
                    v = float(packed[0]['advance_memory'].sum())
                # Device contexts are thread-local and must be entered inside
                # every worker, including over-budget CPU-input fallbacks.
                scope = device_cache._tf.device(device_cache.device) if device_cache else nullcontext()
                with scope:
                    metrics, drift = map(np.asarray, forward(*packed))
                return metrics, drift, n, v
            # The actor is frozen during actor-only checks. Both networks are
            # frozen during full scans. Join every reader before any mutation;
            # reduce in canonical order to preserve the exact acceptance test.
            items = prepared_policy_batches() if device_cache else chunks(range(count))
            jobs = (ordered_jobs(evaluate,items,workers=self.measurement_workers,
                                 capacity=self.measurement_workers,name='ppo-measure')
                    if self.measurement_workers > 1 else nullcontext(map(evaluate,items)))
            with jobs as results:
                for metrics, drift, n, v in results:
                    sums += metrics[1:] * np.asarray([n, v, n, n, n])
                    choice_count += n; value_count += v
                    exact_sum += float(drift[0]) * n; worst = max(worst, float(drift[1]))
            denominator = np.asarray([choice_count, value_count, choice_count, choice_count, choice_count])
            terms = np.divide(sums, denominator, out=np.zeros_like(sums), where=denominator > 0)
            loss = terms[0] + self.value_weight * terms[1] - self.entropy_weight * terms[2]
            return np.r_[loss, terms], np.asarray([exact_sum / max(choice_count, 1), worst])

        def audit_states():
            # Canonical reads use no RNG and do not refresh frozen advantages or
            # behavior probabilities. Only actor variables enter this graph.
            state_kl = np.empty(count, np.float64)
            choice = np.empty(count, bool)
            old_chosen = np.empty(count, np.float64)
            new_chosen = np.empty(count, np.float64)
            for chunk in chunks(range(count)):
                inputs, actions, _, _, _, old = pack(chunk)
                vectors = self.actor_state_drift(inputs, old, actions)
                for destination, vector in zip((state_kl, choice, old_chosen, new_chosen), vectors):
                    destination[chunk] = np.asarray(vector)
            ids = np.flatnonzero(choice)
            finite_values = state_kl[ids][np.isfinite(state_kl[ids])]
            quantiles = dict(zip(('p50','p95','p99','p999'),
                map(float, np.quantile(finite_values, [.5,.95,.99,.999])))) if len(finite_values) else dict.fromkeys(('p50','p95','p99','p999'))
            ranked = ids[np.argsort(-np.where(np.isfinite(state_kl[ids]), state_kl[ids], np.inf), kind='stable')[:5]]
            worst_states = []
            for identity in ranked:
                identity = int(identity)
                trajectory_id = int(np.searchsorted(offsets, identity, side='right')-1)
                offset = int(identity-offsets[trajectory_id])
                trajectory = trajectories[trajectory_id]
                # The critic's prefetch can still seek the same spill file.
                with packing_lock:
                    sample = indexed.get(trajectory_id, offset)
                action = int(sample.action)
                action_kind = getattr(sample, 'action_kind', None)
                decision_kind = getattr(sample, 'decision_kind', None)
                from .rustmage_features import FEATURE_SPEC, ACTIONS, DECISIONS, STEPS
                if self.model.feature_spec == FEATURE_SPEC:
                    prefix = sample.arrays['candidates'][action, :len(ACTIONS)]
                    decisions = sample.arrays['context'][len(STEPS):len(STEPS)+len(DECISIONS)]
                    if prefix.max() > 0: action_kind = ACTIONS[int(prefix.argmax())]
                    if decisions.max() > 0: decision_kind = DECISIONS[int(decisions.argmax())]
                worst_states.append(dict(sample_index=identity, trajectory_index=trajectory_id,
                    trajectory_path=str(trajectory['path']), sample_offset=offset, action_index=action,
                    action_kind=action_kind, decision_kind=decision_kind,
                    kl=_json_number(state_kl[identity]), advantage=_json_number(advantages[identity]),
                    old_selected_probability=_json_number(old_chosen[identity]),
                    new_selected_probability=_json_number(new_chosen[identity]),
                    legal_actions=int(sample.arrays['candidate_mask'].sum())))
            return dict(choice_states=len(ids), nonfinite_states=int((~np.isfinite(state_kl[ids])).sum()),
                kl_quantiles=quantiles, states_above_max_state_kl=int((state_kl[ids] > self.max_state_kl).sum()),
                worst_states=worst_states)

        guard_proposals = []
        steps = rejected = 0; norm = 0.; stopped = False; largest = np.zeros(2)
        metrics, exact = measure()
        if policy_cache: policy_cache.finish()
        if device_cache: device_cache.finish()
        value_loss_before = float(metrics[2])
        # Draw the same critic permutations, in the same order, before the
        # actor consumes RNG, including unused epochs when value fitting is
        # capped. Changing the value budget must not reshuffle policy batches.
        # Only the calling thread ever touches self.rng.
        critic_orders = iter([self.rng.permutation(count) for _ in range(self.epochs)]) if self.separate_value else None
        def critic_batches():
            order = next(critic_orders) if critic_orders is not None else self.rng.permutation(count)
            with batches(order) as prepared:
                for inputs, _, rewards, *_ in prepared:
                    yield inputs, rewards
        def fit_values():
            if self.critic_unroll == 1:
                return self.fit_critic(critic_batches)
            from .sequence_critic import fit_sequence_critic
            def get_record(identity, position):
                with packing_lock:
                    return indexed.get(identity, position)
            return fit_sequence_critic(self, trajectories, critic_orders, get_record)
        # The executor joins even when either side fails, before the spill file
        # is closed or a caller can save a checkpoint. Optimizers and variables
        # are disjoint; advantages and behavior memories stay frozen.
        pool = (ThreadPoolExecutor(max_workers=1, thread_name_prefix='ppo-critic')
                if self.concurrent_updates and not critic_only else nullcontext())
        with pool as executor:
            future = executor.submit(fit_values) if executor else None
            critic_stats = {} if future else fit_values()
            for epoch in range(0 if critic_only else self.epochs):
                saved = self.snapshot(); rate = self.learning_rate
                order = self.rng.permutation(count); accepted = False
                with actor_epoch_inputs(epoch) as epoch_cache:
                    for attempt in range(self.backtracks+1):
                        self.optimizer.learning_rate.assign(rate * .5**attempt)
                        epoch_steps = 0
                        with batches(order, actor_cache=epoch_cache) as prepared:
                            for inputs, actions, rewards, chosen, adv, _ in prepared:
                                if future and future.done(): future.result()
                                scope = epoch_cache._tf.device(epoch_cache.device) if epoch_cache else nullcontext()
                                with scope:
                                    norm = float(self.step(inputs, actions, rewards, chosen, adv))
                                epoch_steps += 1
                        if epoch_cache is not None:
                            # Do not keep the last consumed device batch alive
                            # across the epoch boundary after closing its cache.
                            del inputs, actions, rewards, chosen, adv, _
                        if epoch_cache is not None and not epoch_cache.finished:
                            epoch_cache.finish()
                        proposed, drift = measure(actor_only=self.separate_value)
                        accepted_proposal, finite, reasons = _guard_status(proposed, drift, self.target_kl, self.max_state_kl)
                        if self.guard_audit:
                            event = dict(epoch=epoch+1, attempt=attempt,
                                configured_learning_rate=float(self.learning_rate),
                                effective_learning_rate=float(self.optimizer.learning_rate.numpy()),
                                finite=finite, accepted=accepted_proposal, rejection_reasons=reasons,
                                mean_kl=_json_number(drift[0]), max_state_kl=_json_number(drift[1]),
                                target_kl=float(self.target_kl), max_state_kl_limit=float(self.max_state_kl),
                                **audit_states())
                            guard_proposals.append(event)
                            if self.guard_observer is not None: self.guard_observer(event)
                        if accepted_proposal:
                            metrics, exact = proposed, drift; largest = np.maximum(largest, drift)
                            steps += epoch_steps; accepted = True; break
                        self.restore(saved); rejected += 1
                if not accepted: stopped = True; break
            if future: critic_stats = future.result()
        if self.separate_value:
            metrics, exact = measure()
            if not np.isfinite(metrics).all() or not np.isfinite(exact).all():
                raise FloatingPointError('Nonfinite final actor/critic metrics')
        result = dict(zip(('loss','policy_loss','value_loss','entropy','approx_kl','clip_fraction'), map(float,metrics)),
            samples=count,optimizer_steps=int(self.optimizer.iterations),accepted_steps=steps,
            rejected_epochs=rejected,early_stop=stopped,gradient_norm=norm,
            exact_kl=float(exact[0]),max_state_kl=float(exact[1]),
            largest_accepted_kl=float(largest[0]),largest_accepted_state_kl=float(largest[1]),
            reward_std=float(np.std(returns)),advantage_std=float(np.std(advantages)),
            advantage_baseline=self.advantage_baseline,
            cache=dict(budget_bytes=self.cache_bytes, retained_bytes=records_cache.bytes+packed_cache.bytes,
                trajectories=len(records_cache.entries), trajectory_hits=records_cache.hits,
                batches=len(packed_cache.entries), batch_hits=packed_cache.hits,
                spill_bytes=indexed.disk_bytes),
            storage=dict(numeric_reader_opens=indexed.numeric_opens,
                indexed_trajectories=sum(str(t['path']).endswith('.rlt') for t in trajectories)),
            sample_order='global_shuffle_v1',
            update_schedule='concurrent' if self.concurrent_updates else 'sequential',
            pipeline=dict(prefetch_batches=self.prefetch_batches,measurement_workers=self.measurement_workers,
                numerical_prepacking=self.numerical_prepacking, cache_policy_batches=self.cache_policy_batches,
                numerical_disk_bytes=numeric.disk_bytes if numeric else 0,
                numerical_gathers=numeric.gathers if numeric else 0,
                policy_cache_disk_bytes=policy_cache.disk_bytes if policy_cache else 0,
                policy_cache_hits=policy_cache.hits if policy_cache else 0,
                policy_cache_device=device_cache.device if device_cache else None,
                policy_cache_device_budget_bytes=self.policy_cache_device_bytes,
                policy_cache_device_bytes=device_cache.bytes if device_cache else 0,
                policy_cache_device_uploads=device_cache.uploads if device_cache else 0,
                policy_cache_device_tensors=device_cache.uploaded_tensors if device_cache else 0,
                policy_cache_device_hits=device_cache.hits if device_cache else 0,
                policy_cache_device_fallback_batches=device_cache.fallback_batches if device_cache else 0,
                policy_cache_device_fallback_bytes=device_cache.fallback_bytes if device_cache else 0,
                actor_epoch_cache_device=self.actor_epoch_cache_device,
                actor_epoch_cache_budget_bytes=self.actor_epoch_cache_bytes,
                actor_epoch_cache_epochs=actor_epoch_cache_stats,
                actor_epoch_cache_uploads=sum(r['uploads'] for r in actor_epoch_cache_stats),
                actor_epoch_cache_hits=sum(r['hits'] for r in actor_epoch_cache_stats),
                actor_epoch_cache_fallback_batches=sum(r['fallback_batches'] for r in actor_epoch_cache_stats),
                actor_epoch_cache_fallback_bytes=sum(r['fallback_bytes'] for r in actor_epoch_cache_stats),
                actor_epoch_cache_max_logical_bytes=max((r['logical_bytes'] for r in actor_epoch_cache_stats), default=0),
                actor_epoch_cache_max_epoch_input_bytes=max((r['unique_input_bytes'] for r in actor_epoch_cache_stats), default=0)),
            behavior_critic=value_prediction_diagnostics(returns, values, advancing),
            recurrent_updates='detached behavior states',terminal_rewards_only=True,
            value_loss_before=value_loss_before, **critic_stats)
        if self.guard_audit: result['guard_proposals'] = guard_proposals
        return result
