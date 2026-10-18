"""Bounded-memory PPO over frozen full-game trajectories with terminal returns.

Recurrent states are carried during collection and detached during updates.
This is one-step truncated recurrence, not sequence backpropagation.
"""
import itertools
import numpy as np
from .ppo import PPOTrainer, normalized_advantages
from .features import collate
from .batching import example_batch
from .experiment import read_trajectory
from .rollout_cache import RolloutCache, retained_bytes


class StreamingPPOTrainer(PPOTrainer):
    def __init__(self, model, *, seed=0, cache_bytes=256 * 1024**2, **kwargs):
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError('Rollout cache budget must be nonnegative integer bytes')
        model(example_batch(model.memory_size, model.feature_spec), training=False)
        super().__init__(model, **kwargs)
        self.rng = np.random.default_rng(seed)
        self.cache_bytes = cache_bytes

    def compile(self):
        batch = example_batch(self.model.memory_size, self.model.feature_spec)
        zero = np.zeros(1, np.float32)
        self.measure(batch, np.zeros(1, np.int32), zero, zero, zero)
        self.drift(batch, np.zeros((1, 1), np.float32))
        self.measure_with_drift(batch, np.zeros(1, np.int32), zero, zero, zero,
                                np.zeros((1, 1), np.float32))

    def update_files(self, trajectories):
        trajectories = list(trajectories)
        # Caches live for one frozen rollout only. Larger rollouts retain the
        # streaming fallback; they never force all trajectories into memory.
        records_cache = RolloutCache(self.cache_bytes * 3 // 4)
        packed_cache = RolloutCache(self.cache_bytes - records_cache.limit)
        offsets = []; values = []; returns = []; choices = []
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

        def batches(order):
            def samples():
                for index in order:
                    trajectory = trajectories[index]
                    records = records_cache.get(int(index))
                    if records is None: records = read_trajectory(trajectory['path'])
                    for position, sample in enumerate(records):
                        yield sample, trajectory['reward'], advantages[offsets[index]+position], (int(index), position)
            stream = samples()
            while chunk := list(itertools.islice(stream, self.batch_size)):
                permutation = self.rng.permutation(len(chunk))
                key = tuple(record[3] for record in chunk)
                packed = packed_cache.get(key)
                if packed is not None:
                    yield ({k: v[permutation] for k, v in packed[0].items()},
                           *(v[permutation] for v in packed[1:]))
                    continue
                # Cache in canonical order; apply the exact same random row
                # permutation on every visit, whether cached or streamed.
                may_cache = packed_cache.bytes < packed_cache.limit
                selected = chunk if may_cache else [chunk[i] for i in permutation]
                records, rewards, adv, _ = zip(*selected)
                inputs = collate([s.arrays for s in records])
                inputs['memory_in'] = np.stack([s.memory for s in records])
                inputs['advance_memory'] = np.asarray([[float(s.advance)] for s in records], np.float32)
                old = np.full(inputs['candidate_mask'].shape, -1.e9, np.float32)
                for row, sample in enumerate(records): old[row, :len(sample.log_probs)] = sample.log_probs
                actions = np.asarray([s.action for s in records], np.int32)
                chosen = old[np.arange(len(records)), actions]
                packed = (inputs, actions, np.asarray(rewards, np.float32), chosen, np.asarray(adv, np.float32), old)
                if may_cache:
                    if not packed_cache.put(key, packed):
                        # Stop admission at the first full prefix, avoiding
                        # repeated copies of batches that cannot fit the cache.
                        packed_cache.limit = packed_cache.bytes
                    yield ({k: v[permutation] for k, v in inputs.items()},
                           *(v[permutation] for v in packed[1:]))
                else:
                    yield packed

        def measure():
            sums = np.zeros(5); choice_count = value_count = 0; exact_sum = worst = 0.
            for inputs, actions, rewards, chosen, adv, old in batches(range(len(trajectories))):
                metrics, drift = map(np.asarray, self.measure_with_drift(inputs, actions, rewards, chosen, adv, old))
                n = int((inputs['candidate_mask'].sum(axis=1) > 1).sum())
                v = float(inputs['advance_memory'].sum())
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
        def critic_batches():
            for inputs, _, rewards, *_ in batches(self.rng.permutation(len(trajectories))):
                yield inputs, rewards
        critic_stats = self.fit_critic(critic_batches)
        if self.separate_value: metrics, exact = measure()
        for epoch in range(self.epochs):
            saved = self.snapshot(); rate = self.learning_rate
            order = self.rng.permutation(len(trajectories)); accepted = False
            for attempt in range(self.backtracks+1):
                self.optimizer.learning_rate.assign(rate * .5**attempt)
                epoch_steps = 0
                for inputs, actions, rewards, chosen, adv, _ in batches(order):
                    norm = float(self.step(inputs, actions, rewards, chosen, adv)); epoch_steps += 1
                proposed, drift = measure()
                if (np.isfinite(proposed).all() and np.isfinite(drift).all()
                        and drift[0] <= self.target_kl and drift[1] <= self.max_state_kl):
                    metrics, exact = proposed, drift; largest = np.maximum(largest, drift)
                    steps += epoch_steps; accepted = True; break
                self.restore(saved); rejected += 1
            if not accepted: stopped = True; break
        return dict(zip(('loss','policy_loss','value_loss','entropy','approx_kl','clip_fraction'), map(float,metrics)),
            samples=count,optimizer_steps=int(self.optimizer.iterations),accepted_steps=steps,
            rejected_epochs=rejected,early_stop=stopped,gradient_norm=norm,
            exact_kl=float(exact[0]),max_state_kl=float(exact[1]),
            largest_accepted_kl=float(largest[0]),largest_accepted_state_kl=float(largest[1]),
            reward_std=float(np.std(returns)),advantage_std=float(np.std(advantages)),
            cache=dict(budget_bytes=self.cache_bytes, retained_bytes=records_cache.bytes+packed_cache.bytes,
                trajectories=len(records_cache.entries), trajectory_hits=records_cache.hits,
                batches=len(packed_cache.entries), batch_hits=packed_cache.hits),
            recurrent_updates='detached behavior states',terminal_rewards_only=True,
            value_loss_before=value_loss_before, **critic_stats)
