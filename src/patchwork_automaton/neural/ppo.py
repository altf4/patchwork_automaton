"""PPO updates for frozen numeric rollouts; reward construction stays external.

Actor inputs include detached behavior recurrent states; optional ordered
critic TBPTT is independent. This does not implement simulator search. Never
refresh old probabilities or advantages between epochs of the same rollout.
"""
import numpy as np
import tensorflow as tf
import keras

from .batching import example_batch, input_signature


def _assert_finite_gradients(gradients, message):
    """Reduce on-device and transfer one validity flag, never gradient arrays.

    The returned token must guard the *whole* optimizer call: Adam's slot and
    iteration writes must wait even when they do not consume a gradient value.
    Empty gradients are finite, matching assert_all_finite's behavior.
    """
    flags = [tf.reduce_all(tf.math.is_finite(gradient)) for gradient in gradients]
    finite = tf.reduce_all(tf.stack(flags)) if flags else tf.constant(True)
    assertion = tf.raw_ops.Assert(condition=finite, data=[message], summarize=1)
    # Eager assertions execute immediately and return None; graph assertions
    # need an explicit edge so none of the optimizer writes can race failure.
    with tf.control_dependencies([] if assertion is None else [assertion]):
        return tf.identity(finite)


def clipped_surrogate(log_probability, old_log_probability, advantage, clip_ratio):
    ratio = tf.exp(log_probability - old_log_probability)
    return tf.minimum(ratio * advantage,
                      tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage)


def normalized_advantages(returns, values, choices, scale_floor):
    """Normalize once per rollout without magnifying tiny critic residuals."""
    advantages = np.asarray(returns, np.float32) - np.asarray(values, np.float32)
    if np.any(choices):
        selected = advantages[choices]
        if selected.std() > 1e-8:
            advantages = (advantages - selected.mean()) / max(float(selected.std()), scale_floor)
    return np.asarray(advantages, np.float32)


def value_prediction_diagnostics(returns, values, mask):
    """Audit frozen behavior predictions before learning from these outcomes.

    A low MSE from a nearly constant prediction can just reflect outcome
    imbalance. Explained variance and prediction spread expose that collapse.
    """
    target = np.asarray(returns, np.float64)[np.asarray(mask, bool)]
    prediction = np.asarray(values, np.float64)[np.asarray(mask, bool)]
    if not len(target): return {'samples': 0}
    residual = target - prediction
    variance = float(target.var())
    return dict(samples=len(target), mse=float(np.mean(residual**2)),
        zero_baseline_mse=float(np.mean(target**2)),
        constant_baseline_mse=variance, target_mean=float(target.mean()),
        prediction_mean=float(prediction.mean()), prediction_std=float(prediction.std()),
        explained_variance=float(1-residual.var()/variance) if variance > 0 else None)


class PPOTrainer:
    def __init__(self, model, *, learning_rate=0.0003, epochs=4, batch_size=48,
                 clip_ratio=0.2, target_kl=0.02, value_weight=0.5,
                 entropy_weight=0.01, clip_norm=1.0, advantage_scale_floor=0.1, trust_guard=False, max_state_kl=0.1, backtracks=6,
                 value_learning_rate=None, value_max_steps=None, advantage_baseline='critic',
                 critic_update_scope='all', critic_unroll=1):
        if (epochs < 1 or batch_size < 1 or not 0 < clip_ratio < 1
                or min(learning_rate, target_kl, clip_norm, advantage_scale_floor) <= 0):
            raise ValueError('Invalid PPO configuration')
        if advantage_baseline not in ('critic', 'zero'):
            raise ValueError('Actor advantage baseline must be critic or zero')
        if critic_update_scope not in ('all', 'head'):
            raise ValueError('Critic update scope must be all or head')
        if type(critic_unroll) is not int or critic_unroll < 1:
            raise ValueError('Critic unroll must be a positive integer')
        self.model, self.epochs, self.batch_size = model, epochs, batch_size
        self.clip_ratio, self.target_kl = clip_ratio, target_kl
        self.value_weight, self.entropy_weight = value_weight, entropy_weight
        self.clip_norm = clip_norm
        self.advantage_scale_floor = advantage_scale_floor
        self.advantage_baseline = advantage_baseline
        self.trust_guard, self.max_state_kl, self.backtracks = trust_guard, max_state_kl, backtracks
        self.learning_rate = learning_rate
        self.separate_value = bool(getattr(model, 'separate_value', False))
        if not self.separate_value and (critic_update_scope != 'all' or critic_unroll != 1):
            raise ValueError('Critic update controls require separate actor and critic networks')
        self.critic_update_scope, self.critic_unroll = critic_update_scope, critic_unroll
        if value_learning_rate is not None and (not np.isfinite(value_learning_rate) or value_learning_rate <= 0):
            raise ValueError('Value learning rate must be finite and positive')
        if value_max_steps is not None and (type(value_max_steps) is not int or value_max_steps < 1):
            raise ValueError('Value step limit must be a positive integer')
        if not self.separate_value and (value_learning_rate is not None or value_max_steps is not None):
            raise ValueError('Independent value settings require separate policy and value networks')
        self.value_learning_rate = learning_rate if value_learning_rate is None else value_learning_rate
        self.value_max_steps = value_max_steps
        if max_state_kl <= 0 or backtracks < 0:
            raise ValueError('Invalid trust-region limits')
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.actor_variables = tuple(model.actor.trainable_variables if self.separate_value else model.trainable_variables)
        self.optimizer.build(self.actor_variables)
        if self.separate_value:
            self.critic_optimizer = keras.optimizers.Adam(self.value_learning_rate)
            self.critic_optimizer.build(model.critic.trainable_variables)
        signature = [input_signature(example_batch(model.memory_size, getattr(model, 'feature_spec', None))),
                     tf.TensorSpec([None], tf.int32)] + [tf.TensorSpec([None], tf.float32)] * 3
        self.step = tf.function(self._step, input_signature=signature, autograph=False)
        if self.separate_value:
            self.critic_step = tf.function(self._critic_step,
                input_signature=[signature[0], tf.TensorSpec([None], tf.float32)], autograph=False)
            if critic_unroll > 1:
                from .sequence_critic import sequence_step
                sequence_signature = {k:v for k,v in signature[0].items() if k != 'memory_in'}
                self.critic_sequence_step = tf.function(lambda *args: sequence_step(self, *args),
                    input_signature=[tuple(sequence_signature for _ in range(critic_unroll)),
                        tf.TensorSpec([None, model.critic.memory_size], tf.float32),
                        tf.TensorSpec([None], tf.float32),
                        tf.TensorSpec([critic_unroll, None], tf.float32)], autograph=False)
        self.measure = tf.function(self._measure, input_signature=signature, autograph=False)
        self.drift = tf.function(self._drift, input_signature=[signature[0], tf.TensorSpec([None, None], tf.float32)], autograph=False)
        self.measure_with_drift = tf.function(self._measure_with_drift,
            input_signature=signature + [tf.TensorSpec([None, None], tf.float32)], autograph=False)
        if self.separate_value:
            self.measure_actor_with_drift = tf.function(self._measure_actor_with_drift,
                input_signature=signature + [tf.TensorSpec([None, None], tf.float32)], autograph=False)
        self.actor_state_drift = tf.function(self._actor_state_drift,
            input_signature=[signature[0], tf.TensorSpec([None, None], tf.float32), signature[1]],
            autograph=False)
        self.guard_variables = tuple(model.actor.variables if self.separate_value else model.variables) + tuple(self.optimizer.variables)
        self.snapshot = tf.function(lambda: tuple(tf.identity(v) for v in self.guard_variables), autograph=False)
        self.restore = tf.function(self._restore,
            input_signature=[tuple(tf.TensorSpec(v.shape, tf.as_dtype(v.dtype)) for v in self.guard_variables)], autograph=False)

    def _terms(self, inputs, actions, returns, old_logp, advantages):
        output = self.model(inputs, training=True)
        logp = tf.nn.log_softmax(output['logits'])
        return self._terms_from_output(output, logp, inputs, actions, returns, old_logp, advantages)

    def _terms_from_output(self, output, logp, inputs, actions, returns, old_logp, advantages):
        chosen = tf.gather(logp, actions, axis=1, batch_dims=1)
        mask = tf.cast(tf.reduce_sum(tf.cast(inputs['candidate_mask'], tf.int32), axis=1) > 1, tf.float32)
        def mean_choice(x):
            return tf.math.divide_no_nan(tf.reduce_sum(x * mask), tf.reduce_sum(mask))
        actor = -mean_choice(clipped_surrogate(chosen, old_logp, advantages, self.clip_ratio))
        value_mask = inputs['advance_memory'][:, 0]
        critic = tf.math.divide_no_nan(tf.reduce_sum(tf.square(output['value'][:, 0] - returns) * value_mask), tf.reduce_sum(value_mask))
        entropy = mean_choice(-tf.reduce_sum(tf.exp(logp) * logp, axis=1))
        delta = chosen - old_logp
        # Nonnegative sampled KL estimator; all behavior statistics stay frozen.
        kl = mean_choice(tf.exp(delta) - 1 - delta)
        fraction = mean_choice(tf.cast(tf.abs(tf.exp(delta) - 1) > self.clip_ratio, tf.float32))
        loss = actor + self.value_weight * critic - self.entropy_weight * entropy
        return tf.stack([loss, actor, critic, entropy, kl, fraction])

    def _restore(self, values):
        for variable, value in zip(self.guard_variables, values):
            variable.assign(value)
        return tf.constant(0)

    def _drift(self, inputs, old_log_probs):
        new = tf.nn.log_softmax(self.model(inputs, training=False)['logits'])
        return self._drift_from_logp(inputs, old_log_probs, new)

    def _drift_from_logp(self, inputs, old_log_probs, new):
        kl = tf.maximum(tf.reduce_sum(tf.exp(old_log_probs) * (old_log_probs - new), axis=1), 0.)
        choices = tf.cast(tf.reduce_sum(tf.cast(inputs['candidate_mask'], tf.int32), axis=1) > 1, tf.float32)
        return tf.stack([tf.math.divide_no_nan(tf.reduce_sum(kl * choices), tf.reduce_sum(choices)), tf.reduce_max(kl * choices)])

    def _actor_state_drift(self, inputs, old_log_probs, actions):
        """Read-only per-state KL and chosen probabilities; never read a separate critic."""
        output = (self.model.actor_output(inputs, training=False) if self.separate_value
                  else self.model(inputs, training=False))
        new = tf.nn.log_softmax(output['logits'])
        kl = tf.maximum(tf.reduce_sum(tf.exp(old_log_probs) * (old_log_probs - new), axis=1), 0.)
        choices = tf.reduce_sum(tf.cast(inputs['candidate_mask'], tf.int32), axis=1) > 1
        return (kl, choices, tf.exp(tf.gather(old_log_probs, actions, axis=1, batch_dims=1)),
                tf.exp(tf.gather(new, actions, axis=1, batch_dims=1)))

    def _measure_with_drift(self, inputs, actions, returns, old_logp, advantages, old_log_probs):
        if not getattr(self.model, 'deterministic_forward', False):
            return (self._terms(inputs, actions, returns, old_logp, advantages),
                    self._drift(inputs, old_log_probs))
        output = self.model(inputs, training=False)
        logp = tf.nn.log_softmax(output['logits'])
        return (self._terms_from_output(output, logp, inputs, actions, returns, old_logp, advantages),
                self._drift_from_logp(inputs, old_log_probs, logp))

    def _measure(self, *args):
        return self._terms(*args)

    def _measure_actor_with_drift(self, inputs, actions, returns, old_logp, advantages, old_log_probs):
        """Trust-region checks that never read the independently updated critic."""
        output = self.model.actor_output(inputs, training=False)
        logp = tf.nn.log_softmax(output['logits'])
        terms = self._terms_from_output(output, logp, inputs, actions, returns, old_logp, advantages)
        # Value loss is measured after joining the critic worker. It plays no
        # part in accepting or rolling back a separate actor's optimizer step.
        metrics = tf.stack([terms[1] - self.entropy_weight * terms[3], terms[1],
                            tf.constant(0., tf.float32), terms[3], terms[4], terms[5]])
        return metrics, self._drift_from_logp(inputs, old_log_probs, logp)

    def _step(self, *args):
        with tf.GradientTape() as tape:
            if self.separate_value:
                output = self.model.actor_output(args[0], training=True)
                metrics = self._terms_from_output(output, tf.nn.log_softmax(output['logits']), *args)
                loss = metrics[1] - self.entropy_weight * metrics[3]
            else:
                loss = self._terms(*args)[0]
        gradients = tape.gradient(loss, self.actor_variables)
        gradients = [tf.zeros_like(v) if g is None else tf.convert_to_tensor(g)
                     for g, v in zip(gradients, self.actor_variables)]
        with tf.control_dependencies([_assert_finite_gradients(gradients, 'Nonfinite PPO gradient')]):
            gradients, norm = tf.clip_by_global_norm(gradients, self.clip_norm)
            self.optimizer.apply_gradients(zip(gradients, self.actor_variables))
        return norm

    def _critic_step(self, inputs, returns):
        with tf.GradientTape() as tape:
            value = self.model.value_output(inputs, training=True)['value'][:, 0]
            mask = inputs['advance_memory'][:, 0]
            loss = tf.math.divide_no_nan(tf.reduce_sum(tf.square(value - returns) * mask), tf.reduce_sum(mask))
        variables = (self.model.critic.value_head.trainable_variables if self.critic_update_scope == 'head'
                     else self.model.critic.trainable_variables)
        gradients = tape.gradient(loss, variables)
        gradients = [tf.zeros_like(v) if g is None else tf.convert_to_tensor(g) for g, v in zip(gradients, variables)]
        with tf.control_dependencies([_assert_finite_gradients(gradients, 'Nonfinite critic gradient')]):
            gradients, norm = tf.clip_by_global_norm(gradients, self.clip_norm)
            self.critic_optimizer.apply_gradients(zip(gradients, variables))
        return norm

    def fit_critic(self, batches):
        """Fit terminal returns, optionally capping value steps across all epochs.

        The cap resets for each rollout update. It never repeats a short rollout
        beyond the configured epochs merely to exhaust the budget.
        """
        if not self.separate_value: return {}
        steps = 0; norm = 0.
        for _ in range(self.epochs):
            stream = batches()
            try:
                for inputs, returns in stream:
                    norm = float(self.critic_step(inputs, returns)); steps += 1
                    if self.value_max_steps is not None and steps >= self.value_max_steps:
                        break
            finally:
                # A prefetched stream owns background readers. Join them even
                # if the optimizer fails, before shared rollout files close.
                close = getattr(stream,'close',None)
                if close is not None: close()
            if self.value_max_steps is not None and steps >= self.value_max_steps:
                break
        return dict(critic_steps=steps, critic_optimizer_steps=int(self.critic_optimizer.iterations),
                    critic_gradient_norm=norm, value_max_steps=self.value_max_steps,
                    value_step_limit_reached=self.value_max_steps is not None and steps >= self.value_max_steps,
                    value_learning_rate=float(self.critic_optimizer.learning_rate.numpy()),
                    critic_update_scope=self.critic_update_scope, critic_unroll=self.critic_unroll)

    def update(self, inputs, actions, returns, old_logp, old_values, *, rng, old_log_probs=None):
        """Optimize one frozen rollout; returns must come from its task outcome.

        For real games use undiscounted terminal win/loss/draw returns. The
        caller must collect all samples before updating the behavior model.
        """
        if self.critic_unroll > 1:
            raise ValueError('Sequence critic training requires ordered trajectories through update_files')
        actions = np.asarray(actions, np.int32)
        returns, old_logp, old_values = [np.asarray(x, np.float32).copy() for x in (returns, old_logp, old_values)]
        count = len(actions)
        if not count or any(x.shape != (count,) for x in (returns, old_logp, old_values)):
            raise ValueError('Expected nonempty aligned rollout vectors')
        if any(not np.isfinite(x).all() for x in (returns, old_logp, old_values)):
            raise ValueError('Nonfinite rollout statistics')
        if any(len(x) != count for x in inputs.values()):
            raise ValueError('Misaligned rollout inputs')
        mask = np.asarray(inputs['candidate_mask'])
        if np.any(actions < 0) or np.any(actions >= mask.shape[1]) or not mask[np.arange(count), actions].all():
            raise ValueError('Rollout action is not legal')
        actor_values = old_values if self.advantage_baseline == 'critic' else np.zeros_like(old_values)
        advantages = normalized_advantages(returns, actor_values, mask.sum(axis=1) > 1,
                                           self.advantage_scale_floor)
        if self.trust_guard:
            if old_log_probs is None:
                raise ValueError('Guarded PPO requires the full behavior distribution')
            old_log_probs = np.asarray(old_log_probs, np.float32).copy()
            if old_log_probs.shape != mask.shape or not np.isfinite(old_log_probs).all():
                raise ValueError('Invalid behavior distribution')
            if not np.allclose(old_log_probs[np.arange(count), actions], old_logp, atol=1e-5):
                raise ValueError('Chosen and full behavior probabilities disagree')
            if not np.allclose(np.exp(old_log_probs).sum(axis=1), 1., atol=1e-5):
                raise ValueError('Behavior probabilities are not normalized')
            # Each new rollout may try the configured step size again. Backtracking
            # adapts proposals to this rollout, not a permanently shrinking schedule.
            self.optimizer.learning_rate.assign(self.learning_rate)
        steps, norm, stopped, rejected = 0, 0., False, 0
        exact = np.zeros(2)
        largest_exact = np.zeros(2)
        metrics = np.asarray(self.measure(inputs, actions, returns, old_logp, advantages))
        value_loss_before = float(metrics[2])
        # Reserve all value permutations even when fitting stops early, so the
        # value budget cannot change the policy's minibatch RNG sequence.
        critic_orders = iter([rng.permutation(count) for _ in range(self.epochs)]) if self.separate_value else None
        def critic_batches():
            order = next(critic_orders)
            for start in range(0, count, self.batch_size):
                ix = order[start:start + self.batch_size]
                yield {k: np.asarray(v)[ix] for k, v in inputs.items()}, returns[ix]
        critic_stats = self.fit_critic(critic_batches)
        if self.separate_value:
            metrics = np.asarray(self.measure(inputs, actions, returns, old_logp, advantages))
        for epoch in range(self.epochs):
            order = rng.permutation(count)
            saved = self.snapshot() if self.trust_guard else ()
            rate = float(self.optimizer.learning_rate.numpy())
            accepted = False
            for attempt in range(self.backtracks + 1 if self.trust_guard else 1):
                epoch_steps = 0
                for start in range(0, count, self.batch_size):
                    ix = order[start:start + self.batch_size]
                    batch = {k: np.asarray(v)[ix] for k, v in inputs.items()}
                    norm = float(self.step(batch, actions[ix], returns[ix], old_logp[ix], advantages[ix]))
                    epoch_steps += 1
                if self.trust_guard:
                    proposed, drift = map(np.asarray, self.measure_with_drift(
                        inputs, actions, returns, old_logp, advantages, old_log_probs))
                else:
                    proposed = np.asarray(self.measure(inputs, actions, returns, old_logp, advantages))
                    drift = np.zeros(2)
                if (np.isfinite(proposed).all() and np.isfinite(drift).all()
                        and (not self.trust_guard or (drift[0] <= self.target_kl and drift[1] <= self.max_state_kl))):
                    metrics, exact = proposed, drift
                    largest_exact = np.maximum(largest_exact, drift)
                    steps += epoch_steps
                    accepted = True
                    break
                if not self.trust_guard:
                    raise FloatingPointError('Nonfinite PPO metrics')
                # Restore Adam's moments and iteration counter too. Keeping those
                # from a rejected proposal would corrupt the next accepted update.
                self.restore(saved)
                rejected += 1
                if attempt < self.backtracks:
                    self.optimizer.learning_rate.assign(rate * .5 ** (attempt + 1))
            if not accepted:
                stopped = True
                break
            if not self.trust_guard and metrics[4] > self.target_kl:
                stopped = True
                break
        return dict(zip(('loss', 'policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction'), map(float, metrics)),
                    gradient_norm=norm, epochs=epoch + 1, steps=steps, early_stop=stopped,
                    rejected_epochs=rejected, exact_kl=float(exact[0]), max_state_kl=float(exact[1]),
                    reward_std=float(returns.std()), raw_advantage_std=float((returns-actor_values).std()),
                    normalized_advantage_std=float(advantages.std()),
                    advantage_baseline=self.advantage_baseline,
                    behavior_critic=value_prediction_diagnostics(returns, old_values, np.asarray(inputs['advance_memory'])[:, 0]),
                    largest_accepted_kl=float(largest_exact[0]), largest_accepted_state_kl=float(largest_exact[1]),
                    learning_rate=float(self.optimizer.learning_rate.numpy()),
                    value_loss_before=value_loss_before, **critic_stats)
