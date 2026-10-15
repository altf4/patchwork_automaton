"""Terminal-reward actor-critic, with detached recurrent rollout states.

One optimizer update per completed self-play game. Gradients are accumulated
across minibatches before applying, keeping the entire rollout on-policy.
"""
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import keras

from .features import collate
from .policy import NeuralPolicy


def terminal_reward(outcome):
    if outcome is None:
        raise ValueError('A verified terminal outcome is required')
    if outcome.draw is True:
        return 0.0
    if outcome.won is True:
        return 1.0
    if outcome.won is False:
        return -1.0
    raise ValueError('Unknown terminal outcome')


@dataclass
class Sample:
    arrays: dict
    memory: np.ndarray
    advance: bool
    action: int
    value: float
    log_probs: np.ndarray | None = None


class RolloutPolicy(NeuralPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory = []

    def reset(self):
        super().reset()
        self.trajectory = []

    def _select(self, arrays, state, *, advance, stage, records):
        index, updated, value = super()._select(arrays, state, advance=advance, stage=stage, records=records)
        self.trajectory.append(Sample({k: v.copy() for k, v in arrays.items()},
                                     state.copy(), advance, index, value))
        return index, updated, value


class TerminalActorCritic:
    def __init__(self, model, *, learning_rate=0.0003, batch_size=16,
                 value_weight=0.5, entropy_weight=0.01, clip_norm=1.0):
        self.model = model
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.batch_size = batch_size
        self.value_weight, self.entropy_weight = value_weight, entropy_weight
        self.clip_norm = clip_norm
        if batch_size < 1:
            raise ValueError('Batch size must be positive')

    def update(self, trajectories):
        """Each entry is (samples, terminal reward); truncated games must not enter."""
        samples, rewards = [], []
        for trajectory, reward in trajectories:
            if reward not in (-1.0, 0.0, 1.0):
                raise ValueError('Reward must be win/loss/draw')
            samples.extend(trajectory)
            rewards.extend([reward] * len(trajectory))
        if not samples:
            raise ValueError('Cannot train on an empty rollout')
        variables = self.model.trainable_variables
        accumulated = [tf.zeros_like(v) for v in variables]
        totals = dict(loss=0., policy_loss=0., value_loss=0., entropy=0.)
        for start in range(0, len(samples), self.batch_size):
            chunk = samples[start:start + self.batch_size]
            inputs = collate([s.arrays for s in chunk])
            inputs['memory_in'] = np.stack([s.memory for s in chunk])
            inputs['advance_memory'] = np.asarray([[float(s.advance)] for s in chunk], np.float32)
            target = tf.constant(rewards[start:start + len(chunk)], tf.float32)
            # Monte Carlo advantage from the behavior model; no step rewards or discount.
            advantage = target - tf.constant([s.value for s in chunk], tf.float32)
            actions = tf.constant([s.action for s in chunk], tf.int32)
            with tf.GradientTape() as tape:
                output = self.model(inputs, training=True)
                logp = tf.nn.log_softmax(output['logits'])
                chosen = tf.gather(logp, actions, axis=1, batch_dims=1)
                entropy = -tf.reduce_sum(tf.exp(logp) * logp, axis=1)
                actor = -tf.reduce_mean(chosen * tf.stop_gradient(advantage))
                # Internal integer digits share a state; fit value once per decision.
                value_mask = tf.constant([float(s.advance) for s in chunk])
                critic = tf.reduce_mean(tf.square(tf.squeeze(output['value'], -1) - target) * value_mask)
                exploration = tf.reduce_mean(entropy)
                loss = actor + self.value_weight * critic - self.entropy_weight * exploration
            gradients = tape.gradient(loss, variables)
            weight = len(chunk) / len(samples)
            accumulated = [a + weight * tf.convert_to_tensor(g) if g is not None else a
                           for a, g in zip(accumulated, gradients)]
            for key, value in zip(totals, (loss, actor, critic, exploration)):
                totals[key] += weight * float(value)
        for gradient in accumulated:
            tf.debugging.assert_all_finite(gradient, 'Nonfinite training gradient')
        clipped, norm = tf.clip_by_global_norm(accumulated, self.clip_norm)
        self.optimizer.apply_gradients(zip(clipped, variables))
        return dict(totals, samples=len(samples), gradient_norm=float(norm),
                    optimizer_steps=int(self.optimizer.iterations))
