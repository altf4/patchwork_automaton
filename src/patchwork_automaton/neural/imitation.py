"""Supervised legal-action cross entropy with truncated recurrent backpropagation.

Only expert action indices are targets. No returns, value loss, entropy bonus,
PPO ratios, or validation-based optimizer decisions enter this trainer.
"""
from collections import defaultdict

import keras
import numpy as np
import tensorflow as tf

from .batching import example_batch, input_signature
from .demonstrations import read_demonstration
from .features import collate
from .rustmage_features import ACTIONS


class ImitationTrainer:
    def __init__(self, model, *, learning_rate=.0003, batch_size=32, epochs=2,
                 unroll=8, seed=5721, clip_norm=1.):
        if min(batch_size, epochs, unroll) < 1 or not 0 < learning_rate < float('inf') or clip_norm <= 0:
            raise ValueError('Positive counts, learning rate, and clipping norm required')
        self.model = model
        self.policy = model.actor if getattr(model, 'separate_value', False) else model
        self.batch_size, self.epochs, self.unroll = batch_size, epochs, unroll
        self.clip_norm = clip_norm
        self.rng = np.random.default_rng(seed)
        self.optimizer = keras.optimizers.Adam(learning_rate)
        example = example_batch(self.policy.memory_size, model.feature_spec)
        self.policy(example, training=False)
        self.optimizer.build(self.policy.trainable_variables)
        sig = {k: tf.TensorSpec((None, *v.shape), v.dtype)
               for k, v in input_signature(example).items() if k not in ('memory_in', 'advance_memory')}
        signature = [tf.TensorSpec((None, self.policy.memory_size), tf.float32), sig,
                     tf.TensorSpec((None, None), tf.int32), tf.TensorSpec((None, None), tf.float32)]
        self.train_chunk = tf.function(self._train_chunk, input_signature=signature, autograph=False)
        self.evaluate_chunk = tf.function(self._evaluate_chunk, input_signature=signature, autograph=False)

    def compile(self):
        """Checkpoint loader compatibility; weights/optimizer are already built."""
        return self

    def _forward(self, memory, packed, labels, weights, training):
        losses = tf.TensorArray(tf.float32, size=tf.shape(labels)[0])
        predictions = tf.TensorArray(tf.int32, size=tf.shape(labels)[0])

        def body(t, state, losses, predictions):
            inputs = {k: v[t] for k, v in packed.items()}
            inputs.update(memory_in=state, advance_memory=weights[t, :, None])
            result = self.policy(inputs, training=training)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[t], logits=result['logits'])
            predicted = tf.argmax(result['logits'], axis=-1, output_type=tf.int32)
            return t + 1, result['memory_out'], losses.write(t, loss), predictions.write(t, predicted)

        _, state, losses, predictions = tf.while_loop(
            lambda t, *_: t < tf.shape(labels)[0], body, (0, memory, losses, predictions))
        return state, losses.stack(), predictions.stack()

    def _train_chunk(self, memory, packed, labels, weights):
        with tf.GradientTape() as tape:
            state, losses, predictions = self._forward(memory, packed, labels, weights, True)
            loss = tf.reduce_sum(losses * weights) / tf.reduce_sum(weights)
        gradients = tape.gradient(loss, self.policy.trainable_variables)
        pairs = [(g, v) for g, v in zip(gradients, self.policy.trainable_variables) if g is not None]
        for g, _ in pairs:
            tf.debugging.assert_all_finite(tf.convert_to_tensor(g), 'Nonfinite imitation gradient')
        clipped, _ = tf.clip_by_global_norm([g for g, _ in pairs], self.clip_norm)
        self.optimizer.apply_gradients(zip(clipped, [v for _, v in pairs]))
        return tf.stop_gradient(state), losses, predictions

    def _evaluate_chunk(self, memory, packed, labels, weights):
        return self._forward(memory, packed, labels, weights, False)

    def _chunks(self, streams):
        for start in range(0, max(map(len, streams)), self.unroll):
            length = min(self.unroll, max(map(len, streams)) - start)
            samples = [s[t] if t < len(s) else s[0]
                       for t in range(start, start + length) for s in streams]
            active = np.asarray([[float(t < len(s)) for s in streams]
                for t in range(start, start + length)], np.float32)
            packed = collate([s.arrays for s in samples])
            packed = {k: v.reshape(length, len(streams), *v.shape[1:]) for k, v in packed.items()}
            labels = np.asarray([s.action for s in samples], np.int32).reshape(active.shape)
            yield packed, labels, active, samples

    def _pass(self, trajectories, *, training):
        if not trajectories:
            raise ValueError('No expert trajectories')
        order = self.rng.permutation(len(trajectories)) if training else range(len(trajectories))
        ordered = [trajectories[i] for i in order]
        totals = defaultdict(lambda: dict(samples=0, correct=0, kind_correct=0, nll=0., teacher_probability=0.))
        for start in range(0, len(ordered), self.batch_size):
            streams = [read_demonstration(t) for t in ordered[start:start + self.batch_size]]
            # Each seat starts with zero memory; state never crosses a trajectory.
            memory = np.zeros((len(streams), self.policy.memory_size), np.float32)
            for packed, labels, active, samples in self._chunks(streams):
                fn = self.train_chunk if training else self.evaluate_chunk
                memory, losses, predicted = [np.asarray(v) for v in fn(memory, packed, labels, active)]
                if not all(np.isfinite(v).all() for v in (memory, losses)):
                    raise FloatingPointError('Nonfinite imitation output')
                for pos in zip(*np.nonzero(active)):
                    sample = samples[pos[0] * len(streams) + pos[1]]
                    row = totals[sample.action_kind]
                    nll = float(losses[pos])
                    chosen = int(predicted[pos])
                    kind = ACTIONS[int(np.argmax(packed['candidates'][(*pos, chosen, slice(0, len(ACTIONS)))]))]
                    row['samples'] += 1
                    row['correct'] += int(chosen == sample.action)
                    row['kind_correct'] += int(kind == sample.action_kind)
                    row['nll'] += nll
                    row['teacher_probability'] += float(np.exp(-nll))
        total = sum(r['samples'] for r in totals.values())
        for row in totals.values():
            row['accuracy'] = row['correct'] / row['samples']
            row['kind_accuracy'] = row['kind_correct'] / row['samples']
            row['cross_entropy'] = row['nll'] / row['samples']
        nonpass = [r for k, r in totals.items() if k != 'Pass']
        return dict(samples=total, cross_entropy=sum(r['nll'] for r in totals.values()) / total,
            accuracy=sum(r['correct'] for r in totals.values()) / total,
            kind_accuracy=sum(r['kind_correct'] for r in totals.values()) / total,
            teacher_probability=sum(r['teacher_probability'] for r in totals.values()) / total,
            macro_action_accuracy=float(np.mean([r['accuracy'] for r in totals.values()])),
            non_pass_accuracy=(sum(r['correct'] for r in nonpass) / sum(r['samples'] for r in nonpass)) if nonpass else None,
            per_action=dict(totals))

    def fit(self, games):
        if any(g.get('partition') != 'training' for g in games):
            raise ValueError('Only the training partition may enter the optimizer')
        trajectories = [t for g in games for t in g['trajectories']]
        before = int(self.optimizer.iterations)
        epochs = [self._pass(trajectories, training=True) for _ in range(self.epochs)]
        return dict(epochs=epochs, optimizer_steps=int(self.optimizer.iterations),
            steps_this_update=int(self.optimizer.iterations) - before)

    def evaluate(self, games):
        return self._pass([t for g in games for t in g['trajectories']], training=False)
