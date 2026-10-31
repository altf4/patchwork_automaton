"""Supervised terminal-outcome regression; the actor is never called or updated."""
from collections import defaultdict
import time
import keras
import numpy as np
import tensorflow as tf

from .batching import example_batch, input_signature
from .demonstrations import read_demonstration
from .devices import device_scope, assert_output_device


VALUE_INPUTS = ('entities', 'card_ids', 'entity_mask', 'semantics', 'semantic_mask',
                'relations', 'context', 'selection', 'history', 'history_cards', 'history_mask')


def value_chunks(streams, unroll):
    """Keep each seat chronological; padding does not advance its memory."""
    for start in range(0, max(map(len, streams)), unroll):
        length = min(unroll, max(map(len, streams)) - start)
        samples = [s[t] if t < len(s) else s[0]
                   for t in range(start, start + length) for s in streams]
        active = np.asarray([[t < len(s) for s in streams]
                             for t in range(start, start + length)], np.float32)
        packed = {}
        for key in VALUE_INPUTS:
            arrays = [s.arrays[key] for s in samples]
            shape = tuple(max(a.shape[i] for a in arrays) for i in range(arrays[0].ndim))
            batch = np.zeros((len(samples), *shape), arrays[0].dtype)
            for i, a in enumerate(arrays):
                batch[(i, *(slice(0, size) for size in a.shape))] = a
            packed[key] = batch.reshape(length, len(streams), *shape)
        yield packed, active


class ValueImitationTrainer:
    def __init__(self, model, *, batch_size=64, unroll=8, learning_rate=.0003,
                 seed=30261029, device='gpu'):
        if not getattr(model, 'separate_value', False) or min(batch_size, unroll) < 1:
            raise ValueError('Independent critic and positive sequence counts required')
        self.model, self.critic = model, model.critic
        self.batch_size, self.unroll, self.device = batch_size, unroll, device
        self.rng = np.random.default_rng(seed)
        with device_scope(device):
            self.optimizer = keras.optimizers.Adam(learning_rate)
            self.optimizer.build(self.critic.trainable_variables)
        example = example_batch(self.critic.memory_size, model.feature_spec)
        sig = {k: tf.TensorSpec((None, *v.shape), v.dtype)
               for k, v in input_signature(example).items() if k in VALUE_INPUTS}
        signature = [tf.TensorSpec((None, self.critic.memory_size), tf.float32), sig,
                     tf.TensorSpec((None,), tf.float32), tf.TensorSpec((None, None), tf.float32)]
        self.train_chunk = tf.function(self._train_chunk, input_signature=signature, autograph=False)
        self.evaluate_chunk = tf.function(self._forward, input_signature=signature, autograph=False)

    def compile(self):
        return self

    def _forward(self, memory, packed, targets, active, training=False):
        predictions = tf.TensorArray(tf.float32, size=tf.shape(active)[0])
        def step(t, state, values):
            inputs = {k: v[t] for k, v in packed.items()}
            inputs.update(memory_in=state, advance_memory=active[t, :, None])
            out = self.critic(inputs, training=training)
            return t + 1, out['memory_out'], values.write(t, tf.squeeze(out['value'], -1))
        _, memory, predictions = tf.while_loop(lambda t, *_: t < tf.shape(active)[0],
            step, (0, memory, predictions))
        return memory, predictions.stack()

    def _train_chunk(self, memory, packed, targets, active):
        with tf.GradientTape() as tape:
            memory, predictions = self._forward(memory, packed, targets, active, training=True)
            loss = tf.reduce_sum(tf.square(predictions - targets[None, :]) * active) / tf.reduce_sum(active)
        gradients = tape.gradient(loss, self.critic.trainable_variables)
        pairs = [(g, v) for g, v in zip(gradients, self.critic.trainable_variables) if g is not None]
        for gradient, _ in pairs:
            tf.debugging.assert_all_finite(tf.convert_to_tensor(gradient), 'Nonfinite value gradient')
        clipped, _ = tf.clip_by_global_norm([g for g, _ in pairs], 1.)
        self.optimizer.apply_gradients(zip(clipped, [v for _, v in pairs]))
        return tf.stop_gradient(memory), predictions

    def run(self, games, *, training=False, progress=None):
        if training and any(g['partition'] != 'training' for g in games):
            raise ValueError('Validation games cannot enter the optimizer')
        entries = [dict(t, pairing=' vs '.join(g['pairing'])) for g in games for t in g['trajectories']]
        if not entries:
            raise ValueError('No value examples')
        if training:
            entries = [entries[i] for i in self.rng.permutation(len(entries))]
        totals = defaultdict(lambda: np.zeros(6, np.float64))
        started = time.monotonic()
        for begin in range(0, len(entries), self.batch_size):
            batch = entries[begin:begin + self.batch_size]
            streams = [read_demonstration(t) for t in batch]
            targets = np.asarray([t['target'] for t in batch], np.float32)
            if not np.isin(targets, (-1., 0., 1.)).all():
                raise ValueError('Expected seat-relative terminal labels')
            memory = np.zeros((len(batch), self.critic.memory_size), np.float32)
            for packed, active in value_chunks(streams, self.unroll):
                with device_scope(self.device):
                    output = (self.train_chunk if training else self.evaluate_chunk)(memory, packed, targets, active)
                assert_output_device({'memory': output[0], 'predictions': output[1]}, self.device)
                memory, predictions = map(np.asarray, output)
                if not np.isfinite(predictions).all() or not np.isfinite(memory).all():
                    raise FloatingPointError('Nonfinite value output')
                for seat, t in enumerate(batch):
                    mask = active[:, seat].astype(bool)
                    v = predictions[mask, seat].astype(np.float64)
                    y = float(targets[seat]); error = v - y
                    row = np.array([len(v), np.square(error).sum(), np.abs(error).sum(),
                                    y * y * len(v), y * len(v), v.sum()])
                    for key in ('all', 'deck:' + t['deck'], 'pair:' + t['pairing']):
                        totals[key] += row
            if progress:
                progress(dict(seats=min(begin + len(batch), len(entries)), total_seats=len(entries),
                              seconds=time.monotonic() - started))
        def summarize(row):
            n, squared, absolute, zero, target, predicted = row
            return dict(samples=int(n), mse=float(squared/n), mae=float(absolute/n),
                        zero_mse=float(zero/n), target_mean=float(target/n), predicted_mean=float(predicted/n))
        result = summarize(totals.pop('all'))
        return dict(result, groups={k: summarize(v) for k, v in totals.items()},
                    seconds=time.monotonic() - started, optimizer_steps=int(self.optimizer.iterations))
