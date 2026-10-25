"""Bounded ordered critic TBPTT; actor behavior statistics never enter this path.

Each trajectory starts from zero. Memory carries between optimizer chunks and
is detached there, as in standard truncated BPTT. A carried boundary state was
computed immediately before the preceding parameter update; this is not an
exact full-prefix replay under the newest weights.
"""
import numpy as np
import tensorflow as tf
from .features import collate


def sequence_step(trainer, batches, memory, returns, valid):
    """One critic-only update; differentiate through all steps within a chunk."""
    with tf.GradientTape() as tape:
        squared = tf.constant(0., tf.float32)
        samples = tf.constant(0., tf.float32)
        for step,inputs in enumerate(batches):
            output = trainer.model.critic(dict(inputs, memory_in=memory), training=True)
            mask = valid[step] * inputs['advance_memory'][:, 0]
            squared += tf.reduce_sum(tf.square(output['value'][:, 0]-returns) * mask)
            samples += tf.reduce_sum(mask)
            memory = tf.where(valid[step, :, None] > 0., output['memory_out'], memory)
        loss = tf.math.divide_no_nan(squared, samples)
    variables = (trainer.model.critic.value_head.trainable_variables
                 if trainer.critic_update_scope == 'head' else trainer.model.critic.trainable_variables)
    gradients = tape.gradient(loss, variables)
    gradients = [tf.zeros_like(v) if g is None else tf.convert_to_tensor(g)
                 for g,v in zip(gradients, variables)]
    for gradient in gradients:
        tf.debugging.assert_all_finite(gradient, 'Nonfinite sequence critic gradient')
    tf.debugging.assert_all_finite(memory, 'Nonfinite sequence critic memory')
    tf.debugging.assert_all_finite(loss, 'Nonfinite sequence critic loss')
    gradients,norm = tf.clip_by_global_norm(gradients, trainer.clip_norm)
    trainer.critic_optimizer.apply_gradients(zip(gradients, variables))
    return norm,tf.stop_gradient(memory),samples,loss


def fit_sequence_critic(trainer, trajectories, orders, get_record):
    """Train only supplied trajectories, retaining at most one decision batch.

The decision batch budget is shared with detached fitting: floor(batch/unroll)
trajectory streams, each contributing at most unroll records per update.
Existing pre-drawn critic permutations determine trajectory order without
consuming extra trainer RNG or changing actor minibatch permutations.
"""
    width = max(1, trainer.batch_size // trainer.critic_unroll)
    offsets = np.cumsum([0]+[t['samples'] for t in trajectories[:-1]])
    steps = positions = samples = padded = 0
    norm = loss = 0.
    for _ in range(trainer.epochs):
        order = np.asarray(next(orders))
        identities = np.searchsorted(offsets, order, side='right')-1
        _,first = np.unique(identities, return_index=True)
        pending = iter(identities[np.sort(first)].tolist())
        streams = []
        while True:
            streams = [s for s in streams if s['position'] < trajectories[s['trajectory']]['samples']]
            while len(streams) < width:
                identity = next(pending, None)
                if identity is None: break
                streams.append(dict(trajectory=int(identity), position=0,
                    memory=np.zeros(trainer.model.critic.memory_size, np.float32)))
            if not streams: break
            batches = []
            valid = np.zeros((trainer.critic_unroll, len(streams)), np.float32)
            # Load each numeric record once; all decoder access goes through
            # the caller's shared packing lock. No complete game is retained.
            records = []
            for column,stream in enumerate(streams):
                identity = stream['trajectory']
                count = min(trainer.critic_unroll,
                            trajectories[identity]['samples']-stream['position'])
                records.append([get_record(identity, stream['position']+i) for i in range(count)])
                valid[:count, column] = 1.
            for step in range(trainer.critic_unroll):
                selected = [rows[min(step,len(rows)-1)] for rows in records]
                inputs = collate([s.arrays for s in selected])
                inputs['advance_memory'] = np.asarray([[float(s.advance)] for s in selected], np.float32)
                batches.append(inputs)
            memory = np.stack([s['memory'] for s in streams])
            returns = np.asarray([trajectories[s['trajectory']]['reward'] for s in streams], np.float32)
            norm,updated,used,loss = trainer.critic_sequence_step(tuple(batches),memory,returns,valid)
            norm,loss = float(norm),float(loss)
            updated = np.asarray(updated)
            for stream,state in zip(streams,updated):
                stream['position'] += trainer.critic_unroll
                stream['memory'] = state.copy()
            positions += int(valid.sum())
            padded += valid.size-int(valid.sum())
            samples += int(used)
            steps += 1
            if trainer.value_max_steps is not None and steps >= trainer.value_max_steps: break
        if trainer.value_max_steps is not None and steps >= trainer.value_max_steps: break
    return dict(critic_steps=steps, critic_optimizer_steps=int(trainer.critic_optimizer.iterations),
        critic_gradient_norm=norm, value_max_steps=trainer.value_max_steps,
        value_step_limit_reached=trainer.value_max_steps is not None and steps >= trainer.value_max_steps,
        value_learning_rate=float(trainer.critic_optimizer.learning_rate.numpy()),
        critic_update_scope=trainer.critic_update_scope, critic_unroll=trainer.critic_unroll,
        critic_sequence_batch=width, critic_sequence_positions=positions,
        critic_value_samples=samples, critic_sequence_padding=padded,
        critic_sequence_last_loss=loss,
        critic_recurrent_training='zero at trajectory start; ordered TBPTT; detached carried chunk boundaries')
