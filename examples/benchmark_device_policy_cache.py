"""Verify a GPU-resident policy-input cache on one frozen training-only round.

Every canonical batch is compared byte-for-byte with original collation, not
just model outputs. Three real batches compare one joint actor/critic backward
traversal. No optimizer steps, generated games, mutable source data or GPU trace.
"""
import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile

from examples.profile_ppo_update import checkpoint_view, select_training, sha, write


def run(args):
    settings = {
        'KERAS_BACKEND': 'tensorflow', 'CUDA_VISIBLE_DEVICES': '0',
        'TF_NUM_INTRAOP_THREADS': '1', 'TF_NUM_INTEROP_THREADS': '4',
        'TF_ENABLE_ONEDNN_OPTS': '0', 'TF_CPP_MIN_LOG_LEVEL': '2',
    }
    for key, value in settings.items():
        os.environ.setdefault(key, value)
    import keras
    import numpy as np
    import tensorflow as tf
    from examples.benchmark_fused_gru import compare_arrays, graph_ops
    from examples.train_rustmage import array_digest
    from patchwork_automaton.neural.batching import input_signature
    from patchwork_automaton.neural.devices import (
        assert_model_device, configure_runtime, device_scope,
    )
    from patchwork_automaton.neural.experiment import load_checkpoint, read_trajectory
    from patchwork_automaton.neural.features import collate
    from patchwork_automaton.neural.packed_rollouts import (
        DevicePolicyBatchCache, NumericRollout, PolicyBatchCache,
    )
    from patchwork_automaton.neural.ppo import normalized_advantages
    from patchwork_automaton.neural.rollout_cache import IndexedTrajectories, RolloutCache

    configure_runtime('gpu')
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    report_path = source / 'training/report.json'
    saved = json.loads(report_path.read_text())
    row, paths = select_training(saved, 40)
    train_ids = set(row['split']['training_indices'])
    held_ids = set(row['split']['validation_indices'])
    train_paths = {t['path'] for t in paths}
    held_paths = {t['path'] for g in row['games'] if g['index'] in held_ids
                  for t in g.get('trajectories', ())}
    assert len(train_ids) == 800 and len(held_ids) == 200
    assert not train_paths.intersection(held_paths)
    assert sum(t['samples'] for t in paths) == 85199
    checkpoint = source / 'training/checkpoints/round-0039'
    protected = {str(p): sha(p) for p in checkpoint.iterdir() if p.is_file()}
    protected[str(report_path)] = sha(report_path)
    implementation = Path(__file__).resolve().parents[1] / 'src/patchwork_automaton/neural'
    source_hashes = {str(p): sha(p) for p in implementation.rglob('*.py')}
    source_hashes[str(Path(__file__).resolve())] = sha(__file__)
    inputs = {t['path']: sha(t['path']) for t in paths}
    keras.utils.set_random_seed(saved['config']['seed'])
    with device_scope('gpu'), checkpoint_view(checkpoint) as view:
        model, trainer, _ = load_checkpoint(view, gru_backend='cudnn-memory')
    initial = [array_digest(v) for v in (
        model.weights, trainer.optimizer.variables, trainer.critic_optimizer.variables)]
    rng = copy.deepcopy(trainer.rng.bit_generator.state)
    result = dict(
        status='running', training_round=40, checkpoint_round=39,
        training_games=len(train_ids), heldout_games_excluded=len(held_ids),
        heldout_paths_disjoint=True, checkpoint_hashes=protected,
        source_hashes=source_hashes, input_hashes=inputs,
        canonical_batches=0, shuffled_batches=0, cases=[],
        placements=assert_model_device(model, 'gpu'),
        initial_state_sha256=dict(model=initial[0], actor_adam=initial[1], critic_adam=initial[2]),
        initial_pcg_state=rng,
        tolerances=dict(outputs=dict(atol=2e-5, rtol=2e-4),
                        gradients=dict(atol=2e-5, rtol=2e-3),
                        gradient_global_relative_l2=1e-3),
        no_optimizer_steps=True, model_and_rl_settings_unchanged=True,
    )
    write(args.output / 'report.json', result)
    store = NumericRollout()
    host_cache = PolicyBatchCache()
    device_cache = DevicePolicyBatchCache(device='/GPU:0', max_bytes=args.max_device_bytes)
    values, rewards, choices = [], [], []
    input_host_hash, input_device_hash = hashlib.sha256(), hashlib.sha256()
    logical_bytes = 0
    gpu_cases = []

    def records():
        for trajectory in paths:
            for sample in read_trajectory(trajectory['path']):
                yield sample, trajectory['reward']

    def reference(samples):
        x = collate([s.arrays for s in samples])
        x['memory_in'] = np.stack([s.memory for s in samples])
        x['advance_memory'] = np.asarray([[float(s.advance)] for s in samples], np.float32)
        old = np.full(x['candidate_mask'].shape, -1.e9, np.float32)
        for i, sample in enumerate(samples):
            old[i, :len(sample.log_probs)] = sample.log_probs
        return x, np.asarray([s.action for s in samples], np.int32), old

    def candidate(indices):
        x = store.gather(indices)
        old = x.pop('_old')
        actions = x.pop('_action')[:, 0]
        return x, actions, old

    def flattened(batch):
        return [*batch[0].values(), *batch[1:]]

    def assert_same(reference_batch, candidate_batch, *, on_gpu=False, digest=False):
        assert list(reference_batch[0]) == list(candidate_batch[0])
        left, right = flattened(reference_batch), flattened(candidate_batch)
        assert len(left) == len(right)
        for expected, actual in zip(left, right):
            if on_gpu:
                assert tf.is_tensor(actual), type(actual)
                assert actual.device.upper().endswith('/DEVICE:GPU:0'), actual.device
                actual = actual.numpy()  # One bounded batch at a time, never a full-cohort copy.
            assert expected.dtype == actual.dtype and expected.shape == actual.shape
            np.testing.assert_array_equal(expected, actual)
            if digest:
                descriptor = str((expected.dtype.str, expected.shape)).encode()
                input_host_hash.update(descriptor)
                input_host_hash.update(expected.tobytes())
                input_device_hash.update(descriptor)
                input_device_hash.update(actual.tobytes())

    def counts(batch):
        x = batch[0]
        return int((x['candidate_mask'].sum(axis=1) > 1).sum()), float(x['advance_memory'].sum())

    def cache_stats(cache):
        return dict(device=cache.device, max_bytes=cache.max_bytes,
                    bytes=cache.bytes, uploads=cache.uploads,
                    uploaded_tensors=cache.uploaded_tensors, hits=cache.hits,
                    entries=len(cache.entries), fallback_batches=cache.fallback_batches,
                    fallback_bytes=cache.fallback_bytes,
                    admission_closed=cache.admission_closed)

    try:
        for sample, reward in records():
            store.append(sample)
            values.append(sample.value)
            rewards.append(reward)
            choices.append(sample.arrays['candidate_mask'].sum() > 1)
        store.finish()
        assert store.count == 85199
        rewards = np.asarray(rewards, np.float32)
        actor_values = values if trainer.advantage_baseline == 'critic' else np.zeros(len(values), np.float32)
        advantages = normalized_advantages(rewards, actor_values, np.asarray(choices), trainer.advantage_scale_floor)
        stream = iter(records())
        start = 0
        while chunk := list(itertools.islice(stream, 512)):
            indices = np.arange(start, start + len(chunk))
            a = reference([s for s, _ in chunk])
            b = candidate(indices)
            assert_same(a, b)
            packed = (b[0], b[1], rewards[indices], b[2][np.arange(len(chunk)), b[1]],
                      advantages[indices], b[2])
            key = tuple(map(int, indices))
            host_cache.put(key, packed)
            admitted = device_cache.put(key, packed)
            assert admitted is not None, 'Frozen cohort must fit in declared device cache budget'
            hit = device_cache.get(key)
            assert hit is not None
            assert_same(packed, hit, on_gpu=True, digest=True)
            assert device_cache.counts(key) == counts(packed)
            logical_bytes += sum(v.nbytes for v in flattened(packed))
            if start in (0, 512 * (store.count // 1024)) or start + len(chunk) == store.count:
                gpu_cases.append((indices, a))
            start += len(chunk)
            result['canonical_batches'] += 1
        assert start == store.count and result['canonical_batches'] == 167
        host_cache.finish()
        device_cache.finish()
        assert device_cache.bytes == logical_bytes <= args.max_device_bytes
        assert len(device_cache.entries) == device_cache.uploads == 167
        assert device_cache.fallback_batches == device_cache.fallback_bytes == 0
        assert input_host_hash.hexdigest() == input_device_hash.hexdigest()
        # Noncanonical optimization orders are still exactly reconstructed; the
        # canonical device cache must not accidentally claim these shuffled keys.
        with tempfile.TemporaryFile() as spill:
            indexed = IndexedTrajectories(paths, RolloutCache(0), read_trajectory, spill)
            offsets = np.cumsum([0] + [t['samples'] for t in paths])
            try:
                order = np.random.default_rng(538126).permutation(store.count)
                for i in range(6):
                    indices = order[i * 512:(i + 1) * 512]
                    samples = []
                    for position in indices:
                        trajectory = np.searchsorted(offsets, position, side='right') - 1
                        samples.append(indexed.get(int(trajectory), int(position - offsets[trajectory])))
                    assert_same(reference(samples), candidate(indices))
                    assert device_cache.get(tuple(map(int, indices))) is None
                    result['shuffled_batches'] += 1
            finally:
                indexed.close()
        # A too-small budget admits no partial tensors. Once the deterministic
        # prefix closes, a later smaller batch also falls back to the CPU path.
        first_indices, _ = gpu_cases[0]
        first_key = tuple(map(int, first_indices))
        first = host_cache.get(first_key)
        assert_same(gpu_cases[0][1], (first[0], first[1], first[-1]))
        first_bytes = sum(v.nbytes for v in flattened(first))
        tiny = ({k: v[:1] for k, v in first[0].items()}, *[v[:1] for v in first[1:]])
        assert sum(v.nbytes for v in flattened(tiny)) < first_bytes - 1
        limited = DevicePolicyBatchCache(device='/GPU:0', max_bytes=first_bytes - 1)
        try:
            assert limited.put(first_key, first) is None
            assert limited.get(first_key) is None
            assert limited.counts(first_key) == counts(first)
            assert limited.put(('later-smaller-batch',), tiny) is None
            limited.finish()
            assert limited.get(('later-smaller-batch',)) is None
            assert limited.counts(('later-smaller-batch',)) == counts(tiny)
            assert limited.bytes == limited.uploads == len(limited.entries) == 0
            assert limited.fallback_batches == 2 and limited.admission_closed
            result['forced_fallback'] = cache_stats(limited)
            result['forced_fallback']['host_baseline_exact'] = True
        finally:
            limited.close()

        @tf.function(input_signature=[input_signature(gpu_cases[0][1][0]),
                     tf.TensorSpec([None], tf.int32),
                     *[tf.TensorSpec([None], tf.float32)] * 3], autograph=False)
        def evaluate(x, actions, returns, old, adv):
            ax = {k: tf.identity(v) for k, v in x.items()}
            cx = {k: tf.identity(v) for k, v in x.items()}
            watched = [mapping[k] for mapping in (ax, cx)
                       for k in ('entities', 'semantics', 'history', 'memory_in')]
            with tf.GradientTape() as tape:
                tape.watch(watched)
                actor = model.actor_output(ax, training=True)
                critic = model.value_output(cx, training=True)
                terms = trainer._terms_from_output(actor, tf.nn.log_softmax(actor['logits']),
                                                   x, actions, returns, old, adv)
                actor_loss = terms[1] - trainer.entropy_weight * terms[3]
                mask = x['advance_memory'][:, 0]
                critic_loss = tf.math.divide_no_nan(tf.reduce_sum(
                    tf.square(critic['value'][:, 0] - returns) * mask), tf.reduce_sum(mask))
                loss = actor_loss + critic_loss
            variables = model.actor.trainable_variables + model.critic.trainable_variables + watched
            gradients = tuple(tf.zeros_like(v) if g is None else tf.convert_to_tensor(g)
                              for g, v in zip(tape.gradient(loss, variables), variables))
            return (dict(logits=actor['logits'], probabilities=tf.nn.softmax(actor['logits']),
                         value=critic['value'], actor_memory=actor['memory_out'],
                         critic_memory=critic['memory_out'], actor_loss=actor_loss,
                         critic_loss=critic_loss), gradients)

        with device_scope('gpu'):
            for indices, reference_batch in gpu_cases:
                key = tuple(map(int, indices))
                host = host_cache.get(key)
                device = device_cache.get(key)
                assert_same(host, device, on_gpu=True)
                left = evaluate(*host[:5])
                right = evaluate(*device[:5])
                outputs = {k: compare_arrays(v, right[0][k], **result['tolerances']['outputs'])
                           for k, v in left[0].items()}
                gradients = [compare_arrays(u, v, **result['tolerances']['gradients'])
                             for u, v in zip(left[1], right[1])]
                all_a = np.concatenate([np.asarray(g).ravel() for g in left[1]])
                all_b = np.concatenate([np.asarray(g).ravel() for g in right[1]])
                total = compare_arrays(all_a, all_b, **result['tolerances']['gradients'])
                assert all(s['finite'] and s['close'] for s in [*outputs.values(), *gradients])
                assert total['relative_l2'] <= result['tolerances']['gradient_global_relative_l2']
                result['cases'].append(dict(first_index=int(indices[0]), count=len(indices),
                    outputs=outputs, gradient_variables=gradients,
                    gradient_max_absolute=max(s['max_absolute'] for s in gradients),
                    gradient_relative_l2=total['relative_l2']))
                del left, right
        operations = graph_ops(evaluate)
        assert operations.get('CudnnRNNV3') == operations.get('CudnnRNNBackpropV3') == 4
        result.update(
            status='passed', samples=store.count, numeric_disk_bytes=store.disk_bytes,
            host_policy_cache_disk_bytes=host_cache.disk_bytes,
            cache_input_logical_bytes=logical_bytes, device_cache=cache_stats(device_cache),
            canonical_host_sha256=input_host_hash.hexdigest(),
            canonical_device_sha256=input_device_hash.hexdigest(),
            cudnn_ops={k: v for k, v in operations.items() if 'Cudnn' in k},
            weights_and_optimizers_unchanged=initial == [array_digest(v) for v in (
                model.weights, trainer.optimizer.variables, trainer.critic_optimizer.variables)],
            rng_unchanged=rng == trainer.rng.bit_generator.state,
            source_unchanged=all(sha(p) == h for p, h in {**protected, **inputs, **source_hashes}.items()),
        )
        assert all(result[k] for k in ('weights_and_optimizers_unchanged', 'rng_unchanged', 'source_unchanged'))
        write(args.output / 'report.json', result)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        write(args.output / 'report.json', result)
        raise
    finally:
        # Device views do not reference the mapped files. Drop host views and
        # temporary readers before closing their read-only mmap backing.
        gpu_cases.clear()
        host_cache.close()
        device_cache.close()
        store.close()
    print(json.dumps({k: result[k] for k in (
        'status', 'samples', 'canonical_batches', 'shuffled_batches',
        'numeric_disk_bytes', 'host_policy_cache_disk_bytes', 'device_cache')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-device-bytes', type=int, default=8 * 1024**3)
    run(parser.parse_args())
