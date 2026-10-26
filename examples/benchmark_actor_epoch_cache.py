"""Frozen actor-epoch input-cache GPU correctness gate and timed replays.

Preflight clones the saved PCG, reserves both critic permutations, and validates
both real actor orders while the existing canonical policy cache stays on GPU.
Timed modes preserve every model/RL setting and save full state/guard evidence.
No games, source checkpoint writes, GPU traces or allocation retry/replay.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from examples.profile_ppo_update import checkpoint_view, select_training, sha, write
from examples.profile_packed_update import StorageProbe


def run_preflight(args):
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
        DevicePolicyBatchCache, NumericRollout,
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
        host_policy_cache_disk_bytes=0,
        validation_memory_repair='Omit redundant host PolicyBatchCache: canonical inputs are compared directly with original collation and GPU tensors; actor gradient cases use on-the-fly inputs.',
        gpu_value_validation='Resolve each retained GPU input through a distinct ephemeral same-device tf.identity handle; release its CPU array and handle after each exact comparison. This avoids deliberately materializing base cache handles; actual host-memory benefit is measured separately.',
    )
    gate = json.loads(args.policy_preflight_report.read_text())
    assert gate['status'] == 'passed'
    assert gate['input_hashes'] == inputs and gate['initial_pcg_state'] == rng
    assert gate['initial_state_sha256'] == result['initial_state_sha256']
    assert gate['weights_and_optimizers_unchanged'] and gate['rng_unchanged'] and gate['source_unchanged']
    result['policy_preflight_sha256'] = sha(args.policy_preflight_report)
    result['actor_epochs'] = []
    write(args.output / 'report.json', result)
    store = NumericRollout()
    device_cache = DevicePolicyBatchCache(device='/GPU:0', max_bytes=args.cache_budget_bytes)
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
            validation_copy = None
            value = actual
            if on_gpu:
                assert tf.is_tensor(actual), type(actual)
                assert actual.device.upper().endswith('/DEVICE:GPU:0'), actual.device
                # Materializing a retained TensorHandle itself can attach a
                # long-lived CPU mirror. Resolve an ephemeral GPU identity
                # handle instead; Identity preserves every input bit and the
                # original cache handle is never converted to a host array.
                with tf.device(actual.device):
                    validation_copy = tf.identity(actual)
                assert validation_copy is not actual
                assert validation_copy.device == actual.device
                value = validation_copy.numpy()
            try:
                assert expected.dtype == value.dtype and expected.shape == value.shape
                np.testing.assert_array_equal(expected, value)
                if digest:
                    descriptor = str((expected.dtype.str, expected.shape)).encode()
                    input_host_hash.update(descriptor)
                    input_host_hash.update(expected.tobytes())
                    input_device_hash.update(descriptor)
                    input_device_hash.update(value.tobytes())
            finally:
                # Release both the CPU materialization and its disposable
                # handle after EACH array, not after a full epoch/cohort.
                del value, validation_copy

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
        device_cache.finish()
        assert device_cache.bytes == logical_bytes <= args.cache_budget_bytes
        assert len(device_cache.entries) == device_cache.uploads == 167
        assert device_cache.fallback_batches == device_cache.fallback_bytes == 0
        assert input_host_hash.hexdigest() == input_device_hash.hexdigest()
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

        # Reserve critic permutations in exactly the original order, then derive
        # both actor epochs from a COPY. The live trainer PCG remains untouched.
        cloned_rng = np.random.default_rng()
        cloned_rng.bit_generator.state = copy.deepcopy(rng)
        assert trainer.separate_value and trainer.epochs == 2 and trainer.batch_size == 512
        critic_orders = [cloned_rng.permutation(store.count) for _ in range(trainer.epochs)]
        actor_orders = [cloned_rng.permutation(store.count) for _ in range(trainer.epochs)]
        order_sha = lambda order: hashlib.sha256(np.asarray(order, np.int64).tobytes()).hexdigest()
        result['reserved_critic_order_sha256'] = list(map(order_sha, critic_orders))
        result['actor_order_sha256'] = list(map(order_sha, actor_orders))
        result['cloned_pcg_after_full_update_orders'] = copy.deepcopy(cloned_rng.bit_generator.state)
        selected_cases = {(0, 0), (0, 83), (1, 166)}
        with tempfile.TemporaryFile() as spill:
            indexed = IndexedTrajectories(paths, RolloutCache(0), read_trajectory, spill)
            offsets = np.cumsum([0] + [t['samples'] for t in paths])
            try:
                for epoch, order in enumerate(actor_orders):
                    epoch_cache = DevicePolicyBatchCache(device='/GPU:0', max_bytes=args.actor_cache_budget_bytes)
                    row = dict(epoch=epoch+1, order_sha256=order_sha(order), batches=[],
                               total_input_bytes=0, maximum_batch_bytes=0,
                               record_count=0, expected_prefix_bytes=0,
                               expected_uploads=0, expected_fallback_batches=0,
                               expected_fallback_bytes=0)
                    closed_prefix = False
                    epoch_keys = []
                    original_digest, gathered_digest = hashlib.sha256(), hashlib.sha256()
                    try:
                        for batch_number, start in enumerate(range(0, store.count, 512)):
                            indices = order[start:start+512]
                            key = tuple(map(int, indices))
                            epoch_keys.append(key)
                            samples = []
                            for position in indices:
                                trajectory = np.searchsorted(offsets, position, side='right') - 1
                                samples.append(indexed.get(int(trajectory), int(position-offsets[trajectory])))
                            baseline = reference(samples)
                            packed_arrays = candidate(indices)
                            assert_same(baseline, packed_arrays)
                            x, actions, old = packed_arrays
                            packed = (x, actions, rewards[indices], old[np.arange(len(indices)), actions],
                                      advantages[indices], old)
                            for array in flattened((baseline[0], baseline[1], rewards[indices],
                                    baseline[2][np.arange(len(indices)),baseline[1]],
                                    advantages[indices], baseline[2])):
                                original_digest.update(str((array.dtype.str,array.shape)).encode())
                                original_digest.update(array.tobytes())
                            for array in flattened(packed):
                                gathered_digest.update(str((array.dtype.str,array.shape)).encode())
                                gathered_digest.update(array.tobytes())
                            size = sum(array.nbytes for array in flattened(packed))
                            result['shuffled_batches'] += 1
                            row['total_input_bytes'] += size
                            row['maximum_batch_bytes'] = max(row['maximum_batch_bytes'], size)
                            row['record_count'] += len(indices)
                            should_admit = not closed_prefix and row['expected_prefix_bytes'] + size <= args.actor_cache_budget_bytes
                            if should_admit:
                                row['expected_prefix_bytes'] += size
                                row['expected_uploads'] += 1
                            else:
                                closed_prefix = True
                                row['expected_fallback_batches'] += 1
                                row['expected_fallback_bytes'] += size
                            admitted = epoch_cache.put(key, packed)
                            assert (admitted is not None) == should_admit
                            cached = epoch_cache.get(key)
                            assert epoch_cache.counts(key) == counts(packed)
                            if should_admit:
                                assert cached is admitted
                                assert_same(packed, cached, on_gpu=True)
                                actual = cached
                            else:
                                assert cached is None
                                actual = packed
                            row['batches'].append(dict(batch=batch_number, records=len(indices),
                                input_bytes=size, admitted=should_admit,
                                indices_sha256=order_sha(indices)))
                            if (epoch,batch_number) in selected_cases:
                                with device_scope('gpu'):
                                    left, right = evaluate(*packed[:5]), evaluate(*actual[:5])
                                outputs = {k:compare_arrays(v,right[0][k],**result['tolerances']['outputs']) for k,v in left[0].items()}
                                gradients = [compare_arrays(a,b,**result['tolerances']['gradients']) for a,b in zip(left[1],right[1])]
                                aa = np.concatenate([np.asarray(g).ravel() for g in left[1]])
                                bb = np.concatenate([np.asarray(g).ravel() for g in right[1]])
                                total = compare_arrays(aa,bb,**result['tolerances']['gradients'])
                                assert all(s['finite'] and s['close'] for s in [*outputs.values(),*gradients])
                                assert total['relative_l2'] <= 1e-3
                                result['cases'].append(dict(epoch=epoch+1,batch=batch_number,
                                    count=len(indices),cache_admitted=should_admit,
                                    indices_sha256=order_sha(indices),outputs=outputs,
                                    gradient_variables=gradients,
                                    gradient_max_absolute=max(s['max_absolute'] for s in gradients),
                                    gradient_relative_l2=total['relative_l2']))
                                del left,right,aa,bb
                            # Do not retain device batches outside their owning
                            # epoch, especially after its deterministic prefix.
                            del admitted,cached,actual,packed,x,actions,old
                        epoch_cache.finish()
                        initial_uploads = epoch_cache.uploads
                        tensor = None
                        for key in epoch_keys:
                            cached = epoch_cache.get(key)
                            if cached is not None:
                                for tensor in flattened(cached):
                                    assert tensor.device.upper().endswith('/DEVICE:GPU:0')
                        del cached,tensor
                        assert epoch_cache.uploads == initial_uploads == row['expected_uploads']
                        assert epoch_cache.bytes == row['expected_prefix_bytes'] <= args.actor_cache_budget_bytes
                        assert epoch_cache.fallback_batches == row['expected_fallback_batches']
                        assert epoch_cache.fallback_bytes == row['expected_fallback_bytes']
                        assert epoch_cache.bytes+epoch_cache.fallback_bytes == row['total_input_bytes']
                        assert row['record_count'] == 85199 and len(row['batches']) == 167
                        assert original_digest.hexdigest() == gathered_digest.hexdigest()
                        row.update(cache=cache_stats(epoch_cache),original_input_sha256=original_digest.hexdigest(),
                                   gathered_input_sha256=gathered_digest.hexdigest())
                    finally:
                        epoch_cache.close()
                        row['released_before_next_epoch'] = epoch_cache.closed and epoch_cache.bytes == 0 and not epoch_cache.entries
                    assert row['released_before_next_epoch']
                    result['actor_epochs'].append(row)
                    write(args.output/'report.json',result)
                result['original_record_spill_bytes'] = indexed.disk_bytes
            finally:
                indexed.close()
        operations = graph_ops(evaluate)
        assert operations.get('CudnnRNNV3') == operations.get('CudnnRNNBackpropV3') == 4
        assert len(result['cases']) == 3 and result['cases'][-1]['count'] == 207
        assert result['shuffled_batches'] == 334
        assert device_cache.bytes == gate['cache_input_logical_bytes']
        result.update(status='passed',samples=store.count,numeric_disk_bytes=store.disk_bytes,
            host_policy_cache_disk_bytes=0,policy_device_cache=cache_stats(device_cache),
            cache_input_logical_bytes=logical_bytes,
            canonical_host_sha256=input_host_hash.hexdigest(),canonical_device_sha256=input_device_hash.hexdigest(),
            maximum_actor_epoch_input_bytes=max(row['total_input_bytes'] for row in result['actor_epochs']),
            maximum_actor_batch_bytes=max(row['maximum_batch_bytes'] for row in result['actor_epochs']),
            cudnn_ops={k:v for k,v in operations.items() if 'Cudnn' in k},
            weights_and_optimizers_unchanged=initial==[array_digest(v) for v in (model.weights,trainer.optimizer.variables,trainer.critic_optimizer.variables)],
            rng_unchanged=rng==trainer.rng.bit_generator.state,
            source_unchanged=all(sha(p)==h for p,h in {**protected,**inputs,**source_hashes}.items()))
        assert all(result[k] for k in ('weights_and_optimizers_unchanged','rng_unchanged','source_unchanged'))
        write(args.output/'report.json',result)
    except BaseException as error:
        result.update(status='failed',error=repr(error));write(args.output/'report.json',result);raise
    finally:
        gpu_cases.clear();device_cache.close();store.close()
    print(json.dumps(dict(status=result['status'],actor_epochs=[{k:row[k] for k in ('epoch','total_input_bytes','maximum_batch_bytes','expected_uploads','expected_prefix_bytes','expected_fallback_batches','expected_fallback_bytes','released_before_next_epoch')} for row in result['actor_epochs']])),flush=True)


def run_timed(args):
    if args.cache_budget_bytes != 8 * 1024**3:
        raise ValueError('This benchmark uses the preregistered 8 GiB device-input cap')
    settings = {
        'KERAS_BACKEND': 'tensorflow', 'CUDA_VISIBLE_DEVICES': '0',
        'TF_NUM_INTRAOP_THREADS': '1', 'TF_NUM_INTEROP_THREADS': '4',
        'TF_ENABLE_ONEDNN_OPTS': '0', 'TF_CPP_MIN_LOG_LEVEL': '2',
    }
    for key, value in settings.items():
        os.environ.setdefault(key, value)
    from patchwork_automaton.neural.devices import configure_runtime, device_scope, assert_model_device
    configure_runtime('gpu')
    import keras
    import numpy as np
    from examples.benchmark_fused_gru import graph_ops
    from examples.train_rustmage import array_digest, neural_call, place_trainer
    from patchwork_automaton.neural import streaming_ppo
    from patchwork_automaton.neural.experiment import load_checkpoint
    from patchwork_automaton.neural.recurrent import gru_backend_summary

    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Output must be outside the source study')
    output.mkdir(parents=True, exist_ok=False)
    report_path = source / 'training/report.json'
    saved = json.loads(report_path.read_text())
    row, trajectories = select_training(saved, 40)
    assert len(row['split']['training_indices']) == 800
    assert len(row['split']['validation_indices']) == 200
    assert sum(t['samples'] for t in trajectories) == 85199
    checkpoint = source / 'training/checkpoints/round-0039'
    source_hashes = {str(report_path): sha(report_path),
                     **{str(p): sha(p) for p in checkpoint.iterdir() if p.is_file()}}
    input_hashes = {t['path']: sha(t['path']) for t in trajectories}
    implementation = Path(__file__).resolve().parents[1] / 'src/patchwork_automaton/neural'
    implementation_hashes = {str(p): sha(p) for p in implementation.rglob('*.py')}
    implementation_hashes[str(Path(__file__).resolve())] = sha(__file__)
    keras.utils.set_random_seed(saved['config']['seed'])
    with device_scope('gpu'), checkpoint_view(checkpoint) as view:
        model, trainer, state = load_checkpoint(view, gru_backend='cudnn-memory')
        layers = gru_backend_summary(model)
        assert sum(layer['use_cudnn'] is True for layer in layers) == 4
        assert sum(layer['use_cudnn'] is False for layer in layers) == 2
        kernels = {name: graph_ops(getattr(trainer, name))
                   for name in ('step', 'critic_step', 'measure_actor_with_drift')}
        if not all(any('CudnnRNN' in key for key in ops) for ops in kernels.values()):
            raise RuntimeError('Selective cuDNN graph path missing')
        placements = assert_model_device(model, 'gpu')
        traces = place_trainer(trainer, 'gpu')
    assert state['round'] == 39 and state['config'] == saved['config']
    trainer.numerical_prepacking = True
    trainer.cache_policy_batches = True
    trainer.policy_cache_device = '/GPU:0'
    trainer.actor_epoch_cache_device = '/GPU:0' if args.mode == 'epoch-cache' else None
    trainer.actor_epoch_cache_bytes = args.actor_cache_budget_bytes if args.mode == 'epoch-cache' else 0
    trainer.policy_cache_device_bytes = args.cache_budget_bytes if args.cache_placement == 'gpu' else 0
    initial_rng = copy.deepcopy(trainer.rng.bit_generator.state)
    initial_hashes = dict(model=array_digest(model.weights),
                          actor_adam=array_digest(trainer.optimizer.variables),
                          critic_adam=array_digest(trainer.critic_optimizer.variables))
    preflight = None
    if args.preflight_report:
        preflight = json.loads(args.preflight_report.read_text())
        assert preflight['status'] == 'passed'
        assert preflight['training_games'] == 800 and preflight['heldout_games_excluded'] == 200
        assert preflight['samples'] == 85199 and preflight['canonical_batches'] == 167
        assert preflight['input_hashes'] == input_hashes
        assert preflight['initial_state_sha256'] == initial_hashes
        assert preflight['initial_pcg_state'] == initial_rng
        assert preflight['weights_and_optimizers_unchanged'] and preflight['rng_unchanged']
        assert preflight['source_unchanged']

    actor_gate = json.loads(args.actor_preflight_report.read_text())
    assert actor_gate['status'] == 'passed' and len(actor_gate['actor_epochs']) == 2
    assert actor_gate['policy_preflight_sha256'] == sha(args.policy_preflight_report)
    assert actor_gate['input_hashes'] == input_hashes
    assert actor_gate['initial_state_sha256'] == initial_hashes and actor_gate['initial_pcg_state'] == initial_rng
    assert actor_gate['weights_and_optimizers_unchanged'] and actor_gate['rng_unchanged'] and actor_gate['source_unchanged']
    assert all(row['released_before_next_epoch'] for row in actor_gate['actor_epochs'])
    decisions = []
    original_guard = streaming_ppo._guard_status

    def observe_guard(*arguments, **keywords):
        status = original_guard(*arguments, **keywords)
        decisions.append(dict(accepted=status[0], reasons=status[2],
                              mean_kl=float(arguments[1][0]),
                              max_state_kl=float(arguments[1][1])))
        return status

    info = dict(
        status='running', mode='cudnn-memory', packing_variant='both',
        cache_placement=args.cache_placement, cache_budget_bytes=args.cache_budget_bytes,
        actor_cache_mode=args.mode, actor_cache_budget_bytes=args.actor_cache_budget_bytes,
        actor_preflight_report=str(args.actor_preflight_report),
        actor_preflight_sha256=sha(args.actor_preflight_report),
        effective_device_cache_budget_bytes=trainer.policy_cache_device_bytes,
        source=str(source), source_round=40, source_hashes=source_hashes,
        implementation_hashes=implementation_hashes, input_hashes=input_hashes,
        training_trajectories=len(trajectories), training_games=800,
        heldout_games_excluded=200, config=state['config'], device='gpu',
        gru_layers=layers, kernel_ops=kernels, variable_placements=placements,
        graph_traces=traces, initial_rng=initial_rng,
        initial_state_sha256=initial_hashes, initial_model_sha256=initial_hashes['model'],
        initial_actor_optimizer_steps=int(trainer.optimizer.iterations),
        initial_critic_optimizer_steps=int(trainer.critic_optimizer.iterations),
        preflight_report=str(args.preflight_report) if args.preflight_report else None,
        preflight_sha256=sha(args.preflight_report) if args.preflight_report else None,
        runtime={key: os.environ.get(key) for key in settings},
        full_gpu_trace=False,
        placement_validation='DevicePolicyBatchCache.put validates every admitted tensor device; correctness preflight checks every value.',
    )
    write(output / 'report.json', info)
    streaming_ppo._guard_status = observe_guard
    # Both variants use the same passive sampling. No timing wrappers are added
    # to the inner update, no OS caches are flushed and no GPU trace is enabled.
    probe = StorageProbe(trace_dir=None, trace_seconds=0)
    metrics_file = (output / 'nvidia-smi.csv').open('w')
    metrics_error = (output / 'nvidia-smi.stderr.log').open('w')
    try:
        monitor = subprocess.Popen([
            '/usr/lib/wsl/lib/nvidia-smi',
            '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem',
            '--format=csv,noheader,nounits', '--loop-ms=500'],
            stdout=metrics_file, stderr=metrics_error)
    except FileNotFoundError:
        monitor = None
    started = time.perf_counter()
    cpu_started = time.process_time()
    info['update_started_wall_ns'] = time.time_ns()
    try:
        with probe.span('full_update'):
            result = neural_call('gpu', trainer.update_files, trajectories)
        elapsed = time.perf_counter() - started
        cpu_elapsed = time.process_time() - cpu_started
        final_arrays = {}
        for label, variables in (
                ('model', model.weights), ('actor_optimizer', trainer.optimizer.variables),
                ('critic_optimizer', trainer.critic_optimizer.variables)):
            final_arrays.update({f'{label}_{i}': np.asarray(v) for i, v in enumerate(variables)})
        np.savez(output / 'result-state.npz', **final_arrays)
        info.update(
            update_seconds=elapsed, process_cpu_seconds=cpu_elapsed, result=result,
            final_rng=trainer.rng.bit_generator.state,
            final_model_sha256=array_digest(model.weights),
            final_actor_optimizer_steps=int(trainer.optimizer.iterations),
            final_critic_optimizer_steps=int(trainer.critic_optimizer.iterations),
            final_state_sha256=dict(model=array_digest(model.weights),
                                   actor_adam=array_digest(trainer.optimizer.variables),
                                   critic_adam=array_digest(trainer.critic_optimizer.variables)),
        )
        # Cache counters are captured by the trainer before ExitStack cleanup,
        # while tensor residency and its full frozen-input payload still exist.
        pipeline = result['pipeline']
        info['policy_input_cache'] = {key: value for key, value in pipeline.items()
                                      if key.startswith('policy_cache_')}
        assert result['samples'] == 85199
        assert pipeline['numerical_prepacking'] and pipeline['cache_policy_batches']
        if args.cache_placement == 'gpu':
            assert pipeline['policy_cache_device'] == '/GPU:0'
            assert pipeline['policy_cache_device_budget_bytes'] == args.cache_budget_bytes
            assert pipeline['policy_cache_device_uploads'] == 167
            assert pipeline['policy_cache_device_tensors'] > 167
            assert 6_000_000_000 < pipeline['policy_cache_device_bytes'] < 6_500_000_000
            assert pipeline['policy_cache_device_bytes'] <= args.cache_budget_bytes
            assert pipeline['policy_cache_device_hits'] >= 167
            assert pipeline['policy_cache_device_fallback_batches'] == 0
            assert pipeline['policy_cache_device_fallback_bytes'] == 0
            assert pipeline['policy_cache_disk_bytes'] == 0
            if preflight:
                assert pipeline['policy_cache_device_bytes'] == preflight['cache_input_logical_bytes']
        else:
            # The frozen prior host implementation predates device-residency
            # telemetry. Missing device-only fields mean the feature is absent;
            # retain its raw pipeline report and require every present counter
            # to be zero. GPU candidate telemetry remains mandatory above.
            assert pipeline.get('policy_cache_device') is None
            assert pipeline.get('policy_cache_device_budget_bytes', 0) == 0
            assert all(pipeline.get(key, 0) == 0 for key in (
                'policy_cache_device_bytes', 'policy_cache_device_uploads',
                'policy_cache_device_tensors', 'policy_cache_device_hits',
                'policy_cache_device_fallback_batches', 'policy_cache_device_fallback_bytes'))
            assert pipeline['policy_cache_disk_bytes'] == 6_229_903_504
            assert pipeline['policy_cache_hits'] >= 167
        actor_rows = pipeline.get('actor_epoch_cache_epochs', [])
        info['actor_epoch_input_cache'] = {key:value for key,value in pipeline.items() if key.startswith('actor_epoch_cache_')}
        if args.mode == 'epoch-cache':
            assert pipeline['actor_epoch_cache_device'] == '/GPU:0'
            assert pipeline['actor_epoch_cache_budget_bytes'] == args.actor_cache_budget_bytes
            assert 1 <= len(actor_rows) <= 2
            if len(actor_rows) != 2: assert result['early_stop']
            for actual, expected in zip(actor_rows, actor_gate['actor_epochs']):
                assert actual['epoch'] == expected['epoch']
                assert actual['device'] == '/GPU:0' and actual['budget_bytes'] == args.actor_cache_budget_bytes
                assert actual['released']
                assert actual['logical_bytes'] == expected['expected_prefix_bytes'] <= args.actor_cache_budget_bytes
                assert actual['uploads'] == expected['expected_uploads']
                assert actual['fallback_batches'] == expected['expected_fallback_batches']
                assert actual['fallback_bytes'] == expected['expected_fallback_bytes']
                assert actual['unique_input_bytes'] == expected['total_input_bytes']
                assert actual['uploads'] + actual['fallback_batches'] == 167
            assert pipeline['actor_epoch_cache_max_logical_bytes'] <= args.actor_cache_budget_bytes
        else:
            assert pipeline.get('actor_epoch_cache_device') is None
            assert pipeline.get('actor_epoch_cache_budget_bytes', 0) == 0
            assert actor_rows == []
            assert all(pipeline.get(key, 0) == 0 for key in (
                'actor_epoch_cache_uploads','actor_epoch_cache_hits',
                'actor_epoch_cache_fallback_batches','actor_epoch_cache_fallback_bytes',
                'actor_epoch_cache_max_logical_bytes','actor_epoch_cache_max_epoch_input_bytes'))
        info.update(status='complete', cache_admission_verified=True,actor_cache_admission_verified=True)
    except BaseException as error:
        info.update(status='error', error=repr(error))
        raise
    finally:
        if monitor:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        metrics_file.close()
        metrics_error.close()
        streaming_ppo._guard_status = original_guard
        info['guard_decisions'] = decisions
        info['profiling'] = probe.finish(output)
        info['source_unchanged'] = all(sha(path) == digest for path, digest in {
            **source_hashes, **input_hashes, **implementation_hashes}.items())
        info['finished_utc'] = datetime.now(timezone.utc).isoformat()
        if not info['source_unchanged']:
            info.update(status='error', error='Read-only source changed during frozen replay')
        write(output / 'report.json', info)
    if not info['source_unchanged']:
        raise ValueError('Read-only source changed during frozen replay')
    print(json.dumps({key: info[key] for key in (
        'status', 'cache_placement', 'update_seconds', 'process_cpu_seconds',
        'final_model_sha256', 'source_unchanged', 'policy_input_cache','actor_epoch_input_cache')}), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('preflight','baseline','epoch-cache'),required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy-preflight-report',type=Path,required=True)
    parser.add_argument('--actor-preflight-report',type=Path)
    parser.add_argument('--cache-budget-bytes',type=int,default=8*1024**3)
    parser.add_argument('--actor-cache-budget-bytes',type=int,default=8*1024**3)
    args=parser.parse_args()
    if args.cache_budget_bytes != 8*1024**3 or args.actor_cache_budget_bytes != 8*1024**3:
        parser.error('Both logical GPU input caps are fixed at 8 GiB')
    args.cache_placement='gpu'
    args.preflight_report=args.policy_preflight_report
    if args.mode=='preflight':
        run_preflight(args)
    else:
        if not args.actor_preflight_report:
            parser.error('Timed replays require the completed actor preflight')
        run_timed(args)
