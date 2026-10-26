"""Five persistent-runtime actor-cache updates over saved rounds 36 through 40.

This is a finite runtime/cache-lifecycle replay, not new learning evidence.
Checkpoint 35 is loaded once; weights, both Adam states, PCG and compiled graphs
continue without resets. Later saved behavior policies differ from the evolving
replay policy. No games, checkpoint writes, GPU traces, per-update garbage
collection, allocator flushes or allocation-failure retry/replay are performed.
"""
import argparse
import copy
from datetime import datetime, timezone
import functools
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import weakref

from examples.profile_ppo_update import checkpoint_view, select_training, sha, write
from examples.profile_packed_update import StorageProbe, os_snapshot


ROUNDS = (36, 37, 38, 39, 40)
EXPECTED_SAMPLES = (87283, 85957, 87154, 87500, 85199)
CACHE_BYTES = 8 * 1024**3
PLATEAU_BYTES = 256 * 1024**2
HEADROOM_BYTES = 2 * 1024**3


def ppo_threads():
    return [dict(name=t.name, ident=t.ident) for t in threading.enumerate()
            if t.is_alive() and t.name.startswith('ppo-')]


class LifetimeObserver:
    """Process-local diagnostic hooks retaining weak references and scalars only.

    A close hook observes every retained input tensor before the real close
    clears the cache. Boundary inspection never materializes tensors or copies
    payloads. It distinguishes a closed Python shell from an open reader or a
    surviving cached Tensor object; allocator reservation is measured separately.
    """
    def __init__(self):
        self.round = None
        self.rows = []
        self.tensor_refs = []
        self.originals = []
        self.lock = threading.Lock()

    def __enter__(self):
        import gzip
        from patchwork_automaton.neural.packed_rollouts import (
            DevicePolicyBatchCache, NumericRollout, PolicyBatchCache,
        )
        from patchwork_automaton.neural.rollout_cache import IndexedTrajectories, RolloutCache
        from patchwork_automaton.neural.indexed_rollouts import IndexedTrajectoryReader
        for cls in (DevicePolicyBatchCache, NumericRollout, PolicyBatchCache,
                    IndexedTrajectories, RolloutCache, IndexedTrajectoryReader, gzip.GzipFile):
            original_init = cls.__init__
            label = cls.__name__

            @functools.wraps(original_init)
            def init(instance, *args, _original=original_init, _label=label, **kwargs):
                row = dict(round=self.round, kind=_label, ref=weakref.ref(instance),
                           initialized=False, close_calls=0, closed_verified=None)
                with self.lock:
                    self.rows.append(row)
                # IndexedTrajectoryReader.__init__ closes itself on invalid
                # data; register before initialization so that original failure
                # remains visible rather than being masked by this observer.
                _original(instance, *args, **kwargs)
                row['initialized'] = True

            self.originals.append((cls, '__init__', original_init))
            cls.__init__ = init
            if not hasattr(cls, 'close'):
                continue
            original_close = cls.close

            @functools.wraps(original_close)
            def close(instance, *args, _original=original_close, _label=label, **kwargs):
                with self.lock:
                    row = next(r for r in reversed(self.rows) if r['ref']() is instance)
                if _label == 'DevicePolicyBatchCache':
                    row['before_close'] = dict(device=instance.device, budget_bytes=instance.max_bytes,
                        logical_bytes=instance.bytes, uploads=instance.uploads, hits=instance.hits,
                        fallback_batches=instance.fallback_batches, fallback_bytes=instance.fallback_bytes)
                    # Do not retain an entry, tensor, bound method or traceback.
                    # All input tensors stay in entries until the real close.
                    refs = [weakref.ref(tensor) for batch in instance.entries.values()
                            for tensor in (*batch[0].values(), *batch[1:])]
                    with self.lock:
                        self.tensor_refs.extend((row['round'], ref) for ref in refs)
                    del refs
                value = _original(instance, *args, **kwargs)
                row['close_calls'] += 1
                row['closed_verified'] = self.closed(instance, _label)
                return value

            self.originals.append((cls, 'close', original_close))
            cls.close = close
        return self

    @staticmethod
    def closed(instance, kind):
        if kind == 'DevicePolicyBatchCache':
            return instance.closed and instance.bytes == 0 and not instance.entries
        if kind in ('NumericRollout', 'PolicyBatchCache'):
            return instance.mapping is None and instance.file.closed
        if kind == 'IndexedTrajectories':
            return not instance.numeric_readers
        if kind == 'IndexedTrajectoryReader':
            return instance.fd is None
        if kind == 'GzipFile':
            return instance.closed
        return not instance.entries  # RolloutCache has no close: it must expire.

    def boundary(self, round_id):
        rows = [r for r in self.rows if r['round'] == round_id]
        summary = {}
        for kind in sorted({r['kind'] for r in rows}):
            selected = [r for r in rows if r['kind'] == kind]
            alive = open_count = 0
            for row in selected:
                instance = row['ref']()
                if instance is not None:
                    alive += 1
                    open_count += not self.closed(instance, kind)
                del instance
            summary[kind] = dict(created=len(selected), alive_shells=alive,
                open_or_retained_payloads=open_count,
                incomplete_initializations=sum(not r['initialized'] for r in selected),
                close_calls=sum(r['close_calls'] for r in selected),
                close_failures=sum(r['closed_verified'] is False for r in selected))
        cached = [r for r in rows if r['kind'] == 'DevicePolicyBatchCache']
        tensors = [ref for r, ref in self.tensor_refs if r == round_id]
        result = dict(round=round_id, resources=summary,
            cached_tensor_objects_observed=len(tensors),
            cached_tensor_objects_alive=sum(ref() is not None for ref in tensors),
            device_cache_closes=[r.get('before_close') for r in cached],
            ppo_threads=ppo_threads())
        result['released'] = (bool(cached) and not result['ppo_threads']
            and result['cached_tensor_objects_alive'] == 0
            and all(r['close_calls'] == 1 and r['closed_verified'] for r in cached)
            and all(v['open_or_retained_payloads'] == v['close_failures'] == v['incomplete_initializations'] == 0
                    for v in summary.values()))
        return result

    def __exit__(self, *args):
        for cls, name, original in reversed(self.originals):
            setattr(cls, name, original)


def variable_groups(model, trainer):
    return dict(model=model.weights, actor_adam=trainer.optimizer.variables,
                critic_adam=trainer.critic_optimizer.variables)


def state_summary(model, trainer):
    from examples.train_rustmage import array_digest
    return dict(sha256={name: array_digest(values)
                       for name, values in variable_groups(model, trainer).items()},
                actor_steps=int(trainer.optimizer.iterations),
                critic_steps=int(trainer.critic_optimizer.iterations),
                pcg=copy.deepcopy(trainer.rng.bit_generator.state))


def check_pipeline(result, trainer, expected_samples, cap, actor_cap, decisions):
    """Check budgets and complete passes without assuming equal guard work."""
    import numpy as np
    pipeline = result['pipeline']
    batches = (expected_samples + trainer.batch_size - 1) // trainer.batch_size
    assert result['samples'] == expected_samples
    assert pipeline['numerical_prepacking'] and pipeline['cache_policy_batches']
    assert pipeline['policy_cache_device'] == pipeline['actor_epoch_cache_device'] == '/GPU:0'
    assert pipeline['policy_cache_device_budget_bytes'] == cap
    assert pipeline['actor_epoch_cache_budget_bytes'] == actor_cap
    assert pipeline['policy_cache_device_bytes'] <= cap
    assert pipeline['policy_cache_device_uploads'] + pipeline['policy_cache_device_fallback_batches'] == batches
    epochs = pipeline['actor_epoch_cache_epochs']
    assert 1 <= len(epochs) <= trainer.epochs
    if len(epochs) != trainer.epochs:
        assert result['early_stop']
    consumed = 0
    for epoch, row in enumerate(epochs, 1):
        assert row['epoch'] == epoch and row['device'] == '/GPU:0'
        assert row['budget_bytes'] == actor_cap and row['logical_bytes'] <= actor_cap
        assert row['released'] and row['uploads'] + row['fallback_batches'] == batches
        assert row['unique_input_bytes'] == row['logical_bytes'] + row['fallback_bytes']
        passes = 0
        for _ in range(trainer.backtracks + 1):
            assert consumed < len(decisions)
            decision = decisions[consumed]
            consumed += 1; passes += 1
            if decision['accepted']:
                break
        assert row['hits'] == row['uploads'] * (passes-1)
    assert consumed == len(decisions)
    assert pipeline['actor_epoch_cache_max_logical_bytes'] <= actor_cap
    assert result['optimizer_steps'] == int(trainer.optimizer.iterations)
    assert result['critic_optimizer_steps'] == int(trainer.critic_optimizer.iterations)
    assert np.isfinite([result[k] for k in ('loss', 'value_loss', 'exact_kl', 'max_state_kl')]).all()


def nvidia_summary(path, before_wall_ns, after_wall_ns):
    """Cumulative raw sampling; do not gate on adjusted Windows timestamps."""
    rows = []
    for line in Path(path).read_text().splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) != 8:
            continue
        try:
            rows.append(dict(used_mib=float(fields[3]), total_mib=float(fields[4]),
                             activity_percent=float(fields[1])))
        except ValueError:
            continue
    if not rows:
        return dict(available=False, samples=0, headroom_gate_passed=False)
    minimum = min(r['total_mib'] - r['used_mib'] for r in rows) * 1024**2
    return dict(available=True, samples=len(rows), minimum_headroom_bytes=int(minimum),
        maximum_used_mib=max(r['used_mib'] for r in rows),
        total_mib=min(r['total_mib'] for r in rows),
        mean_activity_percent=sum(r['activity_percent'] for r in rows)/len(rows),
        headroom_gate_passed=minimum >= HEADROOM_BYTES,
        started_wall_ns=before_wall_ns, ended_wall_ns=after_wall_ns,
        limitations='Raw cumulative samples include state evidence reads between updates. Windows NVIDIA timestamps can step backwards; use monotonic update durations. Activity is not FLOPS or occupancy; sampling may miss short peaks.')


def run(args):
    if args.policy_cache_bytes != CACHE_BYTES or args.actor_cache_bytes != CACHE_BYTES:
        raise ValueError('Both logical input budgets are fixed at 8 GiB')
    settings = dict(KERAS_BACKEND='tensorflow', CUDA_VISIBLE_DEVICES='0',
        TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='4',
        TF_ENABLE_ONEDNN_OPTS='0', TF_CPP_MIN_LOG_LEVEL='2')
    for name, value in settings.items():
        os.environ.setdefault(name, value)
        if os.environ[name] != value:
            raise ValueError(f'Frozen runtime environment mismatch for {name}: {os.environ[name]!r} != {value!r}')
    from patchwork_automaton.neural.devices import configure_runtime, device_scope, assert_model_device
    configure_runtime('gpu')
    import keras
    import numpy as np
    import tensorflow as tf
    from examples.benchmark_fused_gru import graph_ops
    from examples.train_rustmage import neural_call, place_trainer
    from patchwork_automaton.neural import streaming_ppo
    from patchwork_automaton.neural.experiment import load_checkpoint
    from patchwork_automaton.neural.recurrent import gru_backend_summary

    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Output must be outside the source study')
    output.mkdir(parents=True, exist_ok=False)
    report_path = source/'training/report.json'
    saved = json.loads(report_path.read_text())
    checkpoint = source/'training/checkpoints/round-0035'
    protected = {str(report_path): sha(report_path),
                 **{str(p): sha(p) for p in checkpoint.iterdir() if p.is_file()}}
    implementation = Path(__file__).resolve().parents[1]/'src/patchwork_automaton/neural'
    implementation_hashes = {str(p): sha(p) for p in implementation.rglob('*.py')}
    implementation_hashes[str(Path(__file__).resolve())] = sha(__file__)
    work, input_hashes = [], {}
    for round_id, samples in zip(ROUNDS, EXPECTED_SAMPLES):
        row, trajectories = select_training(saved, round_id)
        train = set(row['split']['training_indices'])
        held = set(row['split']['validation_indices'])
        assert len(train) == 800 and len(held) == 200
        held_paths = {t['path'] for game in row['games'] if game['index'] in held
                      for t in game.get('trajectories', ())}
        assert not held_paths.intersection(t['path'] for t in trajectories)
        assert sum(t['samples'] for t in trajectories) == samples
        assert all(t['reward'] in (-1., 0., 1.) for t in trajectories)
        for trajectory in trajectories:
            input_hashes[trajectory['path']] = sha(trajectory['path'])
        work.append((round_id, trajectories, samples, row['split']))
    keras.utils.set_random_seed(saved['config']['seed'])
    with device_scope('gpu'), checkpoint_view(checkpoint) as view:
        model, trainer, state = load_checkpoint(view, gru_backend='cudnn-memory')
        layers = gru_backend_summary(model)
        assert sum(layer['use_cudnn'] is True for layer in layers) == 4
        assert sum(layer['use_cudnn'] is False for layer in layers) == 2
        functions = {name: getattr(trainer, name) for name in (
            'step', 'critic_step', 'critic_sequence_step', 'measure', 'drift',
            'actor_state_drift', 'measure_with_drift', 'measure_actor_with_drift',
            'snapshot', 'restore') if getattr(trainer, name, None) is not None}
        kernels = {name: graph_ops(functions[name]) for name in (
            'step', 'critic_step', 'measure_actor_with_drift')}
        assert all(any('CudnnRNN' in key for key in ops) for ops in kernels.values())
        traces = place_trainer(trainer, 'gpu')
        placements = assert_model_device(model, 'gpu')
    assert state['round'] == 35 and state['config'] == saved['config']
    trainer.numerical_prepacking = trainer.cache_policy_batches = True
    trainer.policy_cache_device = trainer.actor_epoch_cache_device = '/GPU:0'
    trainer.policy_cache_device_bytes = args.policy_cache_bytes
    trainer.actor_epoch_cache_bytes = args.actor_cache_bytes
    assert trainer.separate_value and trainer.epochs == 2 and trainer.batch_size == 512
    variable_ids = {name: [id(v) for v in values]
                    for name, values in variable_groups(model, trainer).items()}
    placed_functions = {name: id(getattr(trainer, name)) for name in functions}
    trainer_identity, model_identity, rng_identity = id(trainer), id(model), id(trainer.rng)
    initial = state_summary(model, trainer)
    assert initial['pcg'] == state['trainer_rng_state']
    assert initial['actor_steps'] == state['optimizer_steps']
    assert initial['critic_steps'] == state['critic_optimizer_steps']
    info = dict(status='running', purpose='runtime/cache-lifecycle replay only',
        learning_strength_claim=False, fresh_on_policy_claim=False,
        behavior_policy_limitation='Saved rounds 37–40 were collected by historical policies that can differ from this uninterrupted diagnostic replay. This tests finite runtime lifecycle, shape variation and state continuity, not fresh on-policy learning or strength.',
        source=str(source), checkpoint_round=35, rounds=list(ROUNDS), expected_samples=list(EXPECTED_SAMPLES),
        source_hashes=protected, input_hashes=input_hashes, implementation_hashes=implementation_hashes,
        config=state['config'], runtime=settings, device='gpu', gru_backend='cudnn-memory',
        policy_cache_budget_bytes=args.policy_cache_bytes, actor_cache_budget_bytes=args.actor_cache_bytes,
        gru_layers=layers, kernel_ops=kernels, variable_placements=placements,
        initial_graph_traces=traces, initial_state=initial,
        initial_object_identity=dict(model=model_identity, trainer=trainer_identity, pcg=rng_identity),
        updates=[], no_new_games=True, source_checkpoints_read_only=True,
        no_state_resets=True, no_clear_session=True, no_forced_gc_between_updates=True,
        allocator_peak_reset_only=True, full_gpu_trace=False,
        acceptance_preregistered=dict(all_five_updates_complete=True, immutable_source_splits_config=True,
            persistent_model_both_adams_pcg=True, input_cache_caps_and_release=True,
            no_live_ppo_workers_readers_or_cached_tensor_objects_at_boundaries=True,
            final_three_post_update_tf_current_range_max_bytes=PLATEAU_BYTES,
            production_readiness_minimum_sampled_nvml_headroom_bytes=HEADROOM_BYTES,
            headroom_gate_distinct_from_lifecycle_completion=True))
    write(output/'report.json', info)
    assert not ppo_threads()
    probe = StorageProbe(trace_dir=None, trace_seconds=0)
    probe.boundaries['before'] = os_snapshot()
    metrics_file = (output/'nvidia-smi.csv').open('w')
    metrics_error = (output/'nvidia-smi.stderr.log').open('w')
    try:
        monitor = subprocess.Popen(['/usr/lib/wsl/lib/nvidia-smi',
            '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem',
            '--format=csv,noheader,nounits', '--loop-ms=500'], stdout=metrics_file, stderr=metrics_error)
    except FileNotFoundError:
        monitor = None
    monitor_start = time.time_ns()
    active_decisions = []
    original_guard = streaming_ppo._guard_status

    def guard(*args, **kwargs):
        result = original_guard(*args, **kwargs)
        active_decisions.append(dict(accepted=result[0], reasons=result[2],
            mean_kl=float(args[1][0]), max_state_kl=float(args[1][1])))
        return result

    streaming_ppo._guard_status = guard
    prior = initial
    failed = False
    try:
        with LifetimeObserver() as lifetimes:
            for round_id, trajectories, samples, split in work:
                lifetimes.round = round_id
                before = state_summary(model, trainer)
                assert before == prior, 'Model/Adam/PCG changed between updates'
                assert (id(model), id(trainer), id(trainer.rng)) == (model_identity, trainer_identity, rng_identity)
                assert variable_ids == {name: [id(v) for v in values]
                                        for name, values in variable_groups(model, trainer).items()}
                assert placed_functions == {name: id(getattr(trainer, name)) for name in functions}
                assert not ppo_threads()
                tf_before_reset = tf.config.experimental.get_memory_info('GPU:0')
                tf.config.experimental.reset_memory_stats('GPU:0')
                tf_before = tf.config.experimental.get_memory_info('GPU:0')
                assert tf_before['current'] == tf_before_reset['current']
                current = dict(round=round_id, samples=samples, training_games=800,
                    heldout_games_excluded=200, split=split, before_state=before,
                    tf_allocator_before_peak_reset=tf_before_reset, tf_allocator_before=tf_before,
                    before_os=os_snapshot(), guard_decisions=active_decisions)
                info['active_round'] = round_id
                info['active_update'] = current
                write(output/'report.json', info)
                started, cpu_started = time.perf_counter(), time.process_time()
                current['started_wall_ns'] = time.time_ns()
                with probe.span('round_%04d_update' % round_id):
                    result = neural_call('gpu', trainer.update_files, trajectories)
                current.update(update_seconds=time.perf_counter()-started,
                    process_cpu_seconds=time.process_time()-cpu_started,
                    ended_wall_ns=time.time_ns(), result=result)
                # Boundary checks precede state materialization or optional GC.
                current['tf_allocator_after'] = tf.config.experimental.get_memory_info('GPU:0')
                current['after_os'] = os_snapshot()
                current['lifetime_boundary'] = lifetimes.boundary(round_id)
                assert current['lifetime_boundary']['released'], 'Update-local resource lifetime gate failed'
                check_pipeline(result, trainer, samples, args.policy_cache_bytes, args.actor_cache_bytes, active_decisions)
                expected_rng = np.random.default_rng()
                expected_rng.bit_generator.state = copy.deepcopy(before['pcg'])
                for _ in range(trainer.epochs + len(result['pipeline']['actor_epoch_cache_epochs'])):
                    expected_rng.permutation(samples)
                after = state_summary(model, trainer)
                assert after['pcg'] == expected_rng.bit_generator.state, 'PCG consumption diverged'
                assert after['actor_steps'] - before['actor_steps'] == result['accepted_steps']
                assert after['critic_steps'] - before['critic_steps'] == result['critic_steps']
                current.update(after_state=after, pcg_consumption_verified=True,
                    persistent_state_verified=True,
                    graph_traces={name: fn.experimental_get_tracing_count() for name, fn in functions.items()})
                assert current['graph_traces'] == traces, 'Unexpected retracing with varied saved shapes'
                final_arrays = {f'{label}_{i}': np.asarray(variable)
                    for label, values in variable_groups(model, trainer).items()
                    for i, variable in enumerate(values)}
                assert all(np.isfinite(value).all() for value in final_arrays.values())
                path = output/f'round-{round_id:04d}-state.npz'
                np.savez(path, **final_arrays)
                current.update(state_arrays=len(final_arrays), state_archive=path.name,
                               state_archive_sha256=sha(path))
                del final_arrays
                # Each update owns a different scalar-only decision list.
                active_decisions = []
                prior = after
                info['updates'].append(current)
                del info['active_update']
                write(output/'report.json', info)
                print(json.dumps(dict(round=round_id, seconds=current['update_seconds'],
                    accepted_steps=result['accepted_steps'], rejected_epochs=result['rejected_epochs'],
                    critic_steps=result['critic_steps'], tf_allocator=current['tf_allocator_after'],
                    cache_lifetime_released=current['lifetime_boundary']['released'])), flush=True)
            post = [u['tf_allocator_after']['current'] for u in info['updates'][-3:]]
            info['post_update_tf_current_bytes'] = [u['tf_allocator_after']['current'] for u in info['updates']]
            info['last_three_tf_current_range_bytes'] = max(post)-min(post)
            info['lifecycle_acceptance'] = dict(all_five_updates_complete=len(info['updates']) == 5,
                state_continuity=all(u['persistent_state_verified'] and u['pcg_consumption_verified'] for u in info['updates']),
                resources_released=all(u['lifetime_boundary']['released'] for u in info['updates']),
                no_retracing=all(u['graph_traces'] == traces for u in info['updates']),
                tf_current_plateau=max(post)-min(post) <= PLATEAU_BYTES)
            if args.final_gc_diagnostic:
                import gc
                info['optional_final_gc'] = dict(before=tf.config.experimental.get_memory_info('GPU:0'))
                info['optional_final_gc']['unreachable_objects'] = gc.collect()
                info['optional_final_gc']['after'] = tf.config.experimental.get_memory_info('GPU:0')
                info['optional_final_gc']['boundaries'] = [lifetimes.boundary(r) for r in ROUNDS]
                info['optional_final_gc']['used_to_pass_normal_boundary_gates'] = False
    except BaseException as caught:
        info.update(status='failed', error=repr(caught), incomplete_round_guard_decisions=active_decisions)
        failed = True
        raise
    finally:
        streaming_ppo._guard_status = original_guard
        probe.boundaries['after'] = os_snapshot()
        if monitor:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill(); monitor.wait()
        metrics_file.close(); metrics_error.close()
        info['profiling'] = probe.finish(output)
        info['nvml_summary'] = nvidia_summary(output/'nvidia-smi.csv', monitor_start, time.time_ns())
        info['source_unchanged'] = all(sha(path) == digest for path, digest in {
            **protected, **input_hashes, **implementation_hashes}.items())
        info['final_ppo_threads'] = ppo_threads()
        info['finished_utc'] = datetime.now(timezone.utc).isoformat()
        if not failed:
            info['lifecycle_acceptance']['immutable_source_splits_config'] = info['source_unchanged']
            info['lifecycle_acceptance']['final_workers_gone'] = not info['final_ppo_threads']
            info['lifecycle_passed'] = all(info['lifecycle_acceptance'].values())
            info['production_readiness_headroom_passed'] = info['nvml_summary']['headroom_gate_passed']
            info['status'] = 'passed' if info['lifecycle_passed'] else 'failed'
        write(output/'report.json', info)
    if not info['lifecycle_passed']:
        raise RuntimeError('Finite runtime lifecycle acceptance failed; see saved report')
    print(json.dumps(dict(status=info['status'], lifecycle_acceptance=info['lifecycle_acceptance'],
        headroom_passed=info['production_readiness_headroom_passed'],
        post_update_tf_current_bytes=info['post_update_tf_current_bytes'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy-cache-bytes', type=int, default=CACHE_BYTES)
    parser.add_argument('--actor-cache-bytes', type=int, default=CACHE_BYTES)
    parser.add_argument('--final-gc-diagnostic', action='store_true')
    run(parser.parse_args())
