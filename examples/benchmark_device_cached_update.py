"""Replay one frozen PPO update with host or GPU policy-input residency.

Both variants use selective cuDNN GRUs, direct numerical prepacking, the
canonical policy-input cache and consolidated finite-gradient checks. The only
variant is bounded device residency of frozen policy-scan inputs. Run fresh
processes in the preregistered order after the separate correctness preflight.
No games, model changes, RL overrides, checkpoint writes or full GPU tracing.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from examples.profile_packed_update import StorageProbe
from examples.profile_ppo_update import checkpoint_view, select_training, sha, write


def run(args):
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
    trainer.policy_cache_device = '/GPU:0' if args.cache_placement == 'gpu' else None
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
        info.update(status='complete', cache_admission_verified=True)
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
        'final_model_sha256', 'source_unchanged', 'policy_input_cache')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-placement', choices=('host', 'gpu'), required=True)
    parser.add_argument('--cache-budget-bytes', '--device-cache-bytes',
                        dest='cache_budget_bytes', type=int, default=8 * 1024**3)
    parser.add_argument('--preflight-report', type=Path)
    run(parser.parse_args())
