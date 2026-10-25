"""One locked, CPU-only, collection-only hardware preflight for rollout pooling."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import traceback


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', type=Path, required=True)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def validate_rollouts(rows, memory_size, *, read_trajectory=None):
    """Read every frozen record; no forward pass or optimizer is constructed."""
    import numpy as np
    if read_trajectory is None:
        from patchwork_automaton.neural.experiment import read_trajectory
    from patchwork_automaton.neural.pooled_collector import file_sha256
    files, choices = [], 0
    for row in rows:
        if row['status'] not in ('win', 'loss', 'draw') or len(row['trajectories']) != 1:
            raise ValueError('Preflight requires every terminal game and its single learner trajectory')
        trajectory = row['trajectories'][0]
        reward = {'win':1., 'loss':-1., 'draw':0.}[row['status']]
        if trajectory.get('player') != row['seat'] or trajectory['reward'] != reward:
            raise ValueError('Preflight recorded an opponent or fabricated return')
        count = 0
        for sample in read_trajectory(trajectory['path']):
            mask = sample.arrays['candidate_mask']
            old = np.asarray(sample.log_probs, np.float64)
            arrays = [*sample.arrays.values(), sample.memory, old]
            if (not all(np.isfinite(value).all() for value in arrays)
                    or not math.isfinite(sample.value) or sample.memory.shape != (memory_size,)
                    or sample.advance is not True or mask.ndim != 1 or len(mask) < 2
                    or not np.all(mask) or old.shape != mask.shape
                    or not 0 <= sample.action < len(mask) or not mask[sample.action]
                    or not np.isclose(np.exp(old).sum(), 1., atol=1e-5, rtol=0.)):
                raise ValueError('Invalid frozen behavior probabilities, values, action or recurrent memory')
            count += 1
        if count != trajectory['samples'] or count != row['neural_choices']:
            raise ValueError('Incomplete or mismatched preflight trajectory')
        path = Path(trajectory['path'])
        files.append(dict(path=str(path), samples=count, bytes=path.stat().st_size,
                          sha256=file_sha256(path), collection_host=row['collection_host']))
        choices += count
    if len({item['path'] for item in files}) != len(files):
        raise ValueError('Repeated trajectory file in preflight round')
    return dict(files=files, samples=choices,
        file_bytes_by_host={host:sum(item['bytes'] for item in files if item['collection_host']==host)
                            for host in ('raytracer','laptop')})


async def run(args):
    # This environment is inherited by all spawned replicas and game engines.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '4'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    exchange_root = Path(config['exchange_root']).resolve()
    if (Path(config['rollout_output']).resolve() != output/'rollouts'
            or not exchange_root.is_relative_to(output)
            or 'preflight' not in config['run_id'].lower()):
        raise ValueError('Preflight must use its own output, exchange and run identifier')
    if (output/'report.json').exists() or (exchange_root/'current.json').exists():
        raise ValueError('Preflight output has already been used')
    seed = config['preflight_seed']
    if type(seed) is not int or seed < 10000:
        raise ValueError('Preflight needs a distinct positive seed range')
    lock_path = Path(config.get('job_lock', config.get('shared_lock', '')))
    if not lock_path.is_absolute():
        raise ValueError('Preflight requires the absolute shared job_lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    collector = None
    report = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  operation='collection-only; no optimizer, update, or policy evaluation')
    from examples.produce_rustmage_rollouts import atomic_json, verify_sources
    try:
        # Import neural code only after GPU hiding and exclusive job admission.
        from patchwork_automaton.neural.devices import configure_runtime, device_scope, assert_model_device
        configure_runtime('cpu', hide_gpu=True)
        import keras
        import tensorflow as tf
        from patchwork_automaton.neural import model as registered_models  # noqa: F401
        from patchwork_automaton.neural.inference_processes import weights_digest
        from patchwork_automaton.neural.pooled_collector import PooledGameCollector, file_sha256
        from patchwork_automaton.neural.rustmage_features import RustMageEncoder
        from patchwork_automaton.neural.validation import split_games, training_normalization
        from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector
        os.environ['PATCHWORK_OBSERVATION_PIPELINE'] = config.get('observation_pipeline', 'rust')
        os.environ['PATCHWORK_ROLLOUT_STORAGE'] = config.get('rollout_storage', 'pickle')
        verify_sources(dict(config, project_root=config.get('project_root', str(Path.cwd()))))
        checkpoint = args.checkpoint.resolve()
        input_files = sorted(p for p in checkpoint.parent.iterdir() if p.is_file())
        before_files = {str(path):file_sha256(path) for path in input_files}
        if file_sha256(checkpoint) != file_sha256(config['initial_policy']):
            raise ValueError('Preflight checkpoint and pooled initial model differ')
        saved = json.loads(checkpoint.with_suffix('.manifest.json').read_text())
        encoder = RustMageEncoder(allow_concede=False).manifest()
        if (saved.get('fingerprint') != encoder['fingerprint'] or saved.get('allow_concede') is not False
                or tuple(saved.get('actions', {})) != tuple(encoder['actions'])):
            raise ValueError('Preflight checkpoint encoder or ordered actions differ')
        with device_scope('cpu'):
            network = keras.saving.load_model(checkpoint, compile=False)
        if getattr(network, 'optimizer', None) is not None:
            raise ValueError('Preflight must never load an optimizer')
        original_digest = weights_digest(network.get_weights())
        report.update(input_sha256=before_files, config_sha256=file_sha256(args.config),
            source_sha256=config['source_sha256'], identity=config['identity'],
            checkpoint_weights_sha256=original_digest, variable_placements=assert_model_device(network,'cpu'),
            visible_gpus=[gpu.name for gpu in tf.config.get_visible_devices('GPU')],
            available_logical_cpus=sorted(os.sched_getaffinity(0)),
            packages={name:importlib.metadata.version(name) for name in ('keras','tensorflow','numpy','orjson')})
        if report['visible_gpus']:
            raise ValueError('Preflight unexpectedly exposed a GPU')
        atomic_json(output/'report.json', report)
        inner = ProcessGameCollector(network, workers=48, inference_processes=4,
                                     wait_ms=0, inference_device='cpu')
        collector = inner
        await inner.start(version=0)
        warm = await inner.run_games(games=48, seed_base=seed-10000, index_base=100000000,
                                    mode='versus_baseline', version=0, output=None, max_own_turns=60)
        if any(row['status'] not in ('win','loss','draw') or row['trajectories'] for row in warm):
            raise ValueError('Native CPU warm-up failed or unexpectedly recorded data')
        report['warmup'] = dict(games=48, output=None, outcomes=dict(Counter(r['status'] for r in warm)))
        collector = PooledGameCollector(inner, network, config=config)
        before = collector.stats()
        started, cpu_started = time.monotonic(), time.process_time()
        rows = await collector.run_games(games=1000, seed_base=seed, index_base=0,
            mode='versus_baseline', version=0, output=output/'rollouts', max_own_turns=60)
        seconds, parent_cpu = time.monotonic()-started, time.process_time()-cpu_started
        after = collector.stats()
        counts = dict(Counter(row['collection_host'] for row in rows))
        local_worker_cpu = sum(r.get('worker_cpu_seconds',0.) for r in rows if r['collection_host']=='raytracer')
        local_replica_cpu = after['replica_cpu_seconds']-before['replica_cpu_seconds']
        report.update(collection_seconds=seconds, host_games=counts, games=rows,
            outcomes=dict(Counter(row['status'] for row in rows)), pool=after['pool'],
            collection_resources=dict(parent_cpu_seconds=parent_cpu,
                raytracer_worker_cpu_seconds=local_worker_cpu, raytracer_replica_cpu_seconds=local_replica_cpu,
                raytracer_mean_logical_cpus=(parent_cpu+local_worker_cpu+local_replica_cpu)/seconds,
                laptop_worker_cpu_seconds=after['pool']['remote_worker_cpu_seconds'],
                laptop_inference=after['pool']['remote_inference']))
        if len(rows) != 1000 or counts.get('laptop', 0) < 32:
            raise ValueError('Preflight requires 1000 scheduled games and at least 32 accepted laptop games')
        train, heldout, partition = split_games(rows, fraction=.2, seed=seed, round_id=1)
        if len(train) != 800 or len(heldout) != 200:
            raise ValueError('Preflight whole-game split differs from 800/200')
        verified = validate_rollouts(rows, network.memory_size)
        normalization = training_normalization(train, 1., advantage_baseline='critic')
        if (normalization['source'] != 'training_partition_only'
                or normalization['training_choices'] != partition['training']['samples']
                or not all(math.isfinite(normalization[k]) for k in ('center','scale'))):
            raise ValueError('Preflight normalization leaked or lost training choices')
        if weights_digest(network.get_weights()) != original_digest or before_files != {
                str(path):file_sha256(path) for path in input_files}:
            raise ValueError('Collection preflight changed model or original checkpoint files')
        verify_sources(dict(config, project_root=config.get('project_root', str(Path.cwd()))))
        report.update(status='complete', collection_seconds=seconds, host_games=counts,
            outcomes=dict(Counter(row['status'] for row in rows)), split=partition,
            training_normalization=normalization, verified_rollouts=verified, games=rows,
            model_and_checkpoint_unchanged=True)
        return report
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        try:
            atomic_json(exchange_root/'producer-stop.json', dict(status=report['status'],
                stopped_utc=report['finished_utc'], operation='collection-only-preflight'))
            if collector:
                try:
                    await collector.close()
                except BaseException:
                    report.update(status='failed', cleanup_error=traceback.format_exc())
                    raise
        finally:
            atomic_json(output/'report.json', report)
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


if __name__ == '__main__':
    asyncio.run(run(parser().parse_args()))
