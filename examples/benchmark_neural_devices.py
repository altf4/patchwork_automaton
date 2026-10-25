"""Matched real-observation CPU/GPU inference and production PPO step timings.

Prepare one trusted local fixture, then copy it and the unchanged checkpoint to
each host. Model architecture, precision and decision samples stay identical.
The measured path includes NumPy handoff and synchronized NumPy results, excludes
imports/tracing/warmup, and performs one actor plus one critic update per step.
Only locally generated trusted pickle fixtures and trajectories may be loaded.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import random
import statistics
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def configure(args):
    """Constrain every subsequently created thread before importing NumPy/TF."""
    available = sorted(os.sched_getaffinity(0))
    cpu = available[0] if args.cpu is None else args.cpu
    if cpu not in available:
        raise ValueError(f'CPU {cpu} is unavailable; choose from {available}')
    if not args.all_cpus:
        os.sched_setaffinity(0, {cpu})
    os.environ.update(KERAS_BACKEND='tensorflow', CUDA_VISIBLE_DEVICES='0' if args.device == 'gpu' else '-1',
        TF_NUM_INTRAOP_THREADS=str(args.intra), TF_NUM_INTEROP_THREADS=str(args.inter), OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        TF_ENABLE_ONEDNN_OPTS='0', TF_CPP_MIN_LOG_LEVEL='2',
        PATCHWORK_OBSERVATION_PIPELINE='python')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.intra)
    tf.config.threading.set_inter_op_parallelism_threads(args.inter)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    gpus = tf.config.list_physical_devices('GPU')
    if args.device == 'gpu' and not gpus:
        raise RuntimeError('GPU requested but TensorFlow sees no GPU; CPU fallback is forbidden')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return tf, '/GPU:0' if args.device == 'gpu' else '/CPU:0', available


def pack_records(records, size):
    """Production padding and detached recurrent state; repeat real records only."""
    import numpy as np
    from patchwork_automaton.neural.features import collate
    if not records or size < 1:
        raise ValueError('Need nonempty records and a positive batch size')
    selected = [records[index % len(records)] for index in range(size)]
    samples, rewards = zip(*selected)
    inputs = collate([sample.arrays for sample in samples])
    inputs['memory_in'] = np.stack([sample.memory for sample in samples])
    inputs['advance_memory'] = np.asarray([[float(sample.advance)] for sample in samples], np.float32)
    actions = np.asarray([sample.action for sample in samples], np.int32)
    old = np.asarray([sample.log_probs[sample.action] for sample in samples], np.float32)
    returns = np.asarray(rewards, np.float32)
    # The fixture's terminal returns and behavior values are frozen. Normalize
    # once over the full fixture, exactly as a production rollout does.
    from patchwork_automaton.neural.ppo import normalized_advantages
    all_rewards = [reward for _, reward in records]
    all_values = [sample.value for sample, _ in records]
    choices = np.asarray([sample.arrays['candidate_mask'].sum() > 1 for sample, _ in records])
    advantages = normalized_advantages(all_rewards, all_values, choices, 1.)
    advantages = np.asarray([advantages[index % len(records)] for index in range(size)], np.float32)
    return inputs, actions, returns, old, advantages


def select_records(descriptors, count, seed):
    """Uniform bounded-memory reservoir across complete saved games."""
    from patchwork_automaton.neural.experiment import read_trajectory
    rng = random.Random(seed)
    selected, seen = [], 0
    for trajectory in descriptors:
        trajectory_seen = 0
        if trajectory['reward'] not in (-1., 0., 1.):
            raise ValueError('Expected a verified terminal reward')
        for sample in read_trajectory(trajectory['path']):
            seen += 1
            trajectory_seen += 1
            record = sample, float(trajectory['reward'])
            if len(selected) < count:
                selected.append(record)
            else:
                index = rng.randrange(seen)
                if index < count:
                    selected[index] = record
        if trajectory_seen != trajectory['samples']:
            raise ValueError('Trajectory sample count mismatch: ' + trajectory['path'])
    if len(selected) < count:
        raise ValueError(f'Only {len(selected)} samples available; requested {count}')
    rng.shuffle(selected)
    return selected, seen


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), *sorted((root/'src/patchwork_automaton/neural').glob('*.py'))]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def metadata(args, tf, available):
    import keras
    import numpy as np
    info = Path('/proc/cpuinfo').read_text()
    gpu_info = []
    for device in tf.config.list_physical_devices('GPU'):
        gpu_info.append(dict(name=device.name, details=tf.config.experimental.get_device_details(device)))
    return dict(started_utc=datetime.now(timezone.utc).isoformat(), hostname=platform.node(),
        platform=platform.platform(), python=sys.version, tensorflow=tf.__version__, keras=keras.__version__,
        numpy=np.__version__, tensorflow_build=tf.sysconfig.get_build_info(), gpu_devices=gpu_info,
        cpu_model=next((line.split(':', 1)[1].strip() for line in info.splitlines()
                        if line.startswith('model name')), 'unknown'),
        initial_available_cpus=available, cpu_affinity=sorted(os.sched_getaffinity(0)),
        runtime={key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'TF_NUM_INTRAOP_THREADS',
            'TF_NUM_INTEROP_THREADS', 'TF_ENABLE_ONEDNN_OPTS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        device=args.device, precision='float32; TensorFloat32 disabled; unchanged production model',
        soft_device_placement=False, source_sha256=source_hashes())


def load_model(checkpoint, device):
    import tensorflow as tf
    from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
    with tf.device(device):
        model = RustMagePolicy.load(checkpoint).model
    if not getattr(model, 'separate_value', False):
        raise ValueError('Benchmark expects the production separate actor/critic model')
    target = 'GPU' if 'GPU' in device else 'CPU'
    placements = {variable.path: variable.value.device for variable in model.variables}
    if not all(target in value for value in placements.values()):
        raise RuntimeError(f'Model variables were not all placed on {target}: {placements}')
    return model, placements


def forward_function(model, device, tf):
    from patchwork_automaton.neural.batching import example_batch, input_signature
    with tf.device(device):
        forward = tf.function(model.call,
            input_signature=[input_signature(example_batch(model.memory_size, model.feature_spec))],
            autograph=False)
    return forward


def require_tensor_placement(outputs, device):
    expected = device.lstrip('/').upper()
    placements = {key: value.device for key, value in outputs.items()}
    if any(not placement.upper().endswith(expected) for placement in placements.values()):
        raise RuntimeError(f'Expected {device} output tensors, got {placements}')
    return placements


def synchronized_forward(forward, inputs, device, tf):
    import numpy as np
    with tf.device(device):
        outputs = forward(inputs)
    require_tensor_placement(outputs, device)
    # Reading every output also synchronizes asynchronous GPU work and includes
    # the copies needed by the actual NumPy policy consumer.
    return {key: np.asarray(value) for key, value in outputs.items()}


def prepare(args, tf, device, report):
    model, placements = load_model(args.checkpoint, device)
    original = json.loads(args.rollout_report.read_text())
    round_ = next(row for row in original['rounds'] if row['round'] == args.round)
    games = [row for row in round_['games'] if row['status'] in ('win', 'loss', 'draw') and row['trajectories']]
    if len(games) < args.fixture_games:
        raise ValueError('Not enough naturally completed fixture games')
    descriptors = [dict(trajectory) for game in games[:args.fixture_games] for trajectory in game['trajectories']]
    for trajectory in descriptors:
        path = Path(trajectory['path'])
        if not path.exists():
            # Reports normally contain paths relative to the package root.
            path = Path(__file__).resolve().parents[1]/path
        trajectory['path'] = str(path.resolve())
    records, scanned = select_records(descriptors, args.fixture_samples, args.seed)
    forward = forward_function(model, device, tf)
    references = {}
    for size in sorted(set(args.forward_batches + args.backward_batches)):
        inputs, *_ = pack_records(records, size)
        references[size] = synchronized_forward(forward, inputs, device, tf)
    fixture = dict(version=1, records=records, references=references,
        checkpoint_sha256=digest(args.checkpoint), source_sha256=source_hashes(),
        trajectory_sha256={str(Path(t['path']).name): digest(t['path']) for t in descriptors},
        fixture_games=args.fixture_games, fixture_samples=args.fixture_samples,
        samples_scanned=scanned, seed=args.seed, rollout_report_sha256=digest(args.rollout_report), round=args.round)
    fixture_path = args.output/'fixture.pkl'
    with fixture_path.open('wb') as stream:
        pickle.dump(fixture, stream, protocol=5)
    report.update(status='complete', fixture=str(fixture_path), fixture_sha256=digest(fixture_path),
        fixture_metadata={key: value for key, value in fixture.items() if key not in ('records', 'references')},
        parameters=model.count_params(), parameter_counts=dict(actor=model.actor.count_params(), critic=model.critic.count_params()),
        variable_placements=placements)


def restore_variables(variables, saved):
    for variable, value in zip(variables, saved):
        variable.assign(value)


def timed_trials(operation, restore, *, seconds, repeats, max_steps, warmups, fixed_steps=None):
    for _ in range(warmups):
        operation()
    restore()
    started = time.perf_counter()
    operation()
    one = time.perf_counter() - started
    steps = fixed_steps or max(1, min(max_steps, math.ceil(seconds/max(one, 1e-9))))
    trials = []
    for index in range(repeats):
        restore()
        started = time.perf_counter()
        cpu = time.process_time()
        for _ in range(steps):
            operation()
        elapsed = time.perf_counter() - started
        cpu = time.process_time() - cpu
        trials.append(dict(trial=index+1, steps=steps, seconds=elapsed, cpu_seconds=cpu,
            milliseconds_per_step=1000*elapsed/steps, mean_cpu_cores=cpu/elapsed))
    return dict(steps_per_trial=steps, trials=trials,
        median_milliseconds_per_step=statistics.median(row['milliseconds_per_step'] for row in trials))


def measure(args, tf, device, report):
    import numpy as np
    from patchwork_automaton.neural.ppo import PPOTrainer
    with args.fixture.open('rb') as stream:
        fixture = pickle.load(stream)
    if fixture.get('version') != 1 or digest(args.checkpoint) != fixture['checkpoint_sha256']:
        raise ValueError('Fixture format or checkpoint hash mismatch')
    current_sources = source_hashes()
    mismatches = [path for path, sha in fixture['source_sha256'].items() if current_sources.get(path) != sha]
    if mismatches:
        raise ValueError('Fixture sources differ: ' + ', '.join(mismatches))
    model, placements = load_model(args.checkpoint, device)
    state = json.loads(args.checkpoint.with_name('state.json').read_text())
    config = state['config']
    with tf.device(device):
        trainer = PPOTrainer(model, learning_rate=config['learning_rate'],
            value_learning_rate=config.get('value_learning_rate'),
            clip_ratio=config.get('ppo_clip', .2), advantage_scale_floor=config.get('advantage_scale_floor', 1.))
        for optimizer, name in [(trainer.optimizer, 'optimizer.npz'), (trainer.critic_optimizer, 'critic-optimizer.npz')]:
            with np.load(args.checkpoint.with_name(name), allow_pickle=False) as archive:
                if len(archive.files) != len(optimizer.variables):
                    raise ValueError('Optimizer variable count mismatch')
                for index, variable in enumerate(optimizer.variables):
                    value = archive[f'v{index}']
                    if tuple(value.shape) != tuple(variable.shape):
                        raise ValueError('Optimizer variable shape mismatch')
                    variable.assign(value)
    variables = list(model.variables) + list(trainer.optimizer.variables) + list(trainer.critic_optimizer.variables)
    saved = [np.asarray(variable).copy() for variable in variables]
    initial_weights = [np.asarray(variable).copy() for variable in model.trainable_variables]

    def restore():
        with tf.device(device):
            restore_variables(variables, saved)
        # Assignments precede the next graph on this thread; this explicit read
        # also keeps any final device transfer outside measured trial time.
        np.asarray(variables[-1])

    forward = forward_function(model, device, tf)
    fixed_counts = {}
    if args.counts_from:
        counts = json.loads(args.counts_from.read_text())
        if counts['fixture_sha256'] != digest(args.fixture) or counts['checkpoint_sha256'] != digest(args.checkpoint):
            raise ValueError('Step-count reference uses a different checkpoint or fixture')
        fixed_counts = {(row['kind'], row['batch_size']): row['steps_per_trial'] for row in counts['measurements']}
        required = [('forward', size) for size in args.forward_batches]
        required += [('actor_and_critic_update', size) for size in args.backward_batches]
        if any(key not in fixed_counts for key in required):
            raise ValueError('Step-count reference is missing a requested case')
    report.update(status='running', fixture_sha256=digest(args.fixture), checkpoint_sha256=digest(args.checkpoint),
        checkpoint_artifact_sha256={name: digest(args.checkpoint.with_name(name)) for name in
            ('policy.keras', 'policy.manifest.json', 'state.json', 'optimizer.npz', 'critic-optimizer.npz')},
        parameters=model.count_params(), parameter_counts=dict(actor=model.actor.count_params(), critic=model.critic.count_params()),
        variable_placements=placements, measurements=[],
        methodology=f'CPU affinity {sorted(os.sched_getaffinity(0))}; TF intra/inter threads {args.intra}/{args.inter}; FP32 with TF32 disabled. '
            'Identical real recorded observations and saved model/Adam state. Imports, graph traces, '
            'warmups and resets excluded. NumPy handoff and synchronized outputs included. '
            'Fixed-count paired actor/critic optimizer calls; no PPO acceptance guard or BPTT. '
            'Padding/fixture reads excluded; tensors remain NumPy inputs on every call.')
    atomic_json(args.output/'report.json', report)
    for kind, sizes in [('forward', args.forward_batches), ('actor_and_critic_update', args.backward_batches)]:
        for size in sizes:
            restore()
            packed = pack_records(fixture['records'], size)
            inputs = packed[0]
            expected = fixture['references'].get(size)
            if expected is None:
                raise ValueError(f'Fixture has no reference for batch {size}; prepare it with this batch size')
            actual = synchronized_forward(forward, inputs, device, tf)
            errors = {}
            for key, value in actual.items():
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite forward output')
                np.testing.assert_allclose(value, expected[key], rtol=2e-3, atol=3e-4,
                    err_msg=f'CPU fixture reference mismatch: {key}, batch {size}')
                errors[key] = float(np.max(np.abs(value-expected[key])))
            last_norms = []
            last_norm_placements = {}

            def operation():
                if kind == 'forward':
                    synchronized_forward(forward, inputs, device, tf)
                else:
                    with tf.device(device):
                        actor = trainer.step(*packed)
                        critic = trainer.critic_step(inputs, packed[2])
                    last_norm_placements.update(require_tensor_placement(dict(actor=actor, critic=critic), device))
                    # Each conversion is deliberate synchronization and mirrors
                    # the actual production trainer's consumed norm results.
                    last_norms[:] = [float(actor), float(critic)]
                    if not np.isfinite(last_norms).all():
                        raise ValueError('Nonfinite production gradient norm')

            result = timed_trials(operation, restore, seconds=args.seconds, repeats=args.repeats,
                max_steps=args.max_steps, warmups=args.warmups,
                fixed_steps=fixed_counts.get((kind, size)))
            result.update(kind=kind, batch_size=size, samples_per_second=1000*size/result['median_milliseconds_per_step'],
                counts_reference=str(args.counts_from) if args.counts_from else None,
                input_shapes={key: list(value.shape) for key, value in inputs.items()},
                reference_max_absolute_error=errors)
            if kind != 'forward':
                deltas = [np.asarray(variable)-before for variable, before in zip(model.trainable_variables, initial_weights)]
                if not all(np.isfinite(delta).all() for delta in deltas):
                    raise ValueError('Nonfinite updated weights')
                actor_delta = sum(float(np.square(delta).sum()) for delta in deltas[:len(model.actor.trainable_variables)])
                critic_delta = sum(float(np.square(delta).sum()) for delta in deltas[len(model.actor.trainable_variables):])
                if actor_delta <= 0 or critic_delta <= 0 or min(last_norms) <= 0:
                    raise ValueError('Actor and critic must both have nonzero gradients and changed weights')
                expected_steps = result['steps_per_trial']
                original_actor = int(saved[len(model.variables)])
                original_critic = int(saved[len(model.variables)+len(trainer.optimizer.variables)])
                if (int(trainer.optimizer.iterations) != original_actor+expected_steps
                        or int(trainer.critic_optimizer.iterations) != original_critic+expected_steps):
                    raise AssertionError('Optimizer step count mismatch')
                result.update(last_gradient_norms=last_norms, actor_weight_delta_l2=math.sqrt(actor_delta),
                    gradient_norm_output_placements=dict(last_norm_placements),
                    critic_weight_delta_l2=math.sqrt(critic_delta),
                    actor_optimizer_iterations=int(trainer.optimizer.iterations),
                    critic_optimizer_iterations=int(trainer.critic_optimizer.iterations))
            report['measurements'].append(result)
            atomic_json(args.output/'report.json', report)
            print(json.dumps(dict(stage='measurement', kind=kind, batch_size=size,
                median_milliseconds_per_step=result['median_milliseconds_per_step'],
                samples_per_second=result['samples_per_second'])), flush=True)
    restore()
    report['forward_graph_traces'] = forward.experimental_get_tracing_count()
    report['actor_graph_traces'] = trainer.step.experimental_get_tracing_count()
    report['critic_graph_traces'] = trainer.critic_step.experimental_get_tracing_count()
    graph = forward.get_concrete_function().graph.as_graph_def()
    matrix_ops = [dict(name=node.name, operation=node.op, device=node.device,
                      graph_device_status='specified' if node.device else 'unknown')
                  for node in graph.node if 'MatMul' in node.op]
    report['forward_matrix_operation_placements'] = matrix_ops
    report['graph_device_audit_scope'] = ('Empty GraphDef device fields are unknown, not proof of GPU execution. '
        'Runtime checks assert model-variable, forward-output and optimizer-norm tensor devices; '
        'calls use an explicit requested device with soft placement disabled. GPU kernel profiling is not captured.')
    if args.device == 'gpu' and any('CPU' in node['device'] for node in matrix_ops):
        raise RuntimeError('A main forward matrix operation was placed on CPU')
    report['status'] = 'complete'


async def neural_games(args, tf, device, report):
    import numpy as np
    from patchwork_automaton.neural.batching import InferenceBatcher, example_batch, input_signature, pack_requests
    from patchwork_automaton.rl.rustmage_experiment import run_games
    model, placements = load_model(args.checkpoint, device)
    original = [np.asarray(variable).copy() for variable in model.variables]

    class PlacedBatcher(InferenceBatcher):
        async def start(self):
            # The production batcher's executor thread has its own TF device
            # context. Place the initial trace explicitly as well as hot calls.
            self.loop = asyncio.get_running_loop()

            def warm():
                batch = example_batch(model.memory_size, model.feature_spec)
                with tf.device(device):
                    model(batch, training=False)
                    self.compiled = tf.function(model.call, input_signature=[input_signature(batch)], autograph=False)
                    outputs = self.compiled(batch)
                self.output_placements = require_tensor_placement(outputs, device)
                for value in outputs.values():
                    np.asarray(value)

            await self.loop.run_in_executor(self.executor, warm)
            self.task = asyncio.create_task(self._serve())
            return self

        def _infer(self, requests):
            # Same packing/cropping as the production batcher, with a runtime
            # placement assertion before tensors are converted to NumPy.
            items, batch = pack_requests(requests)
            with tf.device(device):
                outputs = self.compiled(batch)
            self.output_placements = require_tensor_placement(outputs, device)
            output = {key: np.asarray(value) for key, value in outputs.items()}
            return [{key: (value[index:index+1, :len(items[index]['candidates'])] if key == 'logits'
                           else value[index:index+1]) for key, value in output.items()}
                    for index in range(len(requests))]

    batcher = await PlacedBatcher(model, max_batch=1, wait_ms=0).start()
    try:
        warm = await run_games(games=args.warmup_games, workers=1, seed_base=args.seed+100_000,
            model=model, batcher=batcher, mode='selfplay', max_own_turns=60)
        if any(row['status'] not in ('win', 'loss', 'draw') for row in warm):
            raise ValueError('Warm-up games failed or reached a cutoff')
        before = batcher.stats()
        started = time.perf_counter()
        cpu = time.process_time()
        rows = await run_games(games=args.games, workers=1, seed_base=args.seed,
            model=model, batcher=batcher, mode='selfplay', max_own_turns=60, output=args.output/'rollouts',
            progress=lambda row: print(json.dumps(dict(stage='game', seed=row['seed'], status=row['status'],
                seconds=row['seconds'], decisions=row['decisions'], neural_choices=row['neural_choices'])), flush=True))
        seconds, cpu = time.perf_counter()-started, time.process_time()-cpu
        after = batcher.stats()
        report.update(seconds=seconds, cpu_seconds=cpu, mean_cpu_cores=cpu/seconds,
            games_per_minute=60*len(rows)/seconds, warmup_games=warm, games=rows,
            choices_per_second=sum(row['neural_choices'] for row in rows)/seconds,
            parameters=model.count_params(), variable_placements=placements,
            inference_output_placements=batcher.output_placements,
            checkpoint_sha256=digest(args.checkpoint),
            inference={key: after[key]-before[key] for key in ('calls', 'requests', 'inference_seconds')},
            methodology='One sequential game engine, inference batch 1 with zero queue delay, '
                'one CPU core and TF threads 1/1. Warmup excluded; trajectory writing included.')
        for variable, value in zip(model.variables, original):
            np.testing.assert_array_equal(np.asarray(variable), value)
        if any(row['status'] not in ('win', 'loss', 'draw') for row in rows):
            raise ValueError('Measured games failed or reached a cutoff')
        report['status'] = 'complete'
    finally:
        await batcher.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'measure', 'games'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    p.add_argument('--cpu', type=int, help='One allowed logical CPU; default first available')
    p.add_argument('--all-cpus', action='store_true', help='Allow the complete host CPU affinity for production GPU feeding')
    p.add_argument('--intra', type=int, default=1)
    p.add_argument('--inter', type=int, default=1)
    p.add_argument('--fixture', type=Path)
    p.add_argument('--counts-from', type=Path,
        help='Use exact per-case step counts from a prior host report for matching weight evolution')
    p.add_argument('--rollout-report', type=Path,
        default=Path('.local/red-value500-1000-2026-10-18/report.json'))
    p.add_argument('--round', type=int, default=10)
    p.add_argument('--fixture-samples', type=int, default=128)
    p.add_argument('--fixture-games', type=int, default=8)
    p.add_argument('--forward-batches', type=int, nargs='+', default=[1, 32, 128])
    p.add_argument('--backward-batches', type=int, nargs='+', default=[32, 128])
    p.add_argument('--seconds', type=float, default=1., help='Approximate target per repeat and case')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmups', type=int, default=3)
    p.add_argument('--max-steps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=71_000_000)
    p.add_argument('--games', type=int, default=8)
    p.add_argument('--warmup-games', type=int, default=2)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.mode == 'measure' and args.fixture is None:
        p.error('measure requires --fixture')
    if args.mode == 'prepare' and args.device != 'cpu':
        p.error('prepare must create its reference on CPU')
    if min(args.round, args.fixture_samples, args.fixture_games, args.seconds, args.repeats,
           args.max_steps, args.games, args.intra, args.inter, *args.forward_batches, *args.backward_batches) <= 0 or min(args.warmups, args.warmup_games) < 0:
        p.error('Counts/seconds must be positive; warmups must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='initializing', mode=args.mode, config={key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()})
    checkpoint_before = digest(args.checkpoint)
    try:
        tf, device, available = configure(args)
        report.update(metadata(args, tf, available))
        if args.mode == 'prepare':
            prepare(args, tf, device, report)
        elif args.mode == 'measure':
            measure(args, tf, device, report)
        else:
            asyncio.run(neural_games(args, tf, device, report))
        if digest(args.checkpoint) != checkpoint_before:
            raise AssertionError('Input checkpoint changed')
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    except BaseException as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        atomic_json(args.output/'report.json', report)
    print(json.dumps(dict(stage='complete', report=str(args.output/'report.json'))), flush=True)


if __name__ == '__main__':
    main()
