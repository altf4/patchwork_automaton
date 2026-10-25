"""Mono-red PPO with explicit devices, whole-game holdouts and saved checkpoints.

uv run --no-sync python -m examples.train_rustmage --output .local/red-ppo
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

# Set before importing Keras/TensorFlow, including in spawned inference replicas.
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '4')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import asyncio
import keras
import numpy as np

from patchwork_automaton.neural.batching import InferenceBatcher, example_batch
from patchwork_automaton.neural.experiment import atomic_json, save_checkpoint, load_checkpoint, make_trainer
from patchwork_automaton.neural.model import PolicyNetwork, SeparateActorCritic
from patchwork_automaton.neural.rustmage_features import RustMageEncoder
from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
from patchwork_automaton.neural.validation import GameValidator, split_games, training_normalization
from patchwork_automaton.neural.devices import configure_runtime, device_scope, assert_model_device, assert_output_device
from patchwork_automaton.neural.rustmage_evaluation import baseline_evaluation_config, match_summary, paired_match_improvement
from patchwork_automaton.rl.rustmage_experiment import run_games, goldfish_summary, paired_improvement


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--games-per-round', type=int, default=32)
    p.add_argument('--train-mode', choices=('selfplay','versus_baseline'), default='selfplay',
        help='PPO collection opponent: frozen current policy or active FoundationsAggro (critic warm-up remains self-play)')
    p.add_argument('--validation-fraction', type=float, default=.2,
        help='Whole games excluded from both optimizers (default: 20%%; count rounded up). Use 0 only for legacy runs.')
    p.add_argument('--eval-games', type=int, default=40)
    p.add_argument('--eval-every-games', type=int, default=0,
        help='Evaluate at this self-play interval; must be a multiple of games-per-round (0: endpoints only)')
    p.add_argument('--eval-horizon', type=int, default=20)
    p.add_argument('--train-horizon', type=int, default=60)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--inference-processes',type=int,
        help='Frozen inference replicas sharing the total engine count (new process runs: 2; old resumes: 1)')
    p.add_argument('--inference-wait-ms', type=float, default=0,
        help='Maximum batching delay; 0 immediately serves currently queued requests')
    p.add_argument('--collector', choices=('threads','processes'), default='processes')
    p.add_argument('--pooled-collector-config', type=str,
        help='Immutable configuration for versioned laptop rollout contributions (process collector only)')
    p.add_argument('--observation-pipeline', choices=('python','rust'), default='python',
        help='Observation encoding and history implementation (default: Python; experimental Rust requires scripts/build-observation-pipeline.py)')
    p.add_argument('--architecture', choices=('shared','separate'), default='separate')
    p.add_argument('--learner-device', choices=('cpu','gpu'), default='cpu')
    p.add_argument('--inference-device', choices=('cpu','gpu'), default='cpu')
    p.add_argument('--validation-device', choices=('cpu','gpu'), default='cpu')
    p.add_argument('--validation-batch', type=int, default=32,
        help='Concurrent recorded trajectories during recurrent validation replay')
    p.add_argument('--baseline-eval-games', type=int, default=0,
        help='Additional held-out games against the active FoundationsAggro opponent at each evaluation')
    p.add_argument('--baseline-eval-seed', type=int,
        help='Fixed active-baseline cohort seed (default: training seed + 20000000)')
    p.add_argument('--train-batch', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=.0003)
    p.add_argument('--ppo-epochs', type=int, default=2)
    p.add_argument('--entropy-weight', type=float, default=.01,
        help='Actor entropy bonus coefficient (default: 0.01; 0 disables the bonus)')
    p.add_argument('--advantage-baseline', choices=('critic','zero'), default='critic',
        help='Actor advantages subtract the frozen behavior critic or zero; value training remains unchanged')
    p.add_argument('--target-kl', type=float, default=.02,
        help='Maximum mean behavior-to-current policy KL accepted by the PPO guard')
    p.add_argument('--max-state-kl', type=float, default=.1,
        help='Maximum individual choice-state KL accepted by the PPO guard')
    p.add_argument('--guard-audit', action='store_true',
        help='Record and print per-proposal KL guard diagnostics without changing updates')
    p.add_argument('--value-learning-rate', type=float,
        help='Separate value optimizer learning rate (default: learning-rate; separate architecture only)')
    p.add_argument('--value-max-steps', type=int,
        help='Cap value optimizer steps per rollout update across all PPO epochs (default: uncapped; separate architecture only)')
    p.add_argument('--critic-update-scope', choices=('all','head'), default='all',
        help='Update all critic variables or only its value head, preserving optimizer checkpoint shapes')
    p.add_argument('--critic-unroll', type=int, default=1,
        help='Critic recurrent training chunk length (default: 1 detached behavior state)')
    p.add_argument('--critic-warmup-games', type=int, default=0,
        help='Separate frozen-policy self-play games for critic-only initialization before PPO (default: 0)')
    p.add_argument('--critic-warmup-max-steps', type=int,
        help='Critic optimizer step cap during warm-up only (default: value-max-steps)')
    p.add_argument('--concurrent-updates', action='store_true',
        help='Overlap independent actor/critic updates (requires separate architecture)')
    p.add_argument('--prefetch-batches',type=int,default=0,
        help='Prepared minibatches queued ahead per optimizer stream (0: synchronous)')
    p.add_argument('--measurement-workers',type=int,default=1,
        help='Concurrent loss/KL batches while the measured network is frozen')
    p.add_argument('--rollout-cache-mib', type=int, default=256,
        help='Decoded-record and prepared-minibatch cache budget in MiB (default: 256)')
    p.add_argument('--rollout-storage', choices=('pickle','indexed'), default='pickle',
        help='Trajectory format: compressed pickle or indexed numeric buffers (default: pickle)')
    p.add_argument('--seed', type=int, default=721)
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument('--resume', action='store_true')
    initialization.add_argument('--initialize-from', type=Path,
        help='Warm-start a new experiment from policy.keras or its folder; fresh optimizers and RNG')
    p.add_argument('--plot', action='store_true', help='Save a learning-curve PNG/SVG (requires the plots extra)')
    return p


def inference_process_count(args, saved=None):
    count = args.inference_processes
    if count is None:
        count = (saved.get('inference_processes',1) if saved is not None
                 else min(2,args.workers) if args.collector == 'processes' else 1)
    if not 1 <= count <= args.workers: raise ValueError('Inference processes must be between one and workers')
    if args.collector != 'processes' and count != 1:
        raise ValueError('Multiple inference processes require the process collector')
    return count


def neural_call(device, function, *arguments, **keywords):
    """TensorFlow device contexts must be re-entered by every executor thread."""
    with device_scope(device):
        return function(*arguments, **keywords)


def initialization_provenance(source, state):
    """Recorded collection counts, including supervised checkpoints, without guessing."""
    info = dict(source_round=state['round'], source_algorithm=state.get('config', {}).get('algorithm', 'actor-critic'))
    result = state.get('round_result', {})
    if 'games_seen' in result:
        info['source_collected_games'] = result['games_seen']
    elif 'games_per_round' in state.get('config', {}):
        info['source_collected_games'] = state['round'] * state['config']['games_per_round']
    report_path = source.parent.parent.parent/'report.json'
    if report_path.is_file():
        previous = json.loads(report_path.read_text())
        completed = [r for r in previous.get('rounds', []) if r['round'] <= state['round']]
        if len(completed) == state['round'] and all('split' in r for r in completed):
            info['source_optimization_games'] = sum(r['split']['training']['attempted_games'] for r in completed)
            info['source_validation_games'] = sum(r['split']['validation']['attempted_games'] for r in completed)
            info.setdefault('source_collected_games', info['source_optimization_games'] + info['source_validation_games'])
        info.update(source_report=str(report_path.resolve()),
                    source_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())
    return info


def resume_checkpoint_root(output, warmup_games):
    """An incomplete warm-up always restarts from its immutable prewarm snapshot."""
    normal = output/'checkpoints'
    if (normal/'latest.json').is_file():
        return normal
    # A crash can happen between publishing round zero and publishing latest.
    # Its files were fsynced before the directory rename in save_checkpoint.
    zero = normal/'round-0000'/'state.json'
    if warmup_games and zero.is_file():
        state = json.loads(zero.read_text())
        if state.get('critic_warmup', {}).get('status') != 'complete':
            raise ValueError('Unpublished round zero has no completed critic warm-up')
        atomic_json(normal/'latest.json', {'checkpoint': 'round-0000'})
        return normal
    if warmup_games:
        prewarm = output/'prewarm-checkpoints'
        if not (prewarm/'latest.json').is_file() and (prewarm/'round-0000'/'state.json').is_file():
            atomic_json(prewarm/'latest.json', {'checkpoint': 'round-0000'})
        return prewarm
    return normal


def array_digest(arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


async def critic_warmup(args, model, trainer, validator, validation_model, report,
                        collect, sync_weights, collection_version, progress):
    """Commit critic initialization separately; actor state is an invariant."""
    report_path = args.output/'report.json'
    attempt = report.get('critic_warmup', {}).get('attempt', 0) + 1
    rollout_root = args.output/'warmup-rollouts'/f'attempt-{attempt:04d}'
    result = dict(status='running', attempt=attempt, requested_games=args.critic_warmup_games,
        seed_base=args.seed+30_000_000, rollout_root=str(rollout_root),
        validation_fraction=.2, counts_toward_ppo_games=False,
        methodology='Frozen current actor; terminal self-play returns; whole-game 20% holdout; critic-only updates; validation replays current memory from zero')
    report['critic_warmup'] = result
    atomic_json(report_path, report)
    actor_before = [np.asarray(v).copy() for v in model.actor.weights]
    adam_before = [np.asarray(v).copy() for v in trainer.optimizer.variables]
    started = time.monotonic()
    rows = await collect(games=args.critic_warmup_games, seed_base=result['seed_base'],
        index_base=0, mode='selfplay', version=collection_version(), output=rollout_root,
        max_own_turns=args.train_horizon, progress=progress('critic-warmup'))
    train, heldout, partition = split_games(rows, fraction=.2, seed=args.seed+30_000_000, round_id=0)
    heldout_ids = set(partition['validation_indices'])
    for row in rows:
        row['partition'] = 'validation' if row['index'] in heldout_ids else 'training'
    trajectories = [t for row in train for t in row.get('trajectories', ())]
    if not trajectories or not any(row.get('trajectories') for row in heldout):
        raise RuntimeError('Critic warm-up requires terminal training and held-out games')
    result.update(games=rows, split=partition, collection_seconds=time.monotonic()-started)
    normalization = training_normalization(train, trainer.advantage_scale_floor,
        advantage_baseline=getattr(args,'advantage_baseline','critic'))
    async def validate():
        if validation_model is not model:
            neural_call(args.validation_device, validation_model.set_weights, model.get_weights())
        return {name: await asyncio.to_thread(neural_call, args.validation_device,
            validator.evaluate, games, normalization) for name, games in [('training', train), ('validation', heldout)]}
    result['generalization_before'] = await validate()
    atomic_json(report_path, report)
    update_started = time.monotonic()
    result['update'] = await asyncio.to_thread(neural_call, args.learner_device,
        trainer.warmup_critic_files, trajectories, max_steps=args.critic_warmup_max_steps)
    result['update_seconds'] = time.monotonic()-update_started
    if (len(actor_before) != len(model.actor.weights)
            or any(not np.array_equal(a, np.asarray(b)) for a, b in zip(actor_before, model.actor.weights))):
        raise RuntimeError('Critic warm-up changed actor weights')
    if (len(adam_before) != len(trainer.optimizer.variables)
            or any(not np.array_equal(a, np.asarray(b)) for a, b in zip(adam_before, trainer.optimizer.variables))):
        raise RuntimeError('Critic warm-up changed actor Adam state')
    result.update(actor_unchanged=True, actor_optimizer_unchanged=True,
        actor_sha256=array_digest(actor_before), actor_optimizer_sha256=array_digest(adam_before))
    result['generalization_after'] = await validate()
    result.update(status='complete', seconds=time.monotonic()-started)
    # This is the first normal checkpoint, including both optimizer states/RNG.
    # If report publication fails, --resume recovers this result from its state.
    save_checkpoint(args.output/'checkpoints', model, trainer,
        dict(round=0, config=report['config'], critic_warmup=result), policy_type=RustMagePolicy)
    atomic_json(report_path, report)
    await sync_weights(0)
    print(json.dumps(dict(stage='critic-warmup-complete', split=partition,
        update=result['update'], actor_unchanged=True, actor_optimizer_unchanged=True)), flush=True)
    return result


def place_trainer(trainer, device):
    """Trace once on the learner, including graphs later called by PPO threads."""
    traces = {}
    for name in ('step', 'critic_step', 'critic_sequence_step', 'measure', 'drift', 'actor_state_drift', 'measure_with_drift',
                 'measure_actor_with_drift', 'snapshot', 'restore'):
        function = getattr(trainer, name, None)
        if function is None:
            continue
        with device_scope(device):
            function.get_concrete_function()
        traces[name] = function.experimental_get_tracing_count()
        def placed(*arguments, _function=function, _name=name):
            result = neural_call(device, _function, *arguments)
            if _name in ('step', 'critic_step'):
                assert_output_device({'gradient_norm': result}, device)
            elif _name == 'critic_sequence_step':
                assert_output_device({'gradient_norm': result[0], 'memory_out': result[1]}, device)
            return result
        setattr(trainer, name, placed)
    return traces


def initialize_model(args, encoder):
    """A warm start preserves learned weights and leaves its source immutable."""
    if args.initialize_from:
        source = args.initialize_from
        source = source/'policy.keras' if source.is_dir() else source
        model = RustMagePolicy.load(source).model
        if bool(getattr(model, 'separate_value', False)) != (args.architecture == 'separate'):
            raise ValueError('Initialization architecture does not match --architecture')
        if model.card_count != encoder.card_count or dict(model.feature_spec) != encoder.feature_spec:
            raise ValueError('Initialization encoder is incompatible with current RustMage features')
        info = dict(method='checkpoint weights; fresh optimizers and trainer RNG',
            checkpoint=str(source.resolve()), checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        state_path = source.with_name('state.json')
        if state_path.exists():
            state = json.loads(state_path.read_text())
            info.update(initialization_provenance(source, state))
        return model, info
    if args.architecture == 'separate':
        model = SeparateActorCritic(encoder.card_count, encoder.feature_spec)
    else:
        model = PolicyNetwork(encoder.card_count, width=32, memory_size=32, blocks=1,
            candidate_interactions=True, player_readout=True, stable_heads=True,
            isolate_value=True, fact_readout=True, feature_spec=encoder.feature_spec)
    return model, dict(method='fresh random weights', source_collected_games=0)


async def run(args):
    try:
        configure_runtime(args.learner_device, hide_gpu=args.learner_device == 'cpu' and args.inference_device == 'cpu')
        with device_scope(args.learner_device):
            return await run_experiment(args)
    except BaseException as error:
        report_path = args.output/'report.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report.get('status') != 'complete':
                report['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)) else 'error'
            report['last_error'] = dict(type=type(error).__name__, message=str(error),
                recorded_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(report_path, report)
        raise


async def run_experiment(args):
    os.environ['PATCHWORK_OBSERVATION_PIPELINE'] = args.observation_pipeline
    os.environ['PATCHWORK_ROLLOUT_STORAGE'] = args.rollout_storage
    if args.observation_pipeline == 'rust':
        from patchwork_automaton.neural.native_pipeline import library
        library()  # Fail before creating an experiment if the library is missing.
    if min(args.rounds,args.games_per_round,args.eval_games,args.eval_horizon,
           args.train_horizon,args.workers,args.train_batch,args.validation_batch) < 1:
        raise ValueError('Counts and horizons must be positive')
    if args.baseline_eval_games < 0 or args.baseline_eval_games % 4:
        raise ValueError('Baseline evaluation count must be a nonnegative multiple of four')
    if args.baseline_eval_seed is not None and args.baseline_eval_seed < 0:
        raise ValueError('Baseline evaluation seed must be nonnegative')
    if args.inference_device != args.learner_device and args.collector != 'processes':
        raise ValueError('Independent inference and learner devices require the process collector')
    if args.pooled_collector_config and (args.collector != 'processes' or args.train_mode != 'versus_baseline'):
        raise ValueError('Pooled collection requires process workers and fixed-opponent training')
    if args.validation_device == 'gpu' and args.learner_device != 'gpu':
        raise ValueError('GPU validation requires a GPU learner')
    if args.eval_every_games < 0 or args.eval_every_games % args.games_per_round:
        raise ValueError('Evaluation interval must be a nonnegative multiple of games-per-round')
    if not 0 <= args.inference_wait_ms < float('inf'):
        raise ValueError('Inference wait must be finite and nonnegative')
    if args.concurrent_updates and args.architecture != 'separate':
        raise ValueError('Concurrent updates require separate architecture')
    if args.value_learning_rate is not None and not 0 < args.value_learning_rate < float('inf'):
        raise ValueError('Value learning rate must be finite and positive')
    if args.value_max_steps is not None and args.value_max_steps < 1:
        raise ValueError('Value step limit must be positive')
    if args.critic_unroll < 1:
        raise ValueError('Critic unroll length must be positive')
    if args.architecture != 'separate' and (args.critic_update_scope != 'all' or args.critic_unroll != 1):
        raise ValueError('Independent critic recipes require separate architecture')
    if not all(0 < value < float('inf') for value in (args.target_kl, args.max_state_kl)):
        raise ValueError('KL guard limits must be finite and positive')
    if not 0 <= args.entropy_weight < float('inf'):
        raise ValueError('Entropy weight must be finite and nonnegative')
    if args.critic_warmup_games < 0 or args.critic_warmup_games == 1:
        raise ValueError('Critic warm-up needs zero or at least two games')
    if args.critic_warmup_max_steps is not None and args.critic_warmup_max_steps < 1:
        raise ValueError('Critic warm-up step limit must be positive')
    if (args.critic_warmup_games or args.critic_warmup_max_steps is not None) and args.architecture != 'separate':
        raise ValueError('Critic warm-up requires separate architecture')
    if args.critic_warmup_max_steps is not None and not args.critic_warmup_games:
        raise ValueError('Critic warm-up step limit requires warm-up games')
    if args.critic_warmup_games and args.rounds*args.games_per_round >= 30_000_000:
        raise ValueError('PPO and critic warm-up seed ranges would overlap')
    if args.architecture != 'separate' and (args.value_learning_rate is not None or args.value_max_steps is not None):
        raise ValueError('Independent value settings require separate architecture')
    if args.prefetch_batches < 0 or args.measurement_workers < 1:
        raise ValueError('Invalid prefetch depth or measurement worker count')
    if args.rollout_cache_mib < 0:
        raise ValueError('Rollout cache budget must be nonnegative')
    if not 0 <= args.validation_fraction < 1 or (args.validation_fraction and args.games_per_round < 2):
        raise ValueError('Validation needs a fraction in [0,1) and at least two games per round')
    output = args.output
    config = {k:v for k,v in vars(args).items() if k not in ('output','resume','rounds','plot','initialize_from')}
    config['inference_processes'] = inference_process_count(args)
    config.update(algorithm='ppo', advantage_scale_floor=1., ppo_clip=.2)
    report_path = output/'report.json'
    warmup_pending = False
    if args.resume:
        model,trainer,state = load_checkpoint(resume_checkpoint_root(output, args.critic_warmup_games))
        saved_config = dict(state['config'])
        saved_config.setdefault('eval_every_games', 0)
        saved_config.setdefault('collector', 'threads')
        saved_config.setdefault('architecture', 'shared')
        saved_config.setdefault('inference_wait_ms', 2)
        saved_config.setdefault('concurrent_updates', False)
        saved_config.setdefault('prefetch_batches',0)
        saved_config.setdefault('measurement_workers',1)
        saved_config.setdefault('rollout_cache_mib',256)
        saved_config.setdefault('rollout_storage','pickle')
        saved_config.setdefault('inference_processes',1)
        saved_config.setdefault('validation_fraction',0.)
        saved_config.setdefault('value_learning_rate',None)
        saved_config.setdefault('value_max_steps',None)
        saved_config.setdefault('critic_warmup_games',0)
        saved_config.setdefault('critic_warmup_max_steps',None)
        saved_config.setdefault('target_kl',.02)
        saved_config.setdefault('max_state_kl',.1)
        saved_config.setdefault('guard_audit',False)
        saved_config.setdefault('advantage_baseline','critic')
        saved_config.setdefault('baseline_eval_seed',None)
        saved_config.setdefault('entropy_weight',.01)
        saved_config.setdefault('critic_update_scope','all')
        saved_config.setdefault('critic_unroll',1)
        saved_config.setdefault('train_mode','selfplay')
        saved_config.setdefault('pooled_collector_config', None)
        for key, value in dict(learner_device='cpu', inference_device='cpu', validation_device='cpu',
                               validation_batch=32, baseline_eval_games=0).items():
            saved_config.setdefault(key, value)
        # Encoding is numerically equivalent; this runtime choice may change on resume.
        saved_config['observation_pipeline'] = config['observation_pipeline']
        config['inference_processes'] = inference_process_count(args,saved_config)
        if saved_config != config:
            raise ValueError('Resume requires the same experiment configuration; legacy runs without a holdout require explicit --validation-fraction 0')
        if args.rounds < state['round']: raise ValueError('Target round precedes the saved checkpoint')
        first_round = state['round']+1
        report = json.loads(report_path.read_text()) if report_path.is_file() else state.get('initial_report')
        if report is None:
            raise ValueError('Checkpoint is missing its experiment report')
        report['config'] = config
        if state.get('critic_warmup'):
            report['critic_warmup'] = state['critic_warmup']
        warmup_pending = bool(args.critic_warmup_games and state['round'] == 0
            and not state.get('critic_warmup'))
        if args.critic_warmup_games and state['round'] > 0 and report.get('critic_warmup', {}).get('status') != 'complete':
            raise ValueError('PPO checkpoint is missing its completed critic warm-up report')
        report['rounds'] = [r for r in report['rounds'] if r['round'] < first_round]
        # Recover a completed update if interruption occurred after the atomic
        # checkpoint but before publishing its round in report.json.
        if state.get('round_result') and not any(r['round']==state['round'] for r in report['rounds']):
            report['rounds'].append(state['round_result'])
        if [r['round'] for r in report['rounds']] != list(range(1, first_round)):
            raise ValueError('Report is missing completed rounds')
        report['evaluations'] = [e for e in report.get('evaluations', []) if e['round'] < first_round]
        report['opponent_evaluations'] = [e for e in report.get('opponent_evaluations', []) if e['round'] < first_round]
    else:
        output.mkdir(parents=True,exist_ok=False)
        keras.utils.set_random_seed(args.seed)
        encoder = RustMageEncoder()
        model, initialization = initialize_model(args, encoder)
        model(example_batch(model.memory_size,model.feature_spec),training=False)
        trainer = make_trainer(model,config)
        trainer.compile()
        first_round = 1
        report = dict(config=config, encoder=encoder.manifest(), parameters=model.count_params(),
            training=('mono-red versus mono-red PPO; undiscounted terminal +1/-1/0 only; no imitation'
                if args.train_mode == 'selfplay' else
                'mono-red versus active FoundationsAggro PPO; undiscounted terminal +1/-1/0 only; no imitation'),
            evaluation='sampled policy, temperature 1; do-nothing 60-Mountain opponent; fixed held-out seeds',
            concession='Voluntary Concede is excluded from both training and evaluation; no other strategy masking',
            cutoffs='Excluded from PPO; retained as failed evaluation attempts', rounds=[], evaluations=[],
            initialization=initialization,
            initial_training_games=initialization.get('source_collected_games', 0))
        if 'source_optimization_games' in initialization:
            report['initial_optimization_games'] = initialization['source_optimization_games']
            report['initial_validation_games'] = initialization['source_validation_games']
        report['validation_methodology'] = ('Whole-game holdout before outcome filtering; both seats stay together; '
            'training-only advantage normalization; validation never enters either optimizer. '
            'Policy surrogate gain is an outcome-based proxy, not a win rate. '
            'Legacy training_games fields count collected attempts; optimization/validation counts are explicit.')
        report['architecture'] = model.get_config()
        if args.architecture == 'separate':
            report['parameter_counts'] = dict(actor=model.actor.count_params(), critic=model.critic.count_params())
        library=Path('.local/rustmage/target/release/librustmage.so')
        if library.is_file(): report['library_sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
        report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__), *Path('src/patchwork_automaton/neural').glob('*.py'),
            *Path('src/patchwork_automaton/rl').glob('rustmage*.py'),
            Path('src/patchwork_automaton/baselines/foundations.py')]}
        warmup_pending = bool(args.critic_warmup_games)
        checkpoint_root = output/('prewarm-checkpoints' if warmup_pending else 'checkpoints')
        initial_state = dict(round=0,config=config)
        if warmup_pending:
            initial_state['initial_report'] = report
        save_checkpoint(checkpoint_root,model,trainer,initial_state,policy_type=RustMagePolicy)
        atomic_json(report_path,report)
    report['actor_advantage_baseline'] = config['advantage_baseline']
    report['training'] = ('mono-red versus mono-red PPO; undiscounted terminal +1/-1/0 only; no imitation'
        if args.train_mode == 'selfplay' else
        'mono-red versus active FoundationsAggro PPO; undiscounted terminal +1/-1/0 only; no imitation')
    report['training_opponent'] = dict(mode=args.train_mode,
        policy='frozen current neural policy' if args.train_mode == 'selfplay' else 'FoundationsAggro',
        behavior='sampled neural self-play' if args.train_mode == 'selfplay' else 'fixed active scripted opponent',
        recorded_players='both neural seats' if args.train_mode == 'selfplay' else 'learner seat only',
        learner_seat='index % 2', starting_player='P((index // 2) % 2)',
        learner_deck="foundations_deck('red')", opponent_deck="foundations_deck('red')",
        terminal_rewards_only=True, critic_warmup_mode='selfplay')
    placements = assert_model_device(model, args.learner_device)
    trainer_traces = place_trainer(trainer, args.learner_device)
    invocation=dict(first_round=first_round,requested_last_round=args.rounds,
        observation_pipeline=args.observation_pipeline,
        game_engines=args.workers,inference_processes=config['inference_processes'],
        learner_device=args.learner_device, inference_device=args.inference_device,
        learner_variable_placements=placements, learner_graph_traces=trainer_traces,
        host=dict(platform=platform.platform(), available_logical_cpus=sorted(os.sched_getaffinity(0)),
            cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                            if line.startswith('model name')), platform.processor())),
        runtime={key: os.environ.get(key) for key in ('TF_NUM_INTRAOP_THREADS',
            'TF_NUM_INTEROP_THREADS', 'TF_ENABLE_ONEDNN_OPTS', 'CUDA_VISIBLE_DEVICES')},
        started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__),Path('src/patchwork_automaton/rl/rustmage_experiment.py'))})
    if args.observation_pipeline == 'rust':
        from patchwork_automaton.neural.native_pipeline import library
        native_library = Path(library()._name)
        invocation['observation_library_sha256'] = hashlib.sha256(native_library.read_bytes()).hexdigest()
        invocation['source_sha256'].update({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path('native/observation-pipeline/src').glob('*.rs')})
    report.setdefault('invocations',[]).append(invocation)
    report.update(status='running', training_games_completed=(first_round-1)*args.games_per_round,
                  training_games_target=args.rounds*args.games_per_round)
    report.pop('after', None); report.pop('comparison', None)
    atomic_json(report_path,report)
    batcher = collector = None
    validation_model = model
    if (args.validation_fraction or warmup_pending) and args.validation_device != args.learner_device:
        with device_scope(args.validation_device):
            validation_model = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(model))
            validation_model(example_batch(model.memory_size, model.feature_spec), training=False)
            validation_model.set_weights(model.get_weights())
        assert_model_device(validation_model, args.validation_device)
    with device_scope(args.validation_device):
        replay_validator = GameValidator(validation_model, batch_size=args.validation_batch,
            clip_ratio=trainer.clip_ratio,advantage_baseline=args.advantage_baseline) if args.validation_fraction or warmup_pending else None
        validator = replay_validator if args.validation_fraction else None
    def update_game_counts():
        report['collected_games_completed'] = sum(len(r['games']) for r in report['rounds'])
        report['optimization_games_completed'] = sum(r['split']['training']['terminal_games'] if 'split' in r
            else sum(bool(g['trajectories']) for g in r['games']) for r in report['rounds'])
        report['validation_games_completed'] = sum(r.get('split',{}).get('validation',{}).get('attempted_games',0)
            for r in report['rounds'])
        report['validation_terminal_games_completed'] = sum(r.get('split',{}).get('validation',{}).get('terminal_games',0)
            for r in report['rounds'])
    update_game_counts()
    def collection_version():
        return collector.version if collector else batcher.version
    def collection_stats():
        return collector.stats() if collector else batcher.stats()
    async def sync_weights(version):
        if collector: await collector.sync_weights(model,version)
        else: batcher.set_version(version)
    async def collect(**options):
        if collector: return await collector.run_games(**options)
        return await run_games(workers=args.workers, model=model, batcher=batcher, **options)
    def progress(label):
        count=0
        def completed(row):
            nonlocal count
            count+=1
            print(json.dumps(dict(stage=label,complete=count,seed=row['seed'],status=row['status'],
                turns=row['own_turns_to_win'],seconds=round(row['seconds'],2),
                **({'error':row['error']} if row['status']=='error' else {}))),flush=True)
        return completed
    async def evaluate(label,mode):
        evaluation_started=time.monotonic()
        rows=await collect(games=args.eval_games,
            seed_base=args.seed+10_000_000,mode=mode,
            version=collection_version(),max_own_turns=args.eval_horizon,progress=progress(label))
        result=dict(summary=goldfish_summary(rows,args.eval_horizon),games=rows,
                    seconds=time.monotonic()-evaluation_started)
        print(json.dumps(dict(stage=label,summary=result['summary'])),flush=True)
        return result
    def record_evaluation(round_id, result):
        checkpoint=output/'checkpoints'/f'round-{round_id:04d}'/'policy.keras'
        point=dict(round=round_id,training_games=round_id*args.games_per_round,
            checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            **result)
        point['optimization_games'] = sum(r['split']['training']['terminal_games'] if 'split' in r
            else sum(bool(g['trajectories']) for g in r['games']) for r in report['rounds'] if r['round']<=round_id)
        point['validation_games'] = sum(r.get('split',{}).get('validation',{}).get('attempted_games',0)
            for r in report['rounds'] if r['round']<=round_id)
        point['comparison_to_initial']=paired_improvement(report['before']['games'],result['games'],args.eval_horizon)
        report.setdefault('evaluations',[]).append(point)
        atomic_json(report_path,report)
        return point
    async def evaluate_checkpoint(round_id):
        existing=next((e for e in report.get('evaluations',[]) if e['round']==round_id),None)
        if existing is None:
            result=await evaluate(f'eval-{round_id*args.games_per_round:04d}','goldfish')
            existing=record_evaluation(round_id,result)
        await evaluate_opponent(round_id)
        return existing
    async def evaluate_opponent(round_id):
        if not args.baseline_eval_games:
            return None
        existing = next((e for e in report.get('opponent_evaluations', []) if e['round'] == round_id), None)
        if existing is not None:
            return existing
        cohort = baseline_evaluation_config(seed_base=(args.baseline_eval_seed
            if args.baseline_eval_seed is not None else args.seed+20_000_000),
            games=args.baseline_eval_games, max_own_turns=args.train_horizon, greedy=False)
        started = time.monotonic()
        rows = await collect(**cohort['collection'], version=collection_version(),
            progress=progress(f'opponent-eval-{round_id}'))
        checkpoint = output/'checkpoints'/f'round-{round_id:04d}'/'policy.keras'
        point = dict(round=round_id, collected_games=round_id*args.games_per_round,
            checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            config=cohort, games=rows, summary=match_summary(rows, expected_games=args.baseline_eval_games),
            seconds=time.monotonic()-started)
        initial = next((e for e in report.get('opponent_evaluations', []) if e['round'] == 0), None)
        if initial is not None:
            point['comparison_to_initial'] = paired_match_improvement(initial['games'], rows)
        report.setdefault('opponent_evaluations', []).append(point)
        atomic_json(report_path, report)
        print(json.dumps(dict(stage='opponent-evaluation', round=round_id, summary=point['summary'])), flush=True)
        return point
    try:
        if args.collector == 'processes' and (config['inference_processes'] > 1 or args.inference_device != args.learner_device):
            from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector
            collector = ProcessGameCollector(model,workers=args.workers,
                inference_processes=config['inference_processes'],wait_ms=args.inference_wait_ms,
                inference_device=args.inference_device)
            await collector.start(version=first_round-1)
        else:
            batcher = InferenceBatcher(model,max_batch=args.workers,wait_ms=args.inference_wait_ms,
                                       device=args.inference_device)
            await batcher.start()
            batcher.set_version(first_round-1)
            if args.collector == 'processes':
                from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector
                collector = ProcessGameCollector(model,batcher,workers=args.workers)
                await collector.start(version=first_round-1)
        if args.pooled_collector_config:
            from patchwork_automaton.neural.pooled_collector import PooledGameCollector
            collector = PooledGameCollector(collector, model,
                config=json.loads(Path(args.pooled_collector_config).read_text()))
        started = time.monotonic()
        if warmup_pending:
            await critic_warmup(args, model, trainer, replay_validator, validation_model,
                report, collect, sync_weights, collection_version, progress)
        if 'baseline' not in report:
            report['baseline']=await evaluate('baseline','baseline');atomic_json(report_path,report)
        if 'before' not in report:
            report['before']=await evaluate('before','goldfish');atomic_json(report_path,report)
        if not any(e['round']==0 for e in report.get('evaluations',[])):
            record_evaluation(0,report['before'])
        await evaluate_opponent(0)
        # A resumed boundary evaluation must finish before collecting more data.
        completed=first_round-1
        if completed and args.eval_every_games and completed*args.games_per_round % args.eval_every_games == 0:
            await evaluate_checkpoint(completed)
        for round_id in range(first_round,args.rounds+1):
            collection_started=time.monotonic()
            collection_parent_cpu=time.process_time()
            before_collection=collection_stats()
            rows=await collect(games=args.games_per_round,
                seed_base=args.seed+round_id*args.games_per_round,
                index_base=(round_id-1)*args.games_per_round,mode=args.train_mode,
                version=collection_version(),output=output/'rollouts',
                max_own_turns=args.train_horizon,progress=progress(f'train-{round_id}'))
            train_games, heldout_games, partition = split_games(rows, fraction=args.validation_fraction,
                                                               seed=args.seed, round_id=round_id)
            validation_ids = set(partition['validation_indices'])
            for row in rows:
                row['partition'] = 'validation' if row['index'] in validation_ids else 'training'
            trajectories=[t for r in train_games for t in r['trajectories']]
            stats=dict(round=round_id,collection_seconds=time.monotonic()-collection_started,
                       outcomes=dict(Counter(r['status'] for r in rows)),games=rows,
                       split=partition,
                       seat_wins=dict(Counter(str(r.get('winner')) for r in rows if 'winner' in r)))
            after_collection=collection_stats()
            local_rows = [row for row in rows if row.get('collection_host', 'raytracer') == 'raytracer']
            remote_rows = [row for row in rows if row.get('collection_host') == 'laptop']
            collection_cpu=(time.process_time()-collection_parent_cpu
                +sum(row.get('worker_cpu_seconds',0.) for row in local_rows)
                +after_collection.get('replica_cpu_seconds',0.)-before_collection.get('replica_cpu_seconds',0.))
            inference_calls=after_collection['calls']-before_collection['calls']
            inference_requests=after_collection['requests']-before_collection['requests']
            stats['collection_resources']=dict(aggregate_cpu_seconds=collection_cpu,
                mean_logical_cpus=collection_cpu/stats['collection_seconds'],
                available_logical_cpus=len(os.sched_getaffinity(0)),
                games_per_minute=60*len(rows)/stats['collection_seconds'],
                inference_calls=inference_calls, inference_requests=inference_requests,
                mean_inference_batch=inference_requests/max(inference_calls,1),
                inference_processes=config['inference_processes'], game_workers=args.workers)
            if args.pooled_collector_config:
                stats['collection_resources'].update(
                    scope='Raytracer CPU only; laptop CPU recorded separately',
                    local_games=len(local_rows), laptop_games=len(remote_rows),
                    laptop_worker_cpu_seconds=sum(row.get('worker_cpu_seconds',0.) for row in remote_rows),
                    pool=after_collection.get('pool', {}))
            print(json.dumps(dict(stage='collection-resources', round=round_id,
                **stats['collection_resources'])),flush=True)
            if collector:
                frozen = collection_stats()
                stats['collection_policy'] = dict(version=frozen['version'],weights_sha256=frozen['weights_sha256'])
            if not trajectories: raise RuntimeError('No verified terminal games; refusing a fabricated update')
            normalization = training_normalization(train_games,trainer.advantage_scale_floor,
                advantage_baseline=args.advantage_baseline) if validator else None
            # All workers finished; the frozen behavior model can now change.
            update_started=time.monotonic()
            if args.guard_audit:
                trainer.guard_observer = lambda event, _round=round_id: print(
                    json.dumps(dict(stage='guard-proposal', round=_round, **event)), flush=True)
            stats['update']=await asyncio.to_thread(neural_call,args.learner_device,trainer.update_files,trajectories)
            stats['update_seconds']=time.monotonic()-update_started
            if validator:
                validation_started = time.monotonic()
                if validation_model is not model:
                    neural_call(args.validation_device, validation_model.set_weights, model.get_weights())
                stats['generalization'] = dict(
                    training=await asyncio.to_thread(neural_call,args.validation_device,validator.evaluate,train_games,normalization),
                    validation=await asyncio.to_thread(neural_call,args.validation_device,validator.evaluate,heldout_games,normalization))
                stats['validation_seconds'] = time.monotonic()-validation_started
                print(json.dumps(dict(stage='validation',round=round_id,split=partition,
                    metrics={name:{k:value[k] for k in ('value_before','value_after','policy_surrogate_gain')}
                             for name,value in stats['generalization'].items()})),flush=True)
            stats['checkpoint']=str(output/'checkpoints'/f'round-{round_id:04d}')
            save_checkpoint(output/'checkpoints',model,trainer,
                dict(round=round_id,config=config,round_result=stats),policy_type=RustMagePolicy)
            sync_started=time.monotonic()
            await sync_weights(round_id)
            stats['inference_sync_seconds']=time.monotonic()-sync_started
            report['rounds'].append(stats);report['batching']=collection_stats()
            report['training_games_completed']=round_id*args.games_per_round
            update_game_counts()
            atomic_json(report_path,report)
            print(json.dumps(dict(stage='update',round=round_id,metrics=stats['update'])),flush=True)
            if args.eval_every_games and round_id*args.games_per_round % args.eval_every_games == 0:
                await evaluate_checkpoint(round_id)
        final=await evaluate_checkpoint(args.rounds)
        report['after']={key:final[key] for key in ('summary','games')}
        report['comparison']=paired_improvement(report['before']['games'],report['after']['games'],args.eval_horizon)
        report['seconds_this_invocation']=time.monotonic()-started
        invocation.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                          seconds=report['seconds_this_invocation'])
        timed=[r for r in report['rounds'] if 'collection_seconds' in r]
        if timed:
            games=sum(len(r['games']) for r in timed)
            collection=sum(r['collection_seconds'] for r in timed)
            updates=sum(r['update_seconds'] for r in timed)
            validation=sum(r.get('validation_seconds',0.) for r in timed)
            report['throughput']=dict(timed_selfplay_games=games,collection_seconds=collection,
                update_seconds=updates,validation_seconds=validation,
                collection_games_per_minute=60*games/collection,
                collection_and_updates_games_per_minute=60*games/(collection+updates),
                collection_updates_validation_games_per_minute=60*games/(collection+updates+validation))
        report['batching']=collection_stats()
        report['status']='complete'
        atomic_json(report_path,report)
        if args.plot:
            from .plot_rustmage_training import plot_report
            plot_report(report_path)
        print(json.dumps(dict(stage='result',comparison=report['comparison'],report=str(report_path))),flush=True)
    finally:
        if collector: await collector.close()
        if batcher: await batcher.close()


if __name__ == '__main__': asyncio.run(run(parser().parse_args()))
