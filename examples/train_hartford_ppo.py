"""Fresh ten-deck PPO with sampled self-play and rolling checkpoint evaluation.

The readiness gate runs before TensorFlow imports, output creation, or games.
Use --preflight-only to inspect missing engine support without starting a run.
"""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS
from patchwork_automaton.rl.hartford_preflight import hartford_preflight, require_hartford_ready
from patchwork_automaton.rl.hartford_schedule import (
    training_schedule, previous_checkpoints, evaluation_game_count,
    training_seed_bounds, evaluation_seed_bounds)
from patchwork_automaton.rl.standard_ppo import validate_collection, cleanup_rollouts
from examples.train_standard_ppo import CHECKPOINT_FILES, Resources, checkpoint_hashes


PROJECT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    """Durable JSON without importing the neural runtime."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def pin_json(path, expected):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f'Immutable manifest changed: {path}')
    else:
        atomic_json(path, expected)


def checkpoint_bundle_sha256(path):
    """Pin the policy, both optimizers, saved RNG and checkpoint metadata."""
    value = checkpoint_hashes(Path(path))
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def configure_fork(config, args):
    """Admit the fixed 50-round comparison without resetting training state."""
    from patchwork_automaton.rl.hartford_ab import EXTERNAL_BASE, PARENT_ROUND
    parent = args.fork_from.resolve()
    if args.architecture != 'compact-identity' or args.initialize_from is not None or args.initialize_from_sha256:
        raise ValueError('Checkpoint continuation requires identity architecture without weight transfer')
    if args.opponent_recipe not in ('current', 'diverse') or args.rounds != 50 or not args.skip_evaluations:
        raise ValueError('A/B continuation requires one recipe, 50 rounds and --skip-evaluations')
    if not args.fork_from_sha256 or checkpoint_bundle_sha256(parent) != args.fork_from_sha256:
        raise ValueError('Continuation checkpoint bundle hash mismatch')
    state = json.loads((parent / 'state.json').read_text())
    previous = state['config']
    parent_report_path = parent.parent.parent / 'report.json'
    parent_report = json.loads(parent_report_path.read_text())
    if (parent_report.get('status') != 'complete' or parent_report.get('completed_games') != 100000
            or parent_report.get('checkpoints', {}).get('100') != checkpoint_hashes(parent)):
        raise ValueError('Continuation parent has no matching completed 100K report')
    if (state.get('round') != PARENT_ROUND or previous.get('architecture') != 'compact-identity-standard-v2'
            or parent.name != 'round-0100' or not previous.get('use_identity') or previous.get('use_rules')
            or not previous.get('skip_evaluations') or 'trainer_rng_state' not in state
            or not state.get('optimizer_steps') or not state.get('critic_optimizer_steps')):
        raise ValueError('Continuation requires the completed identity checkpoint 100 with optimizer and RNG state')
    unchanged = ('learning_rate', 'value_learning_rate', 'train_batch', 'ppo_epochs', 'entropy_weight',
        'target_kl', 'max_state_kl', 'advantage_scale_floor', 'ppo_clip', 'value_max_steps',
        'critic_update_scope', 'critic_unroll', 'concurrent_updates', 'validation_fraction',
        'game_timeout', 'max_turns', 'max_decisions', 'compact_input_mib', 'training_decks',
        'workers', 'runners', 'inference_processes', 'opponent_inference_processes', 'playback')
    if any(config[key] != previous[key] for key in unchanged):
        raise ValueError('Continuation changed the inherited training recipe')
    prior_low, prior_high = training_seed_bounds(rounds=previous['rounds'], seed=previous['seed'])
    low, high = training_seed_bounds(rounds=args.rounds, seed=args.seed)
    if not (high < prior_low or low > prior_high):
        raise ValueError('Continuation reuses parent collection seeds')
    external = (97, 98, 99) if args.opponent_recipe == 'current' else (25, 50, 75)
    opponents = {}
    for number in external:
        path = parent.parent / f'round-{number:04d}'
        saved = json.loads((path / 'state.json').read_text())
        if (saved.get('round') != number or saved['config'].get('architecture') != previous['architecture']
                or parent_report['checkpoints'].get(str(number)) != checkpoint_hashes(path)):
            raise ValueError('External opponent is not an identity checkpoint from this lineage')
        opponents[str(EXTERNAL_BASE + number)] = dict(path=str(path), hashes=checkpoint_hashes(path),
            parent_checkpoint=number)
    config.update(experiment='hartford-identity-opponent-ab-v1', architecture='compact-identity-standard-v2',
        input_representation='sparse-identity-standard-v2', initialization='continue_full_identity_checkpoint',
        identity_width=16, use_identity=True, use_rules=False, weight_seed=previous['weight_seed'],
        opponent_recipe=args.opponent_recipe, opponent_mode='fork_' + args.opponent_recipe,
        fork=dict(parent_checkpoint=str(parent), parent_hashes=checkpoint_hashes(parent),
            parent_bundle_sha256=args.fork_from_sha256, parent_round=PARENT_ROUND,
            parent_report=str(parent_report_path), parent_report_sha256=sha256(parent_report_path),
            parent_new_collection_games=100000, parent_total_exposure_games=300000, external_opponents=opponents))
    return config


def prepare_fork_checkpoint(output, config):
    """Caller holds the new run's lock. Copy artifacts; never modify the parent."""
    output = Path(output)
    fork = config['fork']
    parent = Path(fork['parent_checkpoint'])
    if checkpoint_hashes(parent) != fork['parent_hashes']:
        raise ValueError('Parent checkpoint changed before continuation preparation')
    if sha256(fork['parent_report']) != fork['parent_report_sha256']:
        raise ValueError('Parent completion report changed')
    original_state = json.loads((parent / 'state.json').read_text())
    provenance = dict(schema=1, fork=fork, original_parent_state=original_state,
        preserved='Policy and both optimizer files are byte-identical; optimizer steps and trainer RNG are retained',
        local_checkpoint_zero_lineage_checkpoint=100, new_rounds=50)
    pin_json(output / 'fork-provenance.json', provenance)
    root = output / 'checkpoints'
    destination = root / 'round-0000'
    state = {key: value for key, value in original_state.items() if key != 'round_result'}
    state.update(round=0, config=config, lineage_checkpoint=100,
        parent_state_sha256=fork['parent_hashes']['state.json'])
    if not destination.exists():
        root.mkdir(exist_ok=True)
        temporary = root / 'round-0000.tmp'
        temporary.mkdir()
        for name in CHECKPOINT_FILES:
            if name != 'state.json':
                shutil.copy2(parent / name, temporary / name)
        atomic_json(temporary / 'state.json', state)
        for path in temporary.iterdir():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        temporary.rename(destination)
        atomic_json(root / 'latest.json', {'checkpoint': destination.name})
    if json.loads((destination / 'state.json').read_text()) != state:
        raise ValueError('Fork checkpoint-zero metadata changed')
    for name, digest in fork['parent_hashes'].items():
        if name != 'state.json' and sha256(destination / name) != digest:
            raise ValueError('Fork checkpoint-zero model or optimizer changed')
    report_path = output / 'report.json'
    if not report_path.exists():
        atomic_json(report_path, dict(status='prepared', config=config, rounds=[],
            checkpoints={'0': checkpoint_hashes(destination)}, evaluations={}, completed_games=0,
            lineage_checkpoint=100, parent_total_exposure_games=300000,
            started_at=datetime.now(timezone.utc).isoformat()))
    return json.loads(report_path.read_text())


def planned_training(config, number):
    if 'fork' in config:
        from patchwork_automaton.rl.hartford_ab import training_schedule as fork_schedule
        return fork_schedule(tuple(HARTFORD_DECKS), round_id=number, seed=config['seed'],
            recipe=config['opponent_recipe'])
    return training_schedule(tuple(HARTFORD_DECKS), round_id=number, seed=config['seed'])


def source_manifest(root=None):
    root = Path(root or PROJECT)
    paths = [p for folder in ('src', 'examples') for p in (root / folder).rglob('*.py')]
    paths.extend((root / 'src/patchwork_automaton/data').glob('*.json'))
    paths.append(root / 'scripts/build-direct-standard.py')
    return {str(path.relative_to(root)): sha256(path) for path in sorted(paths)}


def adopt_memory_fix(output, previous_source, previous_guard, config, pins):
    """Authorize only the audited driver replacement before round one's update.

    Keep checkpoint zero, all recorded games and the original manifest intact.
    Subsequent resumes use the normal strict manifest check without this flag.
    Caller holds run.lock throughout admission and publication.
    """
    output = Path(output)
    recovery = output / 'memory-recovery'
    backup = recovery / 'manifest-before.json'
    old = json.loads((backup if backup.exists() else output / 'manifest.json').read_text())
    report = json.loads((output / 'report.json').read_text())
    guard = json.loads(Path(previous_guard).read_text())
    if (guard.get('reason') != 'low_memory' or guard.get('status') != 'stopped'
            or guard.get('guard_exit_code') != 75):
        raise ValueError('Memory recovery requires a terminated low-memory guard')
    try:
        os.kill(guard['child_pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('Previous experiment process is still alive')
    if (old['config'] != config or report['config'] != config or report['rounds']
            or report['evaluations'] or set(report['checkpoints']) != {'0'}
            or sorted(p.name for p in (output / 'checkpoints').glob('round-*')) != ['round-0000']):
        raise ValueError('Memory recovery is limited to the unchanged checkpoint-zero experiment')
    checkpoint = checkpoint_hashes(output / 'checkpoints/round-0000')
    if checkpoint != report['checkpoints']['0']:
        raise ValueError('Checkpoint zero changed before memory recovery')
    state = json.loads((output / 'checkpoints/round-0000/state.json').read_text())
    if (state.get('round') != 0 or state.get('optimizer_steps') != 0
            or state.get('critic_optimizer_steps') != 0 or state.get('config') != config):
        raise ValueError('Memory recovery requires the original untrained checkpoint zero')
    if source_manifest(previous_source) != old['pins']['source']:
        raise ValueError('Original frozen source changed')
    changed = {p for p in old['pins']['source'].keys() | pins['source'].keys()
               if old['pins']['source'].get(p) != pins['source'].get(p)}
    if changed != {'examples/train_hartford_ppo.py'}:
        raise ValueError('Memory recovery permits only the training driver fix')
    before, after = deepcopy(old['pins']['readiness']), deepcopy(pins['readiness'])
    for value in (before, after):
        for key in ('checkout', 'library', 'build_manifest'):
            value['engine'].pop(key, None)
    if before != after:
        raise ValueError('Engine, card data or readiness changed during recovery')
    engine = old['pins']['readiness']['engine']
    if sha256(engine['library']) != engine['library_sha256']:
        raise ValueError('Original engine library changed')
    planned = training_schedule(tuple(HARTFORD_DECKS), round_id=1, seed=config['seed'])
    policy_hash = sha256(output / 'checkpoints/round-0000/policy.keras')
    planned = [dict(g, opponent_checkpoint_sha256=policy_hash) for g in planned]
    directory = output / 'rollouts/round-0001'
    if json.loads((directory / 'schedule.json').read_text()) != planned:
        raise ValueError('Collected schedule changed')
    rows = read_collection(directory, planned, 0)
    if len(rows) != 1000:
        raise ValueError('Memory recovery requires all 1,000 original games')
    new = dict(config=config, pins=pins)
    pin_json(backup, old)
    pin_json(recovery / 'report-before.json', report)
    pin_json(recovery / 'adoption.json', dict(original_source=str(Path(previous_source).resolve()),
        previous_guard=guard, checkpoint_zero=checkpoint, collected_games=1000,
        collection_sha256=sha256(directory / 'schedule.json'), new_manifest=new,
        runtime_policy=memory_runtime_policy()))
    atomic_json(output / 'manifest.json', new)


def memory_runtime_policy(config=None):
    if config and config.get('architecture') in ('compact-standard-v1','compact-identity-standard-v2'):
        return dict(profile='sparse-device-resident-v1',measurement_workers=1,prefetch_batches=0,
            close_collectors_before_update=True,input_budget_mib=config['compact_input_mib'],
            stateless=True,representation=config.get('input_representation','sparse-factored-standard-v1'))
    return dict(profile='bounded-host-v1', measurement_workers=1, prefetch_batches=0,
                close_collectors_before_update=True)


def adopt_collection_policy(output, previous_source, previous_guard, config, pins):
    """Audit the explicit paused-run amendment without rewriting checkpoints."""
    output=Path(output);archive=output/'collection-policy-recovery-v1'
    backup=archive/'manifest-before.json';report_backup=archive/'report-before.json'
    old=json.loads((backup if backup.exists() else output/'manifest.json').read_text())
    previous=json.loads((report_backup if report_backup.exists() else output/'report.json').read_text())
    guard=json.loads(Path(previous_guard).read_text())
    if guard.get('status') not in ('stopped','exited') or guard.get('reason') not in ('signal','child_exit'):
        raise ValueError('Collection amendment requires a stopped experiment guard')
    try:os.kill(guard['child_pid'],0)
    except ProcessLookupError:pass
    else:raise ValueError('Previous experiment process is still alive')
    allowed={'collection_mode','game_timeout'}
    if (old['config'].get('architecture')!='compact-standard-v1' or previous['config']!=old['config']
        or {k:v for k,v in config.items() if k not in allowed}!={k:v for k,v in old['config'].items() if k not in allowed}
        or config.get('collection_mode')!='mixed-frozen-opponents-v1' or config['game_timeout']!=20):
        raise ValueError('Collection amendment permits only mixed opponents and the approved 20-second deadline')
    if source_manifest(previous_source)!=old['pins']['source']:raise ValueError('Original frozen source changed')
    changed={p for p in old['pins']['source'].keys()|pins['source'].keys() if old['pins']['source'].get(p)!=pins['source'].get(p)}
    permitted={'examples/train_hartford_ppo.py','src/patchwork_automaton/neural/inference_processes.py',
        'src/patchwork_automaton/rl/standard_selfplay_processes.py','src/patchwork_automaton/rl/hartford_evaluation.py'}
    if changed!=permitted:raise ValueError('Collection amendment changed unrelated source')
    before,after=deepcopy(old['pins']['readiness']),deepcopy(pins['readiness'])
    for value in (before,after):
        for key in ('checkout','library','build_manifest'):value['engine'].pop(key,None)
    if before!=after:raise ValueError('Engine or card data changed during collection amendment')
    engine=old['pins']['readiness']['engine']
    if sha256(engine['library'])!=engine['library_sha256']:raise ValueError('Original engine library changed')
    boundary=max(map(int,previous['checkpoints']))
    if sorted(map(int,previous['checkpoints']))!=list(range(boundary+1)) or len(previous['rounds'])!=boundary:
        raise ValueError('Collection amendment requires committed contiguous checkpoint history')
    if sorted(p.name for p in (output/'checkpoints').glob('round-*'))!=[f'round-{n:04d}' for n in range(boundary+1)]:
        raise ValueError('Unpublished checkpoint must be recovered using the previous frozen source')
    for number,expected in previous['checkpoints'].items():
        if checkpoint_hashes(output/'checkpoints'/f'round-{int(number):04d}')!=expected:
            raise ValueError('Checkpoint changed before collection amendment')
    # This admission deliberately refuses to reinterpret an existing evaluation
    # manifest. The paused run has no milestone evaluation yet.
    if list((output/'evaluation').glob('*/manifest.json')):
        raise ValueError('Collection amendment requires a boundary before the first evaluation manifest')
    directory=output/'rollouts'/f'round-{boundary+1:04d}';preserved={}
    if (directory/'schedule.json').exists():
        planned=training_schedule(tuple(HARTFORD_DECKS),round_id=boundary+1,seed=config['seed'])
        for game in planned:game['opponent_checkpoint_sha256']=sha256(output/'checkpoints'/f"round-{game['opponent_version']:04d}"/'policy.keras')
        if json.loads((directory/'schedule.json').read_text())!=planned:raise ValueError('Saved collection schedule changed')
        preserved=read_collection(directory,planned,boundary)
    amended=deepcopy(previous);amended['config']=config
    amendment=dict(previous_config=old['config'],new_config=config,through_checkpoint=boundary,
        preserved_collection_round=boundary+1,preserved_games=len(preserved),
        previous_deadline_applies_to_preserved_games=True)
    amended.setdefault('execution_amendments',[]).append(amendment)
    amended.update(status='paused',completed_games=boundary*1000);amended.pop('error',None)
    manifest=dict(config=config,pins=pins)
    pin_json(backup,old);pin_json(report_backup,previous)
    pin_json(archive/'adoption.json',dict(previous_source=str(Path(previous_source).resolve()),previous_guard=guard,
        changed_source=sorted(changed),amendment=amendment,checkpoints=previous['checkpoints'],
        preserved_results={str(i):sha256(directory/'results'/f'game-{i:06d}.json') for i in preserved},new_manifest=manifest))
    current=json.loads((output/'manifest.json').read_text())
    if current not in (old,manifest):raise ValueError('Current experiment manifest differs from amendment')
    atomic_json(output/'manifest.json',manifest);atomic_json(output/'report.json',amended)
    return amendment


async def quiesce_collectors(collectors):
    """Join all game/inference processes before allocating training batches."""
    await settle_all(c.close() for c in collectors)
    collectors.clear()


def configuration(args):
    compact=args.architecture in ('compact','compact-identity')
    game_timeout = args.game_timeout if args.game_timeout is not None else (20 if compact else 60)
    if not math.isfinite(game_timeout) or game_timeout<=0:
        raise ValueError('Game timeout must be finite and positive')
    if args.rounds < 1 or args.games_per_round != 1000:
        raise ValueError('Positive round count and exactly 1,000 games per round required')
    if args.eval_games < 4 or args.eval_games % 4:
        raise ValueError('Evaluation games must balance four seat/start combinations')
    if args.eval_every_games < 1000 or args.eval_every_games % 1000:
        raise ValueError('Evaluation interval must be a positive multiple of 1,000 games')
    if args.runners < 1 or any(n < args.runners or n % args.runners for n in
            (args.workers, args.inference_processes, args.opponent_inference_processes)):
        raise ValueError('Worker and inference totals must divide evenly among runners')
    if max(args.inference_processes, args.opponent_inference_processes) > args.workers:
        raise ValueError('Inference replicas cannot exceed game workers')
    if not math.isfinite(args.update_timeout) or args.update_timeout <= 0:
        raise ValueError('Positive finite update timeout required')
    if min(args.policy_cache_mib, args.actor_cache_mib) < 0:
        raise ValueError('Cache sizes cannot be negative')
    if not 0 <= args.weight_seed < 2**32 or min(args.seed, args.eval_seed) < 0:
        raise ValueError('Invalid weight or game seed')
    low, high = evaluation_seed_bounds(rounds=args.rounds,
        games_per_cell=args.eval_games, seed=args.eval_seed)
    _, training_end = training_seed_bounds(rounds=args.rounds, seed=args.seed)
    if not (training_end < low or high < args.seed):
        raise ValueError('Training and evaluation seed ranges overlap')
    result = dict(experiment='hartford-selfplay-v1', algorithm='ppo',
        deck_catalog='hartford', training_decks=list(HARTFORD_DECKS),
        seed=args.seed, weight_seed=args.weight_seed, eval_seed=args.eval_seed,
        rounds=args.rounds, games_per_round=1000, validation_fraction=.2,
        eval_games_per_cell=args.eval_games, evaluation_every_round=args.eval_every_games == 1000,
        evaluation_every_games=args.eval_every_games,
        evaluation_opponents='previous_three_distinct',
        opponent_mode='current_70_previous_three_30', action_selection='sampled',
        policy_scope='shared', initialization='random_actor_and_critic',
        learning_rate=.0001, value_learning_rate=.0003, train_batch=512,
        ppo_epochs=2, entropy_weight=.01, target_kl=.02, max_state_kl=.5,
        value_max_steps=157, critic_update_scope='all', critic_unroll=1,
        concurrent_updates=True, prefetch_batches=2, measurement_workers=4,
        rollout_cache_mib=512, advantage_scale_floor=1., ppo_clip=.2,
        workers=args.workers, runners=args.runners,
        inference_processes=args.inference_processes,
        opponent_inference_processes=args.opponent_inference_processes,
        game_timeout=game_timeout, max_turns=50, max_decisions=6000,
        policy_cache_mib=args.policy_cache_mib, actor_cache_mib=args.actor_cache_mib,
        compact_input_cache=True, release_host_memory=True,
        numeric_disk_gib=128, policy_disk_gib=64, rollout_storage='pickle',
        playback=args.playback, update_timeout=args.update_timeout)
    if compact:
        if type(args.compact_input_mib) is not int or args.compact_input_mib<1:
            raise ValueError('Compact input budget must be positive integer MiB')
        result.update(experiment='hartford-selfplay-compact-v1',architecture='compact-standard-v1',
            input_representation='sparse-factored-standard-v1',actor_width=32,critic_width=32,
            recurrent_memory=False,event_history=False,previous_action_features=False,
            compact_input_mib=args.compact_input_mib,prefetch_batches=0,measurement_workers=1,
            collection_mode='mixed-frozen-opponents-v1',
            policy_cache_mib=0,actor_cache_mib=0,numeric_disk_gib=0,policy_disk_gib=0,
            minimum_free_disk_gib=32)
    source=getattr(args,'initialize_from',None)
    digest=getattr(args,'initialize_from_sha256',None)
    if getattr(args, 'fork_from', None) is not None:
        result = configure_fork(result, args)
    elif args.architecture=='compact-identity':
        if source is None or digest is None:
            raise ValueError('Identity experiment requires --initialize-from and --initialize-from-sha256')
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid initialization checkpoint SHA256')
        if sha256(source)!=digest:raise ValueError('Initialization checkpoint hash mismatch')
        result.update(experiment='hartford-selfplay-identity-v2',architecture='compact-identity-standard-v2',
            input_representation='sparse-identity-standard-v2',initialization='transfer_from_compact_v1',
            initialize_from=str(source.resolve()),initialize_from_sha256=digest,
            identity_width=16,use_identity=True,use_rules=False)
    elif source is not None or digest is not None:
        raise ValueError('Initialization transfer requires --architecture compact-identity')
    if getattr(args, 'fork_from', None) is None and (getattr(args, 'fork_from_sha256', None)
            or getattr(args, 'opponent_recipe', None) or getattr(args, 'prepare_fork_only', False)):
        raise ValueError('Continuation options require --fork-from')
    if getattr(args, 'skip_evaluations', False):
        result['skip_evaluations'] = True
    return result


def initialize_random(config, *, device='gpu'):
    """Construct the configured model; identity transfer starts fresh optimizers."""
    import keras
    from patchwork_automaton.neural.batching import model_example_batch
    from patchwork_automaton.neural.devices import device_scope
    from patchwork_automaton.neural.experiment import make_trainer
    from patchwork_automaton.neural.model import SeparateActorCritic
    from patchwork_automaton.neural.recurrent import configure_gru_backend
    from patchwork_automaton.neural.standard_features import StandardEncoder
    keras.utils.set_random_seed(config['weight_seed'])
    encoder = StandardEncoder()
    with device_scope(device):
        if config.get('architecture')=='compact-identity-standard-v2':
            from patchwork_automaton.neural.standard_policy import StandardPolicy
            from patchwork_automaton.neural.compact_identity_model import transfer_v1
            path=Path(config['initialize_from'])
            if sha256(path)!=config['initialize_from_sha256']:
                raise ValueError('Initialization checkpoint changed')
            source=StandardPolicy.load(path).model
            model=transfer_v1(source,identity_width=config['identity_width'],
                use_identity=config['use_identity'],use_rules=config['use_rules'])
            del source
            if sha256(path)!=config['initialize_from_sha256']:
                raise ValueError('Initialization checkpoint changed during transfer')
        elif config.get('architecture')=='compact-standard-v1':
            from patchwork_automaton.neural.compact_standard_model import CompactActorCritic
            model=CompactActorCritic(encoder.card_count,encoder.feature_spec,
                actor_width=config['actor_width'],critic_width=config['critic_width'])
        else:
            model = SeparateActorCritic(encoder.card_count, encoder.feature_spec)
        model(model_example_batch(model), training=False)
        if not getattr(model,'stateless',False):configure_gru_backend(model, 'cudnn-memory' if device == 'gpu' else 'unfused')
        trainer = make_trainer(model, config)
    if int(trainer.optimizer.iterations) or int(trainer.critic_optimizer.iterations):
        raise AssertionError('Fresh initialization reused optimizer steps')
    return model, trainer


def training_state(model, trainer):
    import numpy as np
    from patchwork_automaton.neural.inference_processes import weights_digest
    return (weights_digest(model.get_weights()),
        weights_digest([np.asarray(v) for v in trainer.optimizer.variables]),
        weights_digest([np.asarray(v) for v in trainer.critic_optimizer.variables]),
        deepcopy(trainer.rng.bit_generator.state))


async def settle_all(awaitables):
    """Drain sibling runners before closing their collectors after a failure."""
    results = await asyncio.gather(*awaitables, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


def assert_runtime_unchanged(pins):
    if source_manifest() != pins['source']:
        raise ValueError('Source changed during experiment')
    engine = pins['readiness']['engine']
    if sha256(engine['library']) != engine['library_sha256']:
        raise ValueError('Selected engine binary changed during experiment')


def checkpoint_configuration(report, config, number):
    """An audited cadence amendment never rewrites historical checkpoints."""
    disabled = report.get('evaluation_disable_amendment')
    if disabled is not None:
        previous = disabled['previous_config']
        boundary = disabled['through_checkpoint']
        if (disabled['new_config'] != config or
                config != dict(previous, skip_evaluations=True) or
                previous.get('skip_evaluations', False) or
                type(boundary) is not int or not 0 <= boundary <= config['rounds']):
            raise ValueError('Evaluation disable amendment changed the training configuration')
        history = dict(report)
        del history['evaluation_disable_amendment']
        historical = checkpoint_configuration(history, previous, min(number, boundary))
        return historical if number <= boundary else config
    extension = report.get('training_target_amendment')
    if extension is not None:
        previous = extension['previous_config']
        boundary = extension['through_checkpoint']
        if (extension['new_config'] != config or type(boundary) is not int or
                boundary != previous['rounds'] or type(config['rounds']) is not int or
                config['rounds'] <= boundary or
                {k: v for k, v in previous.items() if k != 'rounds'} !=
                {k: v for k, v in config.items() if k != 'rounds'}):
            raise ValueError('Target extension changed the training configuration')
        history = dict(report)
        del history['training_target_amendment']
        historical = checkpoint_configuration(history, previous, min(number, boundary))
        return historical if number <= boundary else config
    for amendment in reversed(report.get('execution_amendments',[])):
        previous=amendment['previous_config'];allowed={'collection_mode','game_timeout'}
        if (amendment['new_config']!=config or
            {k:v for k,v in previous.items() if k not in allowed}!={k:v for k,v in config.items() if k not in allowed}
            or type(amendment['through_checkpoint']) is not int):
            raise ValueError('Execution amendment changed the training configuration')
        if number<=amendment['through_checkpoint']:config=previous
    amendment = report.get('evaluation_cadence_amendment')
    if amendment is None:
        return config
    previous = amendment['previous_config']
    cadence_keys = {'evaluation_every_games', 'evaluation_every_round'}
    if (amendment['new_config'] != config or
            {k: v for k, v in previous.items() if k not in cadence_keys} !=
            {k: v for k, v in config.items() if k not in cadence_keys}):
        raise ValueError('Cadence amendment changed the training configuration')
    boundary = amendment['through_checkpoint']
    if type(boundary) is not int or not 0 <= boundary <= config['rounds']:
        raise ValueError('Invalid cadence amendment checkpoint boundary')
    return previous if number <= boundary else config


def evaluation_source_pins(report, pins, number):
    """Reaudit completed historical games using their original source manifest."""
    disabled = report.get('evaluation_disable_amendment')
    if disabled is not None:
        checkpoint_configuration(report, report['config'], number)
        if (pins != disabled['new_pins'] or number > disabled['through_checkpoint'] or
                not report.get('evaluations', {}).get(str(number), {}).get('complete')):
            raise ValueError('Evaluation disable amendment cannot reinterpret evaluation history')
        pins = disabled['previous_pins']
    for amendment in reversed(report.get('runtime_amendments', [])):
        if amendment['new_pins'] != pins or type(amendment['through_checkpoint']) is not int:
            raise ValueError('Runtime source amendment chain changed')
        if number > amendment['through_checkpoint']:
            break
        if not report.get('evaluations', {}).get(str(number), {}).get('complete'):
            raise ValueError('Runtime amendment cannot reinterpret unfinished historical evaluation')
        pins = amendment['previous_pins']
    extension = report.get('training_target_amendment')
    if extension and number <= extension['through_checkpoint']:
        checkpoint_configuration(report, report['config'], number)
        if not report.get('evaluations', {}).get(str(number), {}).get('complete'):
            raise ValueError('Target extension cannot reinterpret unfinished historical evaluation')
        return extension['previous_pins']
    return pins


def required_evaluation_rounds(config, report):
    """Keep completed historical evaluations and evaluate future milestones."""
    disabled = report.get('evaluation_disable_amendment')
    if disabled is not None:
        checkpoint_configuration(report, config, 0)
        retained = disabled['retained_evaluation_rounds']
        if (len(retained) != len(set(retained)) or
                any(type(n) is not int or not 1 <= n <= disabled['through_checkpoint']
                    for n in retained) or
                set(map(str, retained)) != set(report.get('evaluations', {})) or
                any(not e.get('complete') for e in report['evaluations'].values())):
            raise ValueError('Invalid retained evaluation history after disabling evaluations')
        return set(retained)
    if config.get('skip_evaluations', False):
        if report.get('evaluations'):
            raise ValueError('Disabling historical evaluations requires an audited amendment')
        return set()
    interval = config.get('evaluation_every_games', 1000) // 1000
    if interval < 1 or config.get('evaluation_every_games', 1000) % 1000:
        raise ValueError('Invalid evaluation interval')
    rounds = set(range(interval, config['rounds'] + 1, interval)) | {config['rounds']}
    amendment = report.get('evaluation_cadence_amendment')
    if amendment:
        checkpoint_configuration(report, config, 0)
        retained = amendment['retained_evaluation_rounds']
        if (len(retained) != len(set(retained)) or
                any(type(n) is not int or not 1 <= n <= amendment['through_checkpoint']
                    for n in retained)):
            raise ValueError('Invalid retained evaluation history')
        rounds.update(retained)
    return rounds


def validate_evaluation_completion(report, config):
    """Completion counts include preserved history as well as new milestones."""
    expected = {str(k) for k in required_evaluation_rounds(config, report)}
    if set(report['evaluations']) != expected or any(
            not item.get('complete') or
            item['aggregate']['games'] != min(int(key), 3) * 100 * config['eval_games_per_cell']
            for key, item in report['evaluations'].items()):
        raise ValueError('Experiment is missing a complete checkpoint evaluation')
    return sum(item['aggregate']['games'] for item in report['evaluations'].values())


def restore_history(report, root, config):
    """Recover checkpoint/report publication gaps without repeating an update."""
    root = Path(root)
    names = sorted(p.name for p in root.glob('round-*') if p.is_dir() and
                   len(p.name) == 10 and p.name[6:].isdigit())
    if not names:
        if report['rounds'] or report['checkpoints']:
            raise ValueError('Report references missing checkpoints')
        return None
    expected = [f'round-{i:04d}' for i in range(len(names))]
    if names != expected:
        raise ValueError('Checkpoint history is not contiguous from zero')
    completed = len(names) - 1
    if completed > config['rounds'] or len(report['rounds']) not in (completed, completed - 1):
        raise ValueError('Checkpoint/report history mismatch')
    states = []
    for number, name in enumerate(names):
        path = root / name
        actual = checkpoint_hashes(path)
        known = report['checkpoints'].get(str(number))
        if known is not None and known != actual:
            raise ValueError(f'Checkpoint {number} changed')
        if known is None and number < completed:
            raise ValueError('Missing historical checkpoint pin')
        state = json.loads((path / 'state.json').read_text())
        if (state.get('round') != number or
                state.get('config') != checkpoint_configuration(report, config, number)):
            raise ValueError('Checkpoint round/configuration mismatch')
        report['checkpoints'][str(number)] = actual
        states.append(state)
    if len(report['rounds']) < completed:
        report['rounds'].append(states[-1]['round_result'])
    if [r['round'] for r in report['rounds']] != list(range(1, completed + 1)):
        raise ValueError('Noncontiguous round results')
    for number, result in enumerate(report['rounds'], 1):
        if result != states[number]['round_result']:
            raise ValueError('Round report differs from its pinned checkpoint')
        if 'games_file' in result and sha256(result['games_file']) != result.get('games_sha256'):
            raise ValueError('Committed game summaries changed')
        directory = root.parent / 'rollouts' / f'round-{number:04d}'
        if directory.exists():
            pin_json(directory / 'committed.json',
                     dict(round=number, checkpoint=str(root / f'round-{number:04d}')))
    # The checkpoint directory rename precedes latest.json publication.
    latest = root / 'latest.json'
    if latest.exists():
        old = json.loads(latest.read_text()).get('checkpoint')
        if old not in names[-2:]:
            raise ValueError('Latest checkpoint pointer is inconsistent')
    elif completed:
        raise ValueError('Missing checkpoint publication history')
    atomic_json(latest, {'checkpoint': names[-1]})
    return completed


def read_collection(directory, planned, behavior_version):
    """Reject stale, failed, duplicate, or damaged persisted games on resume."""
    rows = {}
    expected = {g['index']: g for g in planned}
    for path in sorted((Path(directory) / 'results').glob('game-*.json')):
        row = json.loads(path.read_text())
        index = row.get('index')
        if index not in expected or index in rows or path.name != f'game-{index:06d}.json':
            raise ValueError('Foreign or duplicate collection result')
        validate_collection([row], [expected[index]], behavior_version)
        for trajectory in row['trajectories']:
            if trajectory.get('sha256') != sha256(trajectory['path']):
                raise ValueError('Recorded trajectory changed')
        rows[index] = row
    return rows


async def collect_round(*, planned, number, directory, collectors, pin_opponent, progress,
                        pin_opponents=None, game_timeout=60):
    directory = Path(directory)
    pin_json(directory / 'schedule.json', planned)
    rows = read_collection(directory, planned, number - 1)
    if pin_opponents is not None:
        remaining = [g for g in planned if g['index'] not in rows]
        if remaining:
            digests = await pin_opponents(sorted({g['opponent_version'] for g in remaining
                if g.get('opponent_kind', 'frozen_neural') == 'frozen_neural'}))
            expected = {g['index']:g for g in remaining}
            def persist_mixed(row):
                row = dict(row, actions=None);index = row['index']
                if index in rows or index not in expected:raise ValueError('Duplicate or unexpected game completion')
                for trajectory in row.get('trajectories',()):trajectory['sha256']=sha256(trajectory['path'])
                atomic_json(directory/'results'/f'game-{index:06d}.json',row)
                validate_collection([row],[expected[index]],number-1);rows[index]=row
                if len(rows)%25==0 or len(rows)==len(planned):
                    progress('collection',completed=len(rows),total=len(planned),opponent='mixed')
            jobs=[dict(seed=g['seed'],index=g['index'],game=g,
                **({'opponent_weights_sha256':digests[g['opponent_version']]}
                    if g.get('opponent_kind', 'frozen_neural') == 'frozen_neural' else {})) for g in remaining]
            await settle_all(collector.run_schedule(jobs[i::len(collectors)],progress=persist_mixed,
                game_runner='standard_ppo',output=directory/'trajectories',diagnostics=directory/'adjudications',
                version=number-1,game_timeout=game_timeout,max_turns=50,max_decisions=6000,greedy=False)
                for i,collector in enumerate(collectors))
        ordered=sorted(rows.values(),key=lambda g:g['index'])
        validate_collection(ordered,planned,number-1)
        return ordered
    for opponent in sorted({g['opponent_version'] for g in planned}, reverse=True):
        remaining = [g for g in planned if g['opponent_version'] == opponent and g['index'] not in rows]
        if not remaining:
            continue
        await pin_opponent(opponent)
        digest = remaining[0]['opponent_checkpoint_sha256']
        def persist(row):
            row = dict(row, actions=None)
            # Persist errors as evidence; never replace them by draws or omit them.
            for trajectory in row.get('trajectories', ()):
                trajectory['sha256'] = sha256(trajectory['path'])
            path = directory / 'results' / f"game-{row['index']:06d}.json"
            if path.exists() or row['index'] in rows:
                raise ValueError('Duplicate game completion')
            atomic_json(path, row)
            expected = next(g for g in remaining if g['index'] == row['index'])
            validate_collection([row], [expected], number - 1)
            rows[row['index']] = row
            if len(rows) % 100 == 0:
                progress('collection', completed=len(rows), total=len(planned), opponent=opponent)
        jobs = [dict(seed=g['seed'], index=g['index'], game=g) for g in remaining]
        await settle_all(collector.run_schedule(jobs[i::len(collectors)],
            progress=persist, game_runner='standard_ppo', output=directory / 'trajectories',
            diagnostics=directory / 'adjudications', version=number - 1,
            opponent_version=opponent, opponent_sha256=digest,
            game_timeout=game_timeout, max_turns=50, max_decisions=6000, greedy=False)
            for i, collector in enumerate(collectors))
    ordered = sorted(rows.values(), key=lambda g: g['index'])
    validate_collection(ordered, planned, number - 1)
    return ordered


def verify_behavior(model, rows):
    import numpy as np
    from patchwork_automaton.neural.devices import device_scope
    from patchwork_automaton.neural.features import collate
    from patchwork_automaton.neural.batching import record_batch
    from patchwork_automaton.neural.rollout_records import read_trajectory
    checked = 0
    for row in rows[::max(1, len(rows) // 24)]:
        for trajectory in row['trajectories']:
            stream = iter(read_trajectory(trajectory['path']))
            for _ in range(2):
                sample = next(stream, None)
                if sample is None:
                    break
                with device_scope('gpu'):
                    inputs=record_batch(model,[sample])
                    output = model(inputs, training=False)
                logits = np.asarray(output['logits'])[0].astype(np.float64)
                logp = logits - logits.max()
                logp -= np.log(np.exp(logp).sum())
                np.testing.assert_allclose(logp, sample.log_probs, rtol=5e-4, atol=2e-4)
                np.testing.assert_allclose(np.asarray(output['value'])[0, 0], sample.value, rtol=5e-4, atol=2e-4)
                checked += 1
    return dict(samples=checked)


async def run(args):
    adoption=args.adopt_memory_fix_from or args.adopt_collection_policy_from
    if (bool(adoption)!=bool(args.previous_guard) or adoption and not args.resume
            or args.adopt_memory_fix_from and args.adopt_collection_policy_from):
        raise ValueError('Policy adoption requires --resume, one adoption source and --previous-guard')
    config = configuration(args)
    readiness = hartford_preflight(playback=args.playback)
    if args.preflight_only:
        print(json.dumps(readiness, indent=2), flush=True)
        return readiness
    require_hartford_ready(readiness)
    if args.output is None:
        raise ValueError('--output is required for training')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Existing experiment requires --resume')
    if args.resume and not (args.output / 'manifest.json').is_file():
        raise ValueError('Resume requires an existing experiment manifest')
    if args.resume:
        saved=json.loads((args.output/'manifest.json').read_text())['config']
        if saved.get('architecture','legacy')!=config.get('architecture','legacy'):
            raise ValueError('Architecture changed: compact Hartford requires a fresh output directory; resume legacy checkpoints from their frozen legacy source')
    disk_floor=config.get('minimum_free_disk_gib',200)
    if shutil.disk_usage(args.output.parent if args.output.parent.exists() else PROJECT).free < disk_floor * 1024**3:
        raise RuntimeError(f'At least {disk_floor} GiB free disk required for bounded update caches')
    if getattr(args, 'prepare_fork_only', False):
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=args.resume)
        with (args.output / 'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pins = dict(source=source_manifest(), readiness=readiness)
            pin_json(args.output / 'manifest.json', dict(config=config, pins=pins))
            prepared = prepare_fork_checkpoint(args.output, config)
        print(json.dumps(dict(status=prepared['status'], output=str(args.output),
            checkpoint_zero=prepared['checkpoints']['0']), indent=2), flush=True)
        return prepared
    os.environ.update(PATCHWORK_STANDARD_PLAYBACK=args.playback, PATCHWORK_ROLLOUT_STORAGE='pickle',
                      PATCHWORK_OBSERVATION_PIPELINE='python')
    if os.environ.get('KERAS_BACKEND', 'tensorflow') != 'tensorflow':
        raise ValueError('Hartford PPO requires the TensorFlow Keras backend')
    for key, value in dict(KERAS_BACKEND='tensorflow', TF_ENABLE_ONEDNN_OPTS='0',
                          TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2',
                          OPENBLAS_NUM_THREADS='1', TF_CPP_MIN_LOG_LEVEL='3').items():
        os.environ.setdefault(key, value)
    import keras
    import numpy as np
    from patchwork_automaton.neural.batching import model_example_batch
    from patchwork_automaton.neural.devices import configure_runtime, device_scope, assert_model_device
    from patchwork_automaton.neural.experiment import load_checkpoint, save_checkpoint
    from patchwork_automaton.neural.host_memory import HostMemoryReclaimer
    from patchwork_automaton.neural.inference_processes import weights_digest
    from patchwork_automaton.neural.standard_policy import StandardPolicy
    from patchwork_automaton.neural.update_pipeline import resolve_update_pipeline, apply_update_pipeline
    from patchwork_automaton.neural.validation import GameValidator, training_normalization
    from patchwork_automaton.rl.hartford_evaluation import evaluate_checkpoint
    from patchwork_automaton.rl.standard_selfplay_processes import StandardSelfplayCollector
    from examples.train_rustmage import place_trainer

    configure_runtime('gpu')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    collectors = []
    old_tempdir = tempfile.tempdir
    report_path = args.output / 'report.json'
    report = None
    try:
        pins = dict(source=source_manifest(), readiness=readiness)
        if args.adopt_memory_fix_from:
            adopt_memory_fix(args.output, args.adopt_memory_fix_from, args.previous_guard, config, pins)
        if args.adopt_collection_policy_from:
            adopt_collection_policy(args.output,args.adopt_collection_policy_from,args.previous_guard,config,pins)
        pin_json(args.output / 'manifest.json', dict(config=config, pins=pins))
        if 'fork' in config:
            prepare_fork_checkpoint(args.output, config)
        (args.output / 'cache').mkdir(exist_ok=True)
        tempfile.tempdir = str(args.output / 'cache')
        root = args.output / 'checkpoints'
        report = json.loads(report_path.read_text()) if report_path.exists() else dict(
            status='initializing', config=config, rounds=[], checkpoints={}, evaluations={},
            started_at=datetime.now(timezone.utc).isoformat())
        if report['config'] != config:
            raise ValueError('Saved experiment configuration changed')
        completed = restore_history(report, root, config)
        if completed is None:
            model, trainer = initialize_random(config)
            save_checkpoint(root, model, trainer, dict(round=0, config=config), policy_type=StandardPolicy)
            report['checkpoints']['0'] = checkpoint_hashes(root / 'round-0000')
            completed = 0
        else:
            with device_scope('gpu'):
                model, trainer, state = load_checkpoint(root, policy_type=StandardPolicy, gru_backend='cudnn-memory')
            if state['round'] != completed:
                raise ValueError('Restored the wrong checkpoint')
        report.update(status='running', completed_games=completed * 1000)
        report.pop('error', None)
        atomic_json(report_path, report)
        assert_model_device(model, 'gpu')
        place_trainer(trainer, 'gpu')
        if config.get('architecture') not in ('compact-standard-v1','compact-identity-standard-v2'):
            pipeline = resolve_update_pipeline('gpu', 'gpu', policy_cache_mib=args.policy_cache_mib,
                actor_epoch_cache_mib=args.actor_cache_mib, compact_inputs=True)
            apply_update_pipeline(trainer, pipeline)
        # Resource concurrency is separate from the immutable PPO recipe saved
        # in checkpoint zero. Canonical reduction order and minibatches stay fixed.
        trainer.measurement_workers = 1
        trainer.prefetch_batches = 0
        report['runtime_policy'] = memory_runtime_policy(config)
        from patchwork_automaton.neural.standard_features import STANDARD_CANDIDATE_LIMIT
        report['action_candidate_policy'] = dict(limit=STANDARD_CANDIDATE_LIMIT,
            retained='deterministic legal prefix', selection='sampled temperature 1')
        atomic_json(report_path, report)
        trainer.numeric_disk_limit_bytes = 128 * 1024**3
        trainer.policy_disk_limit_bytes = 64 * 1024**3
        trainer.update_timeout_seconds = args.update_timeout
        reclaimer = HostMemoryReclaimer()
        with device_scope('cpu'):
            validation_model = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(model))
            validation_model(model_example_batch(validation_model), training=False)
            validator = GameValidator(validation_model, batch_size=512 if getattr(model,'stateless',False) else 32)
            opponent_model = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(model))
            opponent_model(model_example_batch(opponent_model), training=False)
            opponent_model.set_weights(model.get_weights())
        async def ensure_collectors(version):
            if not collectors:
                collectors.extend(StandardSelfplayCollector(model, opponent_model,
                    workers=args.workers // args.runners,
                    inference_processes=args.inference_processes // args.runners,
                    opponent_inference_processes=args.opponent_inference_processes // args.runners)
                    for _ in range(args.runners))
                await settle_all(c.start(version=version, opponent_version=version) for c in collectors)

        def progress(number, phase, **values):
            payload = dict(round=number, phase=phase, **values)
            atomic_json(args.output / 'progress.json', payload)
            print(json.dumps(payload), flush=True)

        def check_checkpoint(number):
            external = config.get('fork', {}).get('external_opponents', {}).get(str(number))
            if external is not None:
                path = Path(external['path'])
                if checkpoint_hashes(path) != external['hashes']:
                    raise ValueError(f'External frozen opponent {number} changed')
                return path / 'policy.keras'
            path = root / f'round-{number:04d}'
            if checkpoint_hashes(path) != report['checkpoints'][str(number)]:
                raise ValueError(f'Frozen checkpoint {number} changed')
            return path / 'policy.keras'

        async def pin_opponent(number):
            before = training_state(model, trainer)
            # The list passed to collect_round is populated only when it has
            # missing games. A complete saved rollout needs no inference workers.
            await ensure_collectors(active_version)
            path = check_checkpoint(number)
            with device_scope('cpu'):
                loaded = StandardPolicy.load(path).model
                opponent_model.set_weights(loaded.get_weights())
            del loaded
            gc.collect()
            await settle_all(c.sync_opponent(opponent_model, number) for c in collectors)
            digest = weights_digest(opponent_model.get_weights())
            if any(c.opponent_digest != digest for c in collectors):
                raise AssertionError('Opponent replicas disagree')
            if training_state(model, trainer) != before:
                raise AssertionError('Opponent loading changed learner state')

        async def pin_opponents(numbers):
            before=training_state(model,trainer)
            await quiesce_collectors(collectors)
            # A resumed remainder can contain only scripted games. Keep a
            # valid idle opponent service so per-game reset handshakes work.
            numbers = numbers or [active_version]
            weights={}
            for number in numbers:
                with device_scope('cpu'):
                    loaded=StandardPolicy.load(check_checkpoint(number)).model
                    weights[number]=loaded.get_weights()
                del loaded
            digests={n:weights_digest(w) for n,w in weights.items()}
            collectors.extend(StandardSelfplayCollector(model,opponent_model,
                workers=args.workers//args.runners,inference_processes=args.inference_processes//args.runners,
                opponent_inference_processes=args.opponent_inference_processes//args.runners) for _ in range(args.runners))
            await settle_all(c.start(version=active_version,opponent_weights=weights) for c in collectors)
            if any(c.replicas[1].version_digests!=digests for c in collectors):
                raise AssertionError('Opponent banks disagree')
            if training_state(model,trainer)!=before:raise AssertionError('Opponent loading changed learner state')
            return digests

        def release(stage):
            before = training_state(model, trainer)
            result = reclaimer.release(stage)
            if training_state(model, trainer) != before:
                raise AssertionError('Memory reclamation changed learner state')
            return result

        async def evaluate(number):
            if not number or number not in required_evaluation_rounds(config, report):
                return
            assert_runtime_unchanged(pins)
            before = training_state(model, trainer)
            paths = {k: check_checkpoint(k) for k in (number, *previous_checkpoints(number))}
            mixed=config.get('collection_mode')=='mixed-frozen-opponents-v1'
            if mixed:await quiesce_collectors(collectors)
            else:await ensure_collectors(number)
            progress(number, 'head_to_head')
            result = await evaluate_checkpoint(model=model, checkpoint_round=number,
                checkpoint_paths=paths, output=args.output / 'evaluation' / f'round-{number:04d}',
                deck_ids=tuple(HARTFORD_DECKS), seed=args.eval_seed, games_per_cell=args.eval_games,
                workers=args.workers, runners=args.runners, inference_processes=args.inference_processes,
                opponent_inference_processes=args.opponent_inference_processes,
                source_manifest=evaluation_source_pins(report, pins, number), opponent_model=opponent_model, collectors=None if mixed else collectors,
                mixed_opponents=mixed,game_timeout=config['game_timeout'],
                progress=lambda event: progress(number, 'head_to_head',
                    **{key: value for key, value in event.items() if key != 'phase'}))
            if training_state(model, trainer) != before:
                raise AssertionError('Head-to-head evaluation changed model, optimizers or RNG')
            for k in paths:
                check_checkpoint(k)
            assert_runtime_unchanged(pins)
            report['evaluations'][str(number)] = result
            release('after_evaluation')
            atomic_json(report_path, report)

        # A committed update is never repeated just because evaluation was interrupted.
        await evaluate(completed)
        for number in range(completed + 1, args.rounds + 1):
            active_version = number - 1
            assert_runtime_unchanged(pins)
            planned = planned_training(config, number)
            opponent_hashes = {k: sha256(check_checkpoint(k)) for k in
                {g['opponent_version'] for g in planned if g['opponent_kind'] == 'frozen_neural'}}
            for game in planned:
                if game['opponent_kind'] == 'frozen_neural':
                    game['opponent_checkpoint_sha256'] = opponent_hashes[game['opponent_version']]
            directory = args.output / 'rollouts' / f'round-{number:04d}'
            before = training_state(model, trainer)
            with Resources(args.output) as resources:
                started = time.monotonic()
                rows = await collect_round(planned=planned, number=number, directory=directory,
                    collectors=collectors, pin_opponent=pin_opponent,
                    pin_opponents=pin_opponents if config.get('collection_mode')=='mixed-frozen-opponents-v1' else None,
                    game_timeout=config['game_timeout'],
                    progress=lambda phase, **kw: progress(number, phase, **kw))
                collection_seconds = time.monotonic() - started
                behavior_digest = weights_digest(model.get_weights())
                collection_inference=[c.stats() for c in collectors]
                await quiesce_collectors(collectors)
                collection_memory = release('after_collection')
                reproduction = verify_behavior(model, rows)
                if training_state(model, trainer) != before:
                    raise AssertionError('Collection changed learner state')
                training = [g for g in rows if g['partition'] == 'training']
                heldout_games = [g for g in rows if g['partition'] == 'validation']
                norm = training_normalization(training, config['advantage_scale_floor'])
                trajectories = [t for g in training for t in g['trajectories']]
                if shutil.disk_usage(args.output).free < disk_floor * 1024**3:
                    raise RuntimeError('Insufficient disk for declared update caches')
                trainer.progress_callback = lambda event: progress(number, 'gpu_update', update=event)
                started = time.monotonic()
                update = await asyncio.to_thread(trainer.update_files, trajectories)
                update_seconds = time.monotonic() - started
                memory = [collection_memory, release('after_update')]
                before = training_state(model, trainer)
                with device_scope('cpu'):
                    validation_model.set_weights(model.get_weights())
                def validate():
                    with device_scope('cpu'):
                        return validator.evaluate(heldout_games, norm)
                progress(number, 'holdout_validation')
                started = time.monotonic()
                heldout = await asyncio.to_thread(validate)
                validation_seconds = time.monotonic() - started
                if training_state(model, trainer) != before:
                    raise AssertionError('Holdout validation changed learner state')
                memory.append(release('after_validation'))
            for array in [*model.get_weights(), *map(np.asarray, trainer.optimizer.variables),
                          *map(np.asarray, trainer.critic_optimizer.variables)]:
                if not np.isfinite(array).all():
                    raise FloatingPointError('Nonfinite model or optimizer state')
            games_path = args.output / 'rounds' / f'round-{number:04d}' / 'games.json'
            atomic_json(games_path, rows)
            row = dict(round=number, training_games=len(training), validation_games=len(heldout_games),
                games_file=str(games_path), games_sha256=sha256(games_path),
                behavior_version=number - 1, behavior_weights_sha256=behavior_digest,
                opponent_games=dict(Counter(str(g['opponent_version']) for g in rows)),
                behavior_reproduction=reproduction, collection_seconds=collection_seconds,
                collection_inference=collection_inference,
                update_seconds=update_seconds, validation_seconds=validation_seconds,
                update=update, validation=heldout, validation_read_only=True,
                resources=resources.result(), host_memory_releases=memory)
            assert_runtime_unchanged(pins)
            checkpoint = save_checkpoint(root, model, trainer,
                dict(round=number, config=config, round_result=row), policy_type=StandardPolicy)
            report['checkpoints'][str(number)] = checkpoint_hashes(checkpoint)
            report['rounds'].append(row)
            report['completed_games'] = number * 1000
            if 'fork' in config:
                report['lineage_checkpoint'] = config['fork']['parent_round'] + number
                report['total_exposure_games'] = config['fork']['parent_total_exposure_games'] + number * 1000
            atomic_json(report_path, report)
            atomic_json(directory / 'committed.json', dict(round=number, checkpoint=str(checkpoint)))
            cleanup_rollouts(args.output / 'rollouts', number)
            await evaluate(number)
        evaluation_games = validate_evaluation_completion(report, config)
        report.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(),
            completed_games=args.rounds * 1000,
            evaluation_games=evaluation_games)
        atomic_json(report_path, report)
        progress(args.rounds, 'complete', completed_games=report['completed_games'],
                 evaluation_games=report['evaluation_games'])
        return report
    except BaseException as error:
        if report is not None:
            report.update(status='failed', error=f'{type(error).__name__}: {error}')
            atomic_json(report_path, report)
        raise
    finally:
        try:
            await settle_all(c.close() for c in collectors)
        finally:
            tempfile.tempdir = old_tempdir
            lock.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--output', type=Path)
    result.add_argument('--preflight-only', action='store_true')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--architecture',choices=('compact','compact-identity','legacy'),default='compact',
        help='Compact stateless sparse actor/critic for fresh runs; legacy preserves the old recurrent recipe')
    result.add_argument('--compact-input-mib',type=int,default=512,help='Maximum encoded sparse rollout payload in MiB')
    result.add_argument('--initialize-from',type=Path,help='Pinned v1 policy for a fresh identity experiment')
    result.add_argument('--initialize-from-sha256',help='Required SHA256 of the initialization policy')
    result.add_argument('--fork-from', type=Path, help='Completed identity checkpoint directory; retain both optimizers and RNG')
    result.add_argument('--fork-from-sha256', help='SHA256 of the canonical checkpoint-file hash mapping')
    result.add_argument('--opponent-recipe', choices=('current', 'diverse'), help='Fixed 50-round continuation arm')
    result.add_argument('--prepare-fork-only', action='store_true', help='Prepare and pin local checkpoint zero without training')
    result.add_argument('--game-timeout',type=float,help='Hard per-game deadline in seconds (compact: 20; legacy: 60)')
    result.add_argument('--adopt-memory-fix-from', type=Path)
    result.add_argument('--adopt-collection-policy-from',type=Path)
    result.add_argument('--previous-guard', type=Path)
    result.add_argument('--rounds', type=int, default=100)
    result.add_argument('--games-per-round', type=int, default=1000)
    result.add_argument('--eval-games', type=int, default=40)
    result.add_argument('--eval-every-games', type=int, default=1000,
                        help='Run checkpoint duels at this collection-game interval (and at the final round)')
    result.add_argument('--skip-evaluations', action='store_true',
                        help='Disable head-to-head evaluations, including the final round; keep training holdouts')
    result.add_argument('--weight-seed', type=int, default=312610310)
    result.add_argument('--seed', type=int, default=412610310000)
    result.add_argument('--eval-seed', type=int, default=912610310000)
    result.add_argument('--workers', type=int, default=48)
    result.add_argument('--runners', type=int, default=2)
    result.add_argument('--inference-processes', type=int, default=2)
    result.add_argument('--opponent-inference-processes', type=int, default=2)
    result.add_argument('--policy-cache-mib', type=int, default=8192)
    result.add_argument('--actor-cache-mib', type=int, default=8192)
    result.add_argument('--update-timeout', type=float, default=1800)
    result.add_argument('--playback', choices=('direct', 'python'), default='direct')
    return result


if __name__ == '__main__':
    asyncio.run(run(parser().parse_args()))
