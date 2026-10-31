"""One shared Standard policy checkpoint, evaluated while piloting all three decks."""
import argparse
import asyncio
from copy import deepcopy
import hashlib
from itertools import combinations_with_replacement
import json
import math
import os
from pathlib import Path
import random
import time

os.environ['PATCHWORK_OBSERVATION_PIPELINE'] = 'python'
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from patchwork_automaton.baselines.standard_decks import STANDARD_DECKS


def audit_datasets(paths):
    """Check actual whole games and both seats before selecting a deck's records."""
    from patchwork_automaton.neural.standard_features import StandardEncoder
    rows, seen, seeds, trajectories, fingerprints = [], set(), set(), set(), []
    expected = StandardEncoder().manifest()
    for directory in paths:
        path = directory / 'dataset.json'
        fingerprints.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        data = json.loads(path.read_text())
        if data['status'] != 'complete' or data['encoder'] != json.loads(json.dumps(expected)):
            raise ValueError('Incomplete or incompatible expert dataset')
        if len(data['games']) != data['completed_games']:
            raise ValueError('Dataset game count mismatch')
        for game in data['games']:
            if game['index'] in seen or game['seed'] in seeds:
                raise ValueError('Duplicated game or seed across collection stages')
            if game['partition'] not in ('training', 'validation') or game['status'] not in ('win', 'loss', 'draw'):
                raise ValueError('Invalid partition or incomplete expert game')
            if not game.get('adjudicated') and (len(game['trajectories']) != 2 or {t['player'] for t in game['trajectories']} != {0, 1}):
                raise ValueError('Both expert seat trajectories are required')
            for t in game['trajectories']:
                if t['path'] in trajectories or 'reward' in t or not Path(t['path']).is_file():
                    raise ValueError('Duplicated, reward-labeled or missing demonstration')
                if t['deck'] != game['decks'][t['player']]:
                    raise ValueError('Deck and player mismatch')
                trajectories.add(t['path'])
            seen.add(game['index'])
            seeds.add(game['seed'])
            rows.append(game)
    return rows, fingerprints


def deck_games(games, deck_id, partition):
    selected = []
    for game in games:
        trajectories = [t for t in game['trajectories'] if t['deck'] == deck_id]
        if trajectories and game['partition'] == partition:
            selected.append(dict(game, trajectories=trajectories))
    return selected


def partition_games(games, partition):
    """Keep both seats and every deck; the shared optimizer sees one pooled set."""
    return [game for game in games if game['partition'] == partition and game['trajectories']]


def wilson(wins, attempts):
    if not attempts:
        return None
    z = 1.959963984540054
    p = wins / attempts
    denominator = 1 + z * z / attempts
    center = (p + z * z / (2 * attempts)) / denominator
    radius = z * math.sqrt(p * (1 - p) / attempts + z * z / (4 * attempts * attempts)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def evaluation_summary(rows):
    attempts = len(rows)
    completed = sum(r['status'] in ('win', 'loss', 'draw') for r in rows)
    wins = sum(r.get('learner_won', False) for r in rows)
    return dict(attempted=attempts, completed=completed, wins=wins,
        losses=sum(r['status'] in ('win', 'loss') and not r.get('learner_won') for r in rows),
        draws=sum(r['status'] == 'draw' for r in rows),
        errors=sum(r['status'] == 'error' for r in rows), cutoffs=sum(r['status'] == 'cutoff' for r in rows),
        adjudicated_draws=sum(r.get('adjudicated', False) for r in rows),
        score_rate=(wins + .5 * sum(r['status'] == 'draw' for r in rows)) / attempts if attempts else None,
        win_rate=wins / attempts if attempts else None, wilson_95=wilson(wins, attempts),
        denominator='all scheduled attempts; errors/cutoffs remain explicit unresolved attempts')


async def evaluate(model, args, stage, report, report_path):
    from patchwork_automaton.neural.experiment import atomic_json
    from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector

    # The critic is unused in imitation. Actor-only inference has exactly the
    # same logits and actor memory as the saved actor/critic container.
    actor = model.actor
    historical_path = Path('docs/benchmarks/standard-baselines-2026-10-27/matchup-stats.json')
    historical = json.loads(historical_path.read_text()) if historical_path.exists() else None
    pair_ids = {pair: i for i, pair in enumerate(combinations_with_replacement(STANDARD_DECKS, 2))}
    collector = ProcessGameCollector(actor, workers=args.eval_workers,
        inference_processes=args.inference_processes, inference_device='cpu')
    await collector.start()
    report.setdefault('evaluations', [])
    from patchwork_automaton.neural.inference_processes import weights_digest
    frozen_actor = weights_digest(actor.get_weights())
    if any(cell.get('policy_weights_sha256') != frozen_actor for cell in report['evaluations']):
        raise ValueError('Evaluation resume would combine different policies')
    report['evaluated_policy_sha256'] = frozen_actor
    try:
        for learner_deck in STANDARD_DECKS:
            for greedy in (False, True):
                for opponent in STANDARD_DECKS:
                    if any(e['deck'] == learner_deck and e['greedy'] == greedy and e['opponent'] == opponent for e in report['evaluations']):
                        continue
                    order = tuple(STANDARD_DECKS)
                    pair = tuple(sorted((learner_deck, opponent), key=order.index))
                    completed = 0
                    def progress(row):
                        nonlocal completed
                        completed += 1
                        if completed % 100 == 0:
                            print(json.dumps(dict(stage='evaluation_progress', checkpoint=stage, deck=learner_deck,
                                opponent=opponent, greedy=greedy, completed=completed, target=args.eval_games)), flush=True)
                    games = await collector.run_games(games=args.eval_games,
                        seed_base=args.eval_seed + pair_ids[pair] * 10000, game_runner='standard',
                        learner_deck=learner_deck, opponent_deck=opponent, greedy=greedy,
                        diagnostics=args.output / 'adjudications' / f'stage-{stage}' / f'{learner_deck}-vs-{opponent}-{greedy}',
                        game_timeout=args.game_timeout, max_turns=args.max_turns, progress=progress)
                    if weights_digest(actor.get_weights()) != frozen_actor:
                        raise AssertionError('Shared policy changed during evaluation')
                    games.sort(key=lambda g: g['index'])
                    result = dict(deck=learner_deck, policy_weights_sha256=frozen_actor, opponent=opponent, greedy=greedy, summary=evaluation_summary(games), games=games)
                    reference = .5 if opponent == learner_deck else None
                    if historical is not None and opponent != learner_deck:
                        match = next(m for m in historical['matchups'] if set(m['pairing']) == set(pair))
                        reference = match['perspectives'][learner_deck]['overall']['win_rate']
                        result['historical_baseline'] = match['perspectives'][learner_deck]['overall']
                    result['baseline_reference_win_rate'] = reference
                    result['delta_percentage_points'] = None if reference is None else 100 * (result['summary']['win_rate'] - reference)
                    report['evaluations'].append(result)
                    atomic_json(report_path, report)
                    print(json.dumps(dict(stage='evaluation', checkpoint=stage, deck=learner_deck,
                        opponent=opponent, greedy=greedy, summary=result['summary'],
                        delta_percentage_points=result['delta_percentage_points'])), flush=True)
        expected = {(deck, opponent, greedy) for deck in STANDARD_DECKS
                    for opponent in STANDARD_DECKS for greedy in (False, True)}
        actual = {(e['deck'], e['opponent'], e['greedy']) for e in report['evaluations']}
        if actual != expected or len(report['evaluations']) != len(expected):
            raise AssertionError('Incomplete or duplicated shared-policy evaluation')
        report['one_shared_policy_verified'] = all(
            e['policy_weights_sha256'] == frozen_actor for e in report['evaluations'])
    finally:
        await collector.close()


async def train(args):
    import keras
    import numpy as np
    import tensorflow as tf
    from patchwork_automaton.neural.batching import example_batch
    from patchwork_automaton.neural.experiment import atomic_json, load_checkpoint, make_trainer, save_checkpoint
    from patchwork_automaton.neural.inference_processes import weights_digest
    from patchwork_automaton.neural.model import SeparateActorCritic
    from patchwork_automaton.neural.standard_features import StandardEncoder
    from patchwork_automaton.neural.standard_policy import StandardPolicy
    from patchwork_automaton.neural.devices import configure_runtime, device_scope, assert_model_device
    from patchwork_automaton.neural.recurrent import configure_gru_backend, gru_backend_summary

    if args.eval_games < 4 or args.eval_games % 4:
        raise ValueError('Evaluation games must balance all four seat/start strata')
    configure_runtime(args.learner_device, hide_gpu=args.learner_device == 'cpu')
    backend = ('cudnn-memory' if args.learner_device == 'gpu' else 'unfused')
    if args.learner_gru_backend != 'default': backend = args.learner_gru_backend
    if backend.startswith('cudnn') and args.learner_device != 'gpu':
        raise ValueError('cuDNN requires GPU training')
    games, fingerprints = audit_datasets(args.datasets)
    current = json.loads((args.datasets[-1] / 'dataset.json').read_text())['games']
    training = partition_games(current, 'training')
    validation = partition_games(games, 'validation')
    if not training or not validation:
        raise ValueError('Empty training/validation partition')
    encoder = StandardEncoder()
    config = dict(algorithm='imitation', seed=args.seed, policy_scope='shared',
        training_decks=list(STANDARD_DECKS), learner_device=args.learner_device, learning_rate=args.learning_rate, epochs=args.epochs,
        train_batch=args.train_batch, unroll=args.unroll, action_kinds=list(encoder.actions))
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.output / 'checkpoints'
    report_path = args.output / f'stage-{args.stage}.json'
    if report_path.exists() and json.loads(report_path.read_text()).get('status') == 'complete':
        completed = json.loads(report_path.read_text())
        if completed['config'] != config or completed['dataset_sha256'] != fingerprints:
            raise ValueError('Completed checkpoint configuration/dataset mismatch')
        print(json.dumps(dict(stage='already_complete', deck='shared', checkpoint=args.stage)), flush=True)
        return
    if (checkpoint_root / 'latest.json').exists():
        with device_scope(args.learner_device):
            model, trainer, state = load_checkpoint(checkpoint_root, policy_type=StandardPolicy, gru_backend=backend)
        if state['config'] != config or state['round'] not in (args.stage - 1, args.stage):
            raise ValueError('Checkpoint configuration/stage mismatch')
        resume_evaluation = state['round'] == args.stage
    else:
        if args.stage != 1:
            raise ValueError('Missing preceding stage checkpoint')
        keras.utils.set_random_seed(config['seed'])
        with device_scope(args.learner_device):
            model = SeparateActorCritic(encoder.card_count, encoder.feature_spec)
            model(example_batch(model.memory_size, model.feature_spec), training=False)
            configure_gru_backend(model, backend)
            trainer = make_trainer(model, config)
        save_checkpoint(checkpoint_root, model, trainer, dict(round=0, config=config), policy_type=StandardPolicy)
        resume_evaluation = False
    placements = assert_model_device(model, args.learner_device)
    print(json.dumps(dict(stage='learner_device', device=args.learner_device,
        variable_devices=sorted(set(placements.values())), gru_layers=gru_backend_summary(model))), flush=True)
    consumed = 0
    recovered_report = None
    recovery = args.output / 'recovery'
    if not resume_evaluation and (recovery / 'latest.json').exists():
        with device_scope(args.learner_device):
            recovered_model, recovered_trainer, recovered_state = load_checkpoint(
                recovery, policy_type=StandardPolicy, gru_backend=backend)
        if recovered_state['stage'] == args.stage:
            if recovered_state['config'] != config or recovered_state['dataset_sha256'] != fingerprints:
                raise ValueError('Recovery configuration/dataset mismatch')
            model, trainer = recovered_model, recovered_trainer
            consumed = recovered_state['games_consumed']
            recovered_report = recovered_state['round_result']
            assert_model_device(model, args.learner_device)
    if resume_evaluation:
        report = json.loads(report_path.read_text()) if report_path.exists() else state['round_result']
    else:
        report = dict(status='training', deck='shared', checkpoint=args.stage, config=config,
            cumulative_games=len(games), training_games=sum(g['partition'] == 'training' for g in games),
            validation_games=sum(g['partition'] == 'validation' for g in games),
            new_training_seat_trajectories=sum(len(g['trajectories']) for g in training),
            cumulative_validation_seat_trajectories=sum(len(g['trajectories']) for g in validation),
            policy_scope='shared', training_decks=list(STANDARD_DECKS),
            new_training_seats_by_deck={deck: sum(t['deck'] == deck for g in training for t in g['trajectories']) for deck in STANDARD_DECKS},
            checkpoint_path=str((checkpoint_root / f'round-{args.stage:04d}').resolve()),
            dataset_sha256=fingerprints, encoder=encoder.manifest(), device=args.learner_device,
            learner_gru_backend=backend, variable_devices=sorted(set(placements.values())), updates=[])
        if recovered_report is not None: report = recovered_report
        report['new_training_games_target'] = len(training)
        atomic_json(report_path, report)
        frozen_critic = weights_digest(model.critic.get_weights())
        ordered = training.copy()
        random.Random(config['seed'] + args.stage).shuffle(ordered)
        started = time.monotonic()
        prior_seconds = report.get('training_seconds', 0.)
        for start in range(consumed, len(ordered), args.chunk_games):
            fit = await asyncio.to_thread(trainer.fit, ordered[start:start + args.chunk_games])
            report['updates'].append(fit)
            report['new_training_games_consumed'] = min(start + args.chunk_games, len(ordered))
            report['training_seconds'] = prior_seconds + time.monotonic() - started
            save_checkpoint(recovery, model, trainer, dict(
                round=args.stage * 100000 + report['new_training_games_consumed'],
                stage=args.stage, games_consumed=report['new_training_games_consumed'],
                config=config, dataset_sha256=fingerprints, round_result=report), policy_type=StandardPolicy)
            atomic_json(report_path, report)
            print(json.dumps(dict(stage='imitation_fit', checkpoint=args.stage, deck='shared',
                games_consumed=report['new_training_games_consumed'], target=len(ordered),
                optimizer_steps=fit['optimizer_steps'], seconds=round(report['training_seconds'], 2),
                metrics={k: fit['epochs'][-1][k] for k in ('cross_entropy', 'accuracy', 'non_pass_accuracy')})), flush=True)
        weights_before = weights_digest(model.get_weights())
        optimizer_before = weights_digest([np.asarray(v) for v in trainer.optimizer.variables])
        rng_before = deepcopy(trainer.rng.bit_generator.state)
        report['validation'] = await asyncio.to_thread(trainer.evaluate, validation)
        if (weights_digest(model.get_weights()) != weights_before
                or weights_digest([np.asarray(v) for v in trainer.optimizer.variables]) != optimizer_before
                or trainer.rng.bit_generator.state != rng_before):
            raise AssertionError('Validation changed weights, optimizer or training sampler')
        if weights_digest(model.critic.get_weights()) != frozen_critic:
            raise AssertionError('Imitation updated the critic')
        report.update(status='evaluating', validation_read_only=True, critic_unchanged=True)
        atomic_json(report_path, report)
        save_checkpoint(checkpoint_root, model, trainer,
            dict(round=args.stage, config=config, round_result=report), policy_type=StandardPolicy)
    await evaluate(model, args, args.stage, report, report_path)
    for saved in fingerprints:
        if hashlib.sha256(Path(saved['path']).read_bytes()).hexdigest() != saved['sha256']:
            raise AssertionError('Expert dataset changed during training/evaluation')
    report.update(status='complete', datasets_unchanged=True, validation_optimizer_updates=0, rl_updates=0)
    atomic_json(report_path, report)
    print(json.dumps(dict(stage='complete', checkpoint=args.stage, deck='shared',
        games_seen=len(games), report=str(report_path))), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', type=int, required=True)
    p.add_argument('--learner-device', choices=('cpu','gpu'), default='gpu')
    p.add_argument('--learner-gru-backend', choices=('default','unfused','cudnn-memory'), default='default')
    p.add_argument('--inference-processes', type=int, default=2)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--train-batch', type=int, default=32)
    p.add_argument('--unroll', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=.0003)
    p.add_argument('--chunk-games', type=int, default=400)
    p.add_argument('--eval-games', type=int, default=400, help='Per opposing deck, for each of sampled and greedy play')
    p.add_argument('--eval-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=30261027)
    p.add_argument('--game-timeout', type=float, default=60)
    p.add_argument('--max-turns', type=int, default=50)
    p.add_argument('--eval-seed', type=int, default=502610270000)
    return p


if __name__ == '__main__':
    asyncio.run(train(parser().parse_args()))
