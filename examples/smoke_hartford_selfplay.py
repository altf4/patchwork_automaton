"""Bounded 400-game Hartford admission smoke using the production random policy.

Both neural seats sample checkpoint zero. This performs no optimizer updates;
its seed range and output are separate from the production experiment. A failed
smoke is retained intact; reruns require a different output directory.
"""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import random
import time

from examples.train_hartford_ppo import (
    Resources, assert_runtime_unchanged, atomic_json, checkpoint_hashes,
    configuration as training_configuration, initialize_random,
    parser as training_parser, settle_all, sha256, source_manifest, training_state)
from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS
from patchwork_automaton.rl.hartford_preflight import hartford_preflight, require_hartford_ready
from patchwork_automaton.rl.hartford_schedule import (
    MAX_GAME_SEED, evaluation_seed_bounds, training_seed_bounds)
from patchwork_automaton.rl.standard_ppo import validate_collection


def production_config(args):
    """Use the same initializer recipe as production, including its weight seed."""
    if args.production_manifest is not None:
        document = json.loads(args.production_manifest.read_text())
        config = document.get('config', document)
        if config.get('weight_seed') != args.weight_seed or config.get('playback') != args.playback:
            raise ValueError('Smoke weight seed/playback differ from the production manifest')
        if config.get('initialization') != 'random_actor_and_critic' or config.get('action_selection') != 'sampled':
            raise ValueError('Smoke requires the random sampled production experiment')
        return config
    defaults = training_parser().parse_args([
        '--weight-seed', str(args.weight_seed), '--playback', args.playback,'--architecture',args.architecture])
    return training_configuration(defaults)


def configuration(args, production):
    if type(args.games) is not int or args.games != 400:
        raise ValueError('Hartford smoke requires exactly 400 games: four in every ordered matchup')
    if args.runners < 1 or any(n < args.runners or n % args.runners for n in
            (args.workers, args.inference_processes, args.opponent_inference_processes)):
        raise ValueError('Worker and inference totals must divide evenly among runners')
    if max(args.inference_processes, args.opponent_inference_processes) > args.workers:
        raise ValueError('Inference replicas cannot exceed worker count')
    if type(args.seed) is not int or args.seed < 0 or args.seed + args.games - 1 > MAX_GAME_SEED:
        raise ValueError('Invalid smoke seed range')
    first, last = args.seed, args.seed + args.games - 1
    production_ranges = dict(
        training=training_seed_bounds(rounds=production['rounds'], seed=production['seed']),
        evaluation=evaluation_seed_bounds(rounds=production['rounds'],
            games_per_cell=production['eval_games_per_cell'], seed=production['eval_seed']))
    for name, (low, high) in production_ranges.items():
        if not (last < low or high < first):
            raise ValueError(f'Smoke seed range overlaps production {name}')
    return dict(games=400, games_per_matchup=4, seed=args.seed, seed_bounds=[first, last],
        production_seed_bounds=production_ranges, weight_seed=args.weight_seed,
        model_device=args.model_device, inference_device='cpu', workers=args.workers,
        runners=args.runners, inference_processes=args.inference_processes,
        opponent_inference_processes=args.opponent_inference_processes,
        game_timeout=production['game_timeout'], max_turns=production['max_turns'],
        max_decisions=production['max_decisions'], playback=args.playback,
        checkpoint_round=0, opponent_round=0, action_selection='sampled temperature 1',
        record_trajectories=True, optimize=False)


def schedule(deck_ids, *, seed, checkpoint_sha256):
    decks = tuple(deck_ids)
    if len(decks) != 10 or len(set(decks)) != 10:
        raise ValueError('Hartford smoke requires ten distinct deck IDs')
    if len(checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in checkpoint_sha256):
        raise ValueError('Checkpoint SHA256 is required')
    rows = []
    for cell, (learner, opponent) in enumerate(product(decks, repeat=2)):
        for repeat in range(4):
            index = cell * 4 + repeat
            seat = repeat % 2
            seating = [opponent, opponent]
            seating[seat] = learner
            rows.append(dict(index=index, seed=seed + index, deck_catalog='hartford',
                partition='smoke', pairing=sorted((learner, opponent), key=decks.index),
                decks=seating, learner_seat=seat, learner_deck=learner, opponent_deck=opponent,
                starting_player=f'P{(repeat // 2) % 2}', behavior_version=0,
                opponent_kind='frozen_neural', opponent_version=0,
                opponent_checkpoint_sha256=checkpoint_sha256))
    random.Random(f'hartford-smoke:{seed}').shuffle(rows)
    return rows


def summarize(rows):
    rows = list(rows)
    terminal = [row for row in rows if row.get('status') in ('win', 'loss', 'draw')]
    natural = [row for row in terminal if not row.get('adjudicated', False)]
    seconds = [row['seconds'] for row in rows if isinstance(row.get('seconds'), (int, float))
               and math.isfinite(row['seconds'])]
    return dict(completed_games=len(rows), terminal_games=len(terminal),
        natural_completed_games=len(natural),
        natural_completion_rate=len(natural) / len(rows) if rows else None,
        learner_wins=sum(bool(row.get('learner_won')) for row in terminal),
        learner_losses=sum(row.get('status') != 'draw' and not row.get('learner_won') for row in terminal),
        draws=sum(row.get('status') == 'draw' for row in terminal),
        natural_draws=sum(row.get('status') == 'draw' for row in natural),
        errors=sum(row.get('status') == 'error' for row in rows),
        cutoffs=sum(row.get('status') == 'cutoff' for row in rows),
        status_counts=dict(Counter(row.get('status', 'missing') for row in rows)),
        status_counts_basis='engine seat zero; learner_wins and learner_losses use focal seat',
        adjudications=dict(Counter(row.get('reason', 'unknown') for row in rows if row.get('adjudicated'))),
        game_seconds=dict(total=sum(seconds), maximum=max(seconds) if seconds else None,
                          mean=sum(seconds) / len(seconds) if seconds else None),
        decisions=sum(row.get('decisions', 0) for row in rows),
        matchups={deck: {opponent: dict(games=len(cell := [row for row in rows
            if row['learner_deck'] == deck and row['opponent_deck'] == opponent]),
            natural_completed_games=sum(row.get('status') in ('win', 'loss', 'draw')
                and not row.get('adjudicated', False) for row in cell),
            statuses=dict(Counter(row.get('status', 'missing') for row in cell)))
            for opponent in HARTFORD_DECKS} for deck in HARTFORD_DECKS})


def verify_behavior(model, rows, *, device):
    """Reproduce two recorded choices per game under the frozen random model."""
    import numpy as np
    from patchwork_automaton.neural.devices import device_scope
    from patchwork_automaton.neural.features import collate
    from patchwork_automaton.neural.batching import record_batch
    from patchwork_automaton.neural.rollout_records import read_trajectory
    checked = 0
    games_checked = 0
    maximum_log_probability_error = maximum_value_error = 0.
    for row in rows:
        game_checked = False
        for trajectory in row['trajectories']:
            stream = iter(read_trajectory(trajectory['path']))
            try:
                for _ in range(2):
                    sample = next(stream, None)
                    if sample is None:
                        break
                    with device_scope(device):
                        inputs=record_batch(model,[sample])
                        result = model(inputs, training=False)
                    logits = np.asarray(result['logits'])[0].astype(np.float64)
                    logp = logits - logits.max()
                    logp -= np.log(np.exp(logp).sum())
                    np.testing.assert_allclose(logp, sample.log_probs, rtol=5e-4, atol=2e-4)
                    value = float(np.asarray(result['value'])[0, 0])
                    np.testing.assert_allclose(value, sample.value, rtol=5e-4, atol=2e-4)
                    if not 0 <= sample.action < len(logp):
                        raise AssertionError('Recorded action is outside the legal candidate domain')
                    maximum_log_probability_error = max(maximum_log_probability_error,
                                                       float(np.max(abs(logp - sample.log_probs))))
                    maximum_value_error = max(maximum_value_error, abs(value - sample.value))
                    checked += 1
                    game_checked = True
            finally:
                stream.close()
        games_checked += game_checked
    if not checked:
        raise AssertionError('Smoke produced no recorded neural choices to verify')
    return dict(samples=checked, games=games_checked,
        maximum_log_probability_error=maximum_log_probability_error,
        maximum_value_error=maximum_value_error, relative_tolerance=5e-4, absolute_tolerance=2e-4)


async def run(args):
    production = production_config(args)
    config = configuration(args, production)
    readiness = hartford_preflight(playback=args.playback)
    if args.preflight_only:
        result = dict(readiness=readiness, smoke=config, production=production)
        print(json.dumps(result, indent=2), flush=True)
        return result
    require_hartford_ready(readiness)
    if args.output is None:
        raise ValueError('--output is required for a smoke run')
    if args.output.exists():
        raise FileExistsError('Smoke output already exists; failed attempts are never replaced')

    import os
    os.environ.update(PATCHWORK_STANDARD_PLAYBACK=args.playback, PATCHWORK_ROLLOUT_STORAGE='pickle',
                      PATCHWORK_OBSERVATION_PIPELINE='python')
    if os.environ.get('KERAS_BACKEND', 'tensorflow') != 'tensorflow':
        raise ValueError('Hartford smoke requires the TensorFlow Keras backend')
    for key, value in dict(KERAS_BACKEND='tensorflow', TF_ENABLE_ONEDNN_OPTS='0',
                          TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2',
                          OPENBLAS_NUM_THREADS='1', TF_CPP_MIN_LOG_LEVEL='3').items():
        os.environ.setdefault(key, value)
    import keras
    from patchwork_automaton.neural.batching import model_example_batch
    from patchwork_automaton.neural.devices import configure_runtime, device_scope, assert_model_device
    from patchwork_automaton.neural.experiment import save_checkpoint
    from patchwork_automaton.neural.inference_processes import weights_digest
    from patchwork_automaton.neural.standard_policy import StandardPolicy
    from patchwork_automaton.rl.standard_selfplay_processes import StandardSelfplayCollector

    configure_runtime(args.model_device)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    pins = dict(source=source_manifest(), readiness=readiness)
    manifest = dict(status='initializing', config=config, production_config=production, pins=pins,
                    started_at=datetime.now(timezone.utc).isoformat(), training_updates=0)
    atomic_json(args.output / 'manifest.json', manifest)
    collectors, rows = [], []
    resources = None
    started = time.monotonic()
    try:
        model, trainer = initialize_random(production, device=args.model_device)
        assert_model_device(model, args.model_device)
        before = training_state(model, trainer)
        focal_digest = weights_digest(model.get_weights())
        if args.checkpoint is None:
            checkpoint = save_checkpoint(args.output / 'checkpoints', model, trainer,
                dict(round=0, config=production), policy_type=StandardPolicy)
        else:
            checkpoint = args.checkpoint.resolve(strict=True)
            if checkpoint.is_file():
                checkpoint = checkpoint.parent
            state = json.loads((checkpoint / 'state.json').read_text())
            if state.get('round') != 0 or state.get('config') != production:
                raise ValueError('Existing checkpoint must be production checkpoint zero with the same configuration')
            if state.get('optimizer_steps') != 0 or state.get('critic_optimizer_steps') != 0:
                raise ValueError('Checkpoint zero must have zero actor and critic optimizer steps')
            with device_scope(args.model_device):
                saved = StandardPolicy.load(checkpoint / 'policy.keras').model
            if weights_digest(saved.get_weights()) != focal_digest:
                raise ValueError('Checkpoint zero differs from reproducible fresh initialization')
            del saved
        hashes = checkpoint_hashes(checkpoint)
        with device_scope('cpu'):
            opponent_model = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(model))
            opponent_model(model_example_batch(opponent_model), training=False)
            opponent_model.set_weights(model.get_weights())
        assert_model_device(opponent_model, 'cpu')
        if weights_digest(opponent_model.get_weights()) != focal_digest:
            raise AssertionError('Smoke seats have different checkpoint zero weights')
        planned = schedule(tuple(HARTFORD_DECKS), seed=args.seed, checkpoint_sha256=hashes['policy.keras'])
        by_id = {row['index']: row for row in planned}
        seen = set()
        atomic_json(args.output / 'schedule.json', planned)
        manifest.update(status='running', checkpoint=str(checkpoint), checkpoint_sha256=hashes,
                        checkpoint_weights_sha256=focal_digest)
        atomic_json(args.output / 'manifest.json', manifest)
        collectors = [StandardSelfplayCollector(model, opponent_model,
            workers=args.workers // args.runners, inference_processes=args.inference_processes // args.runners,
            opponent_inference_processes=args.opponent_inference_processes // args.runners)
            for _ in range(args.runners)]
        await settle_all(collector.start(version=0, opponent_version=0) for collector in collectors)
        if any(collector.policy_digest != focal_digest or collector.opponent_digest != focal_digest
               for collector in collectors):
            raise AssertionError('Smoke inference replicas have stale weights')

        def persist(row):
            index = row.get('index')
            if type(index) is not int or index not in by_id or index in seen:
                raise ValueError('Foreign or duplicate smoke result')
            row = dict(row, actions=None)
            for trajectory in row.get('trajectories', ()):
                trajectory['sha256'] = sha256(trajectory['path'])
            atomic_json(args.output / 'results' / f'game-{index:04d}.json', row)
            seen.add(index)
            rows.append(row)
            if any(row.get(key) != value for key, value in by_id[index].items()):
                raise ValueError('Smoke result differs from the frozen schedule')
            if len(rows) % 20 == 0 or len(rows) == len(planned):
                event = dict(phase='smoke', completed=len(rows), total=len(planned))
                atomic_json(args.output / 'progress.json', event)
                print(json.dumps(event), flush=True)

        jobs = [dict(seed=game['seed'], index=game['index'], game=game) for game in planned]
        with Resources(args.output) as resources:
            await settle_all(collector.run_schedule(jobs[i::len(collectors)], progress=persist,
                game_runner='standard_ppo', output=args.output / 'trajectories',
                diagnostics=args.output / 'adjudications', version=0, opponent_version=0,
                opponent_sha256=hashes['policy.keras'], greedy=False,
                game_timeout=config['game_timeout'], max_turns=config['max_turns'],
                max_decisions=config['max_decisions']) for i, collector in enumerate(collectors))
        ordered = sorted(rows, key=lambda row: row['index'])
        atomic_json(args.output / 'summary.json', summarize(ordered))
        validate_collection(ordered, planned, 0)
        behavior = verify_behavior(model, ordered, device=args.model_device)
        if training_state(model, trainer) != before or weights_digest(opponent_model.get_weights()) != focal_digest:
            raise AssertionError('Smoke changed model, optimizer, or trainer RNG state')
        if checkpoint_hashes(checkpoint) != hashes:
            raise AssertionError('Checkpoint zero changed during smoke')
        assert_runtime_unchanged(pins)
        manifest.update(status='complete', behavior_verification=behavior,
            summary=summarize(ordered), resources=resources.result(),
            elapsed_seconds=time.monotonic()-started, inference=[collector.stats() for collector in collectors],
            finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(args.output / 'manifest.json', manifest)
        atomic_json(args.output / 'progress.json', dict(phase='complete', completed=len(rows), total=400))
        return manifest
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}',
            summary=summarize(rows), elapsed_seconds=time.monotonic()-started,
            resources=resources.result() if resources is not None else None,
            finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(args.output / 'manifest.json', manifest)
        atomic_json(args.output / 'summary.json', manifest['summary'])
        raise
    finally:
        await settle_all(collector.close() for collector in collectors)


def parser():
    defaults = training_parser().parse_args([])
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--output', type=Path)
    result.add_argument('--preflight-only', action='store_true')
    result.add_argument('--architecture',choices=('compact','legacy'),default=defaults.architecture)
    result.add_argument('--checkpoint', type=Path, help='Optional existing production checkpoint-zero directory')
    result.add_argument('--production-manifest', type=Path, help='Optional production manifest/config JSON')
    result.add_argument('--games', type=int, default=400)
    result.add_argument('--seed', type=int, default=1412610310000)
    result.add_argument('--weight-seed', type=int, default=defaults.weight_seed)
    result.add_argument('--model-device', choices=('cpu', 'gpu'), default='gpu')
    result.add_argument('--workers', type=int, default=48)
    result.add_argument('--runners', type=int, default=2)
    result.add_argument('--inference-processes', type=int, default=2)
    result.add_argument('--opponent-inference-processes', type=int, default=2)
    result.add_argument('--playback', choices=('direct', 'python'), default=defaults.playback)
    return result


if __name__ == '__main__':
    asyncio.run(run(parser().parse_args()))
