"""Actual frozen neural-versus-neural games; no training or rollout writers."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

from . import train_rustmage  # Set the CPU/FP32 environment before model imports.
from patchwork_automaton.neural.devices import configure_runtime, device_scope
from patchwork_automaton.neural.experiment import atomic_json
from patchwork_automaton.neural.inference_processes import weights_digest
from patchwork_automaton.neural.rustmage_evaluation import baseline_evaluation_config, _counts
from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
from patchwork_automaton.rl.rustmage_duels import DuelProcessCollector


def config_for(*, games, seed, horizon=60):
    # The same scheduler and limits as active-baseline confirmation, with both
    # seats neural. Keep real mode identity; never relabel baseline outcomes.
    base = baseline_evaluation_config(seed_base=seed, games=games,
        max_own_turns=horizon, greedy=False)
    collection = dict(base['collection'])
    collection.pop('mode')
    collection.pop('output')
    return dict(collection=collection, cohort=base['cohort'], mode='versus_neural',
        policy='sampled temperature 1', focal_seat='index % 2',
        starting_player='P((index // 2) % 2)',
        decks=["foundations_deck('red')", "foundations_deck('red')"])


def summarize_duels(rows, config):
    rows = list(rows)
    cohort = config['cohort']
    if len(rows) != cohort['attempted_games']:
        raise ValueError('Head-to-head evaluation must retain every scheduled attempt')
    for offset, row in enumerate(rows):
        index = cohort['index_base']+offset
        if (row.get('index'), row.get('seed'), row.get('seat'), row.get('starting_player')) != (
                index, cohort['seed_base']+offset, index % 2, f'P{(index//2)%2}'):
            raise ValueError('Head-to-head seed, seat, index or starting player differs')
        if row.get('mode') != 'versus_neural' or row.get('trajectories'):
            raise ValueError('Expected actual neural opponents without training trajectories')
        if row.get('status') not in ('win', 'loss', 'draw', 'cutoff', 'error'):
            raise ValueError('Unknown head-to-head outcome')
        if row['status'] in ('win', 'loss', 'draw'):
            expected = None if row['status'] == 'draw' else row['seat'] if row['status'] == 'win' else 1-row['seat']
            if row.get('winner') != expected:
                raise ValueError('Winner does not agree with the focal policy outcome')
    summary = _counts(rows)
    strata = {f'seat_{seat}_start_{starter}':_counts([row for row in rows
        if row['seat'] == seat and row['starting_player'] == starter])
        for seat in (0, 1) for starter in ('P0', 'P1')}
    summary.update(mode='versus_neural', metric='focal_recorded_wins_per_scheduled_attempt',
        win_rate_denominator='all_attempts_including_cutoffs_and_errors',
        unresolved_outcomes='unknown; no terminal score or training return assigned',
        seat_start_balanced=len({x['attempted_games'] for x in strata.values()}) == 1,
        seat_start_strata=strata)
    return summary


def file_hashes(checkpoints):
    files = [p for path in checkpoints for p in (path, path.with_suffix('.manifest.json'))]
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


async def run(args):
    config = config_for(games=args.games, seed=args.seed, horizon=args.horizon)
    if args.output.exists():
        raise FileExistsError(args.output)
    if type(args.workers) is not int or args.workers < 1:
        raise ValueError('Positive integer game workers required')
    if not 1 <= args.replicas_per_policy <= args.workers:
        raise ValueError('Replicas per policy must be between one and workers')
    before = file_hashes((args.checkpoint, args.opponent_checkpoint))
    configure_runtime('cpu', hide_gpu=True)
    with device_scope('cpu'):
        focal = RustMagePolicy.load(args.checkpoint).model
        opponent = RustMagePolicy.load(args.opponent_checkpoint).model
    digests = [weights_digest(model.get_weights()) for model in (focal, opponent)]
    collector = DuelProcessCollector(focal, opponent, workers=args.workers,
                                     replicas_per_policy=args.replicas_per_policy)
    started = time.monotonic()
    completed = 0
    def progress(row):
        nonlocal completed
        completed += 1
        if completed % 100 == 0 or completed == args.games:
            print(json.dumps(dict(completed=completed, games=args.games,
                status=row['status'], seconds=time.monotonic()-started)), flush=True)
    try:
        await collector.start()
        rows = await collector.run_games(**config['collection'], progress=progress)
        summary = summarize_duels(rows, config)
        batching = collector.stats()
        if any(state.get('visible_gpus') for role in ('focal','opponent')
               for state in batching[role]['replicas']):
            raise RuntimeError('Head-to-head inference must hide GPUs')
        if [batching[role]['weights_sha256'] for role in ('focal','opponent')] != digests:
            raise RuntimeError('Head-to-head inference changed or mixed frozen policy weights')
    finally:
        await collector.close()
    if file_hashes((args.checkpoint, args.opponent_checkpoint)) != before:
        raise RuntimeError('Frozen head-to-head checkpoint files changed')
    if [weights_digest(model.get_weights()) for model in (focal, opponent)] != digests:
        raise RuntimeError('Frozen head-to-head parent weights changed')
    result = dict(status='complete', config=config, checkpoint=str(args.checkpoint),
        checkpoint_sha256=before[str(args.checkpoint.resolve())],
        opponent_checkpoint=str(args.opponent_checkpoint),
        opponent_checkpoint_sha256=before[str(args.opponent_checkpoint.resolve())],
        source_files_sha256=before, frozen_weight_digests=digests,
        game_engines=args.workers, replicas_per_policy=args.replicas_per_policy,
        inference_device='cpu', batching=batching, seconds=time.monotonic()-started,
        training=False, summary=summary, games=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(summary), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--opponent-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--games', type=int, default=5000)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--workers', type=int, default=48)
    p.add_argument('--replicas-per-policy', type=int, default=2)
    p.add_argument('--horizon', type=int, default=60)
    return p


if __name__ == '__main__':
    asyncio.run(run(parser().parse_args()))
