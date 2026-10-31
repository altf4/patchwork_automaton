"""Record balanced Standard expert games with a stratified whole-game holdout."""
import argparse
from contextlib import ExitStack
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations_with_replacement
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time

os.environ['PATCHWORK_OBSERVATION_PIPELINE'] = 'python'
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from patchwork_automaton.baselines.standard_decks import STANDARD_DECKS
from patchwork_automaton.neural.standard_features import StandardEncoder
from patchwork_automaton.rl.standard_imitation import timed_standard_game


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def schedule(games_per_pair, seed, *, index_base=0, validation_fraction=.2):
    if games_per_pair < 20 or games_per_pair % 20:
        raise ValueError('Games per pair must be a positive multiple of 20 for balanced 80/20 strata')
    if validation_fraction != .2:
        raise ValueError('This experiment requires a 20% validation split')
    for pairing_id, pair in enumerate(combinations_with_replacement(STANDARD_DECKS, 2)):
        heldout = set()
        for stratum in range(4):
            indices = list(range(stratum, games_per_pair, 4))
            random.Random(seed + pairing_id * 4 + stratum).shuffle(indices)
            heldout.update(indices[:len(indices) // 5])
        for attempt in range(games_per_pair):
            offset = pairing_id * games_per_pair + attempt
            yield dict(index=index_base + offset, pairing=list(pair),
                decks=list(pair if attempt % 2 == 0 else pair[::-1]),
                seed=seed + offset, starting_player=f'P{(attempt // 2) % 2}',
                partition='validation' if attempt in heldout else 'training')


def collect(args):
    games = list(schedule(args.games_per_pair, args.seed, index_base=args.index_base))
    args.output.mkdir(parents=True, exist_ok=args.resume)
    report_path = args.output / 'dataset.json'
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'}
    if args.workers < args.runners or args.runners < 1:
        raise ValueError('Require at least one worker per runner')
    if args.resume and report_path.exists():
        report = json.loads(report_path.read_text())
        if report['config'] != config or report['encoder'] != json.loads(json.dumps(StandardEncoder().manifest())):
            raise ValueError('Resume requires the same configuration and encoder')
    else:
        report = dict(status='collecting', config=config, encoder=StandardEncoder().manifest(),
            teacher='StandardAggro on both seats; full game including mulligans',
            rewards_used=False, neural_inference=False, games=[])
    completed = {g['index'] for g in report['games'] if g['status'] in ('win', 'loss', 'draw')}
    if len(completed) != len(report['games']):
        raise ValueError('Dataset contains failed games; investigate instead of replacing them')
    atomic_json(report_path, report)
    started = time.monotonic()
    with ExitStack() as stack:
        # Independent worker pools share only the deterministic schedule/report.
        pools = [stack.enter_context(ProcessPoolExecutor(
            max_workers=args.workers // args.runners + (i < args.workers % args.runners),
            mp_context=mp.get_context('spawn'))) for i in range(args.runners)]
        jobs = {pools[game['index'] % args.runners].submit(timed_standard_game, game, output=args.output / 'trajectories',
                diagnostics=args.output / 'adjudications', game_timeout=args.game_timeout, max_turns=args.max_turns): game
                for game in games if game['index'] not in completed}
        for job in as_completed(jobs):
            row = job.result()
            report['games'].append(row)
            if row['status'] not in ('win', 'loss', 'draw'):
                report['status'] = 'failed'
                atomic_json(report_path, report)
                for pending in jobs:
                    pending.cancel()
                raise RuntimeError(row.get('error', row))
            if len(report['games']) % 100 == 0:
                report['seconds_this_invocation'] = time.monotonic() - started
                report['games'].sort(key=lambda g: g['index'])
                atomic_json(report_path, report)
                print(json.dumps(dict(stage='collection', games=len(report['games']), target=len(games),
                    samples=sum(t['samples'] for g in report['games'] for t in g['trajectories']),
                    seconds=round(time.monotonic() - started, 2))), flush=True)
    report['games'].sort(key=lambda g: g['index'])
    if [g['index'] for g in report['games']] != [g['index'] for g in games]:
        raise AssertionError('Missing or duplicate demonstration games')
    report.update(status='complete', completed_games=len(games),
                  partitions=dict(Counter(g['partition'] for g in report['games'])),
                  outcomes=dict(Counter(g['status'] for g in report['games'])))
    atomic_json(report_path, report)
    print(json.dumps({k: report[k] for k in ('status', 'completed_games', 'partitions', 'outcomes')}), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--games-per-pair', type=int, default=2000)
    p.add_argument('--workers', type=int, default=24, help='Total game workers across runner pools')
    p.add_argument('--runners', type=int, default=2)
    p.add_argument('--seed', type=int, default=302610270000)
    p.add_argument('--index-base', type=int, default=0)
    p.add_argument('--game-timeout', type=float, default=60)
    p.add_argument('--max-turns', type=int, default=50)
    p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    collect(parser().parse_args())
