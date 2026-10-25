"""Collect disjoint whole-game expert demonstrations without loading TensorFlow."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

from patchwork_automaton.neural.validation import split_games
from patchwork_automaton.rl.rustmage_experiment import play_game


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def collect(args):
    if min(args.games, args.games_per_batch, args.workers) < 1 or args.games % args.games_per_batch:
        raise ValueError('Positive counts and complete game batches are required')
    if not 0 < args.validation_fraction < 1:
        raise ValueError('A nonempty whole-game holdout is required')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='collecting', config={k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()}, teacher='FoundationsAggro versus FoundationsAggro',
        rewards_used=False, neural_inference=False, batches=[], completed_games=0)
    started = time.monotonic()
    report_path = args.output / 'dataset.json'
    atomic_json(report_path, report)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        for batch in range(args.games // args.games_per_batch):
            offset = batch * args.games_per_batch
            jobs = [pool.submit(play_game, seed=args.seed + offset + i, index=offset + i,
                mode='demonstrations', output=args.output / 'trajectories') for i in range(args.games_per_batch)]
            games = []
            for job in as_completed(jobs):
                row = job.result()
                games.append(row)
                if row['status'] == 'error':
                    raise RuntimeError(row['error'])
            games.sort(key=lambda g: g['index'])
            _, _, split = split_games(games, fraction=args.validation_fraction,
                seed=args.seed, round_id=batch + 1)
            validation = set(split['validation_indices'])
            for game in games:
                game['partition'] = 'validation' if game['index'] in validation else 'training'
            report['batches'].append(dict(batch=batch + 1, games=games, split=split))
            report['completed_games'] += len(games)
            report['collection_seconds'] = time.monotonic() - started
            atomic_json(report_path, report)
            print(json.dumps(dict(stage='collect_expert', games=report['completed_games'],
                samples=split['training']['samples'] + split['validation']['samples'],
                seconds=round(report['collection_seconds'], 2))), flush=True)
    games = [g for b in report['batches'] for g in b['games']]
    report.update(status='complete', outcomes=dict(Counter(g['status'] for g in games)),
        collection_seconds=time.monotonic() - started,
        samples=sum(t['samples'] for g in games for t in g['trajectories']))
    library = Path('.local/rustmage/target/release/librustmage.so')
    report['library_sha256'] = hashlib.sha256(library.read_bytes()).hexdigest()
    atomic_json(report_path, report)
    print(json.dumps(dict(stage='collection_complete', games=len(games), samples=report['samples'],
        seconds=report['collection_seconds'], outcomes=report['outcomes'])), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--games', type=int, default=10000)
    p.add_argument('--games-per-batch', type=int, default=250)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--validation-fraction', type=float, default=.2)
    p.add_argument('--seed', type=int, default=721)
    return p


if __name__ == '__main__':
    collect(parser().parse_args())
