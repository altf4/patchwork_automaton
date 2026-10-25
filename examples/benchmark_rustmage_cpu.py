"""Matched single-core complete-game benchmark, without neural inference.

Uses the normal Python adapter and FoundationsAggro policy around the same
release RustMage library. Natural games, reset, choices, and cleanup are timed.
Compilation, imports, metadata hashing, and explicit warmups are excluded.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def choose_cpu(available):
    def score(cpu):
        path = Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq')
        return (int(path.read_text()) if path.exists() else 0, -cpu)
    return max(available, key=score)


async def game(index, seed, checkout):
    from patchwork_automaton.baselines.foundations import FoundationsAggro, foundations_deck
    from patchwork_automaton.rl import EpisodeEnd, RustMageSelfPlay
    colors = ('red', 'white') if index % 2 == 0 else ('white', 'red')
    start = f'P{(index // 2) % 2}'
    policy = FoundationsAggro()
    factory = RustMageSelfPlay(*(foundations_deck(c) for c in colors),
        checkout=checkout, starting_player=start, max_decisions=6000, timeout=30)
    row = dict(index=index, seed=seed, colors=colors, starting_player=start)
    trace = hashlib.sha256()
    started = time.perf_counter()
    runner = None
    try:
        runner = await factory.reset(seed=seed)
        turn = await runner.next()
        while not isinstance(turn, EpisodeEnd):
            obs = turn.observation
            response = policy.choose(obs)
            trace.update(json.dumps([obs.actor.value, obs.turn, obs.decision.kind,
                response.option, response.value], sort_keys=True, separators=(',', ':')).encode())
            await runner.submit(response)
            turn = await runner.next()
        if turn.truncated:
            raise RuntimeError('Game truncated: ' + str(turn.reason))
        outcomes = [turn.observations[f'player_{i}'].outcome for i in range(2)]
        if any(o is None for o in outcomes):
            raise RuntimeError('Missing terminal result')
        winners = [i for i, outcome in enumerate(outcomes) if outcome.won is True]
        draw = all(outcome.draw is True for outcome in outcomes)
        if not draw and (len(winners) != 1 or outcomes[1-winners[0]].won is not False):
            raise RuntimeError('Inconsistent terminal result')
        row.update(status='completed', winner=winners[0] if winners else None,
            turn=turn.observations['player_0'].turn)
    except Exception as error:
        row.update(status='error', error=repr(error))
    finally:
        row['decisions'] = runner.steps if runner else 0
        await factory.close()
        row.update(seconds=time.perf_counter()-started, trajectory_sha256=trace.hexdigest())
    return row


async def run(args):
    from patchwork_automaton.backends.rustmage import OBSERVATION_SCHEMA, RUSTMAGE_REVISION
    root = Path(__file__).resolve().parents[1]
    library = args.rustmage / 'target/release/librustmage.so'
    sources = [Path(__file__).resolve(),
        root/'src/patchwork_automaton/backends/rustmage.py',
        root/'src/patchwork_automaton/baselines/foundations.py',
        root/'src/patchwork_automaton/data/foundations-aggro.json',
        root/'src/patchwork_automaton/rl/rustmage.py',
        root/'src/patchwork_automaton/rl/runner.py']
    relevant = {str(p.relative_to(root)): digest(p) for p in sources}
    report = dict(schema=1, status='running', methodology=__doc__,
        config=dict(games=args.games, repeats=args.repeats, warmup=args.warmup,
                    seed=args.seed, cpu=args.cpu),
        host=dict(label=args.label, platform=platform.platform(), python=platform.python_version(),
            processor=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines()
                            if x.startswith('model name')), platform.processor()),
            logical_cpus=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0))),
        rustmage_revision=RUSTMAGE_REVISION, observation_schema=OBSERVATION_SCHEMA,
        library_sha256=digest(library), binding_sha256=digest(args.rustmage/'python/rustmage.py'),
        source_sha256=relevant, warmups=[], trials=[])
    atomic(args.output, report)
    for index in range(args.warmup):
        row = await game(index, args.seed + 1_000_000 + index, args.rustmage)
        report['warmups'].append(row)
        if row['status'] != 'completed':
            report['status'] = 'failed'; atomic(args.output, report)
            raise RuntimeError('Warmup failed: ' + repr(row))
    for trial in range(args.repeats):
        started, cpu = time.perf_counter(), time.process_time()
        rows = [await game(index, args.seed+index, args.rustmage) for index in range(args.games)]
        seconds, cpu_seconds = time.perf_counter()-started, time.process_time()-cpu
        valid = all(row['status'] == 'completed' for row in rows)
        records = [{k: v for k, v in row.items() if k != 'seconds'} for row in rows]
        result = dict(trial=trial, seconds=seconds, cpu_seconds=cpu_seconds,
            mean_cpu_cores=cpu_seconds/seconds, games=rows,
            completed=sum(row['status'] == 'completed' for row in rows),
            decisions=sum(row['decisions'] for row in rows),
            games_per_minute=60*args.games/seconds if valid else None,
            workload_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest())
        report['trials'].append(result)
        atomic(args.output, report)
        print(json.dumps({k:v for k,v in result.items() if k != 'games'}), flush=True)
        if not valid:
            report['status'] = 'failed'; atomic(args.output, report)
            raise RuntimeError('A measured game failed; attempts were retained, not replaced')
    if len({t['workload_sha256'] for t in report['trials']}) != 1:
        raise RuntimeError('Repeated game trajectories differed')
    if digest(library) != report['library_sha256'] or any(digest(root/p) != sha for p,sha in relevant.items()):
        raise RuntimeError('Execution inputs changed while the benchmark was running')
    report.update(status='complete', median_seconds=statistics.median(t['seconds'] for t in report['trials']),
        median_games_per_minute=statistics.median(t['games_per_minute'] for t in report['trials']))
    atomic(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--rustmage', type=Path, default=Path('.local/rustmage'))
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args()
    if min(args.games,args.repeats) < 1 or args.warmup < 0:
        parser.error('Positive games/repeats and nonnegative warmup required')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    available = os.sched_getaffinity(0)
    args.cpu = args.cpu if args.cpu is not None else choose_cpu(available)
    if args.cpu not in available: raise ValueError('Requested CPU is unavailable')
    os.sched_setaffinity(0, {args.cpu})
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PATCHWORK_OBSERVATION_PIPELINE='python', CUDA_VISIBLE_DEVICES='-1')
    asyncio.run(run(args))


if __name__ == '__main__': main()
