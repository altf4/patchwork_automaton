"""Complete Foundations aggro games through the same API, policy, and CPU budget.

Both seats use FoundationsAggro. Rotate color/seat and starting player separately.
Interleave engines to reduce ordering bias. Failures are retained, never replaced.
No neural inference, training, network startup, forced concessions or short games.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

from examples.xmage_benchmark import affinity
from patchwork_automaton.backends.rustmage import OBSERVATION_SCHEMA, RUSTMAGE_REVISION
from patchwork_automaton.backends.xmage_prompts import prompt_metadata
from patchwork_automaton.baselines.foundations import DATA, FoundationsAggro, foundations_deck
from patchwork_automaton.core import Response
from patchwork_automaton.rl import EpisodeEnd, RustMageSelfPlay, XMageSelfPlay


def summary(rows):
    seconds = sum(r['seconds'] for r in rows)
    completed = [r for r in rows if r['status'] == 'completed']
    rng = random.Random(0)
    rates = sorted(60*sum(r['status'] == 'completed' for r in sample)/sum(r['seconds'] for r in sample)
                   for sample in (rng.choices(rows, k=len(rows)) for _ in range(2000))) if rows else []
    return dict(attempts=len(rows), completed=len(completed), failures=len(rows)-len(completed),
        seconds=seconds, games_per_minute=60*len(completed)/seconds if seconds else None,
        bootstrap_95_percent_gpm=[rates[50], rates[1949]] if rates else None,
        mean_game_seconds=statistics.mean(r['seconds'] for r in rows) if rows else None,
        mean_setup_seconds=statistics.mean(r['setup_seconds'] for r in rows) if rows else None,
        mean_play_seconds=statistics.mean(r['play_seconds'] for r in rows) if rows else None,
        mean_cleanup_seconds=statistics.mean(r['cleanup_seconds'] for r in rows) if rows else None,
        mean_decisions=statistics.mean(r['decisions'] for r in completed) if completed else None,
        mean_global_turns=statistics.mean(r['turn'] for r in completed) if completed else None,
        wins=dict(Counter(r['winner_color'] for r in completed)),
        decisions_per_second=sum(r['decisions'] for r in rows)/seconds if seconds else None)


async def episode(engine, index, args):
    colors = ('red', 'white') if index % 2 == 0 else ('white', 'red')
    starting_seat = (index // 2) % 2
    decks = tuple(foundations_deck(c) for c in colors)
    policy = FoundationsAggro()
    options = dict(timeout=args.timeout, max_decisions=args.limit)
    factory = (RustMageSelfPlay(*decks, checkout=args.rustmage, starting_player=f'P{starting_seat}', **options)
               if engine == 'rustmage' else XMageSelfPlay(*decks, deck_type='Constructed - Standard', **options))
    row = dict(engine=engine, index=index, colors=colors, starting_seat=starting_seat,
               seed=args.seed+index if engine == 'rustmage' else None)
    counts = Counter(); recent = []; start = time.perf_counter(); first = start; ended = start
    runner = None; first_active = None
    try:
        async with asyncio.timeout(args.game_timeout):
            runner = await factory.reset(seed=args.seed+index) if engine == 'rustmage' else await factory.reset()
            seat_actors = {s: (await g.snapshot()).actor for s, g in runner.games.items()}
            desired_start = seat_actors[f'player_{starting_seat}']
            turn = await runner.next(); first = time.perf_counter()
            while not isinstance(turn, EpisodeEnd):
                obs = turn.observation
                if obs.phase is not None and first_active is None:
                    first_active = obs.active_player
                    if first_active != desired_start: raise RuntimeError('Starting player mismatch')
                purpose = prompt_metadata(obs.decision.kind, obs.decision.native)['purpose'] if engine == 'xmage' else None
                if purpose == 'starting_player':
                    response = Response('object:' + desired_start.value)
                else: response = policy.choose(obs)
                option = next(o for o in obs.decision.options if o.id == response.option)
                counts[obs.decision.kind + ':' + option.label] += 1
                recent.append(dict(turn=obs.turn, actor=turn.actor, kind=obs.decision.kind,
                                   choice=option.label, response=response.option))
                recent = recent[-12:]
                await runner.submit(response)
                turn = await runner.next()
            ended = time.perf_counter()
            if turn.truncated:
                row.update(status='cutoff', reason=turn.reason, recent=recent)
            else:
                if len(turn.observations) != 2: raise RuntimeError('Missing terminal seat')
                outcomes = [turn.observations[f'player_{i}'].outcome for i in range(2)]
                if any(o is None for o in outcomes): raise RuntimeError('Missing result')
                winners = [i for i, o in enumerate(outcomes) if o.won is True]
                draws = all(o.draw is True for o in outcomes)
                if not draws and (len(winners) != 1 or outcomes[1-winners[0]].won is not False):
                    raise RuntimeError('Inconsistent terminal results')
                final = turn.observations['player_0']
                row.update(status='completed', winner_color=colors[winners[0]] if winners else 'draw',
                           winner_seat=winners[0] if winners else None, turn=final.turn,
                           life={s: next(p.life for p in final.players if p.id == a) for s, a in seat_actors.items()})
    except TimeoutError:
        ended = time.perf_counter(); row.update(status='cutoff', reason='wall_clock_limit', recent=recent)
    except Exception as error:
        ended = time.perf_counter(); row.update(status='error', reason=repr(error), recent=recent)
    finally:
        row['decisions'] = runner.steps if runner else 0
        row['callbacks'] = dict(counts)
        await factory.close()
        closed = time.perf_counter()
        row.update(seconds=closed-start, setup_seconds=first-start,
                   play_seconds=ended-first, cleanup_seconds=closed-ended)
    return row


async def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []; warmups = []
    engine_names = ('rustmage', 'xmage') if args.engine == 'both' else (args.engine,)
    library = args.rustmage / 'target/release/librustmage.so'
    if 'rustmage' in engine_names:
        revision = subprocess.check_output(['git', '-C', str(args.rustmage), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(args.rustmage), 'status', '--porcelain'], text=True).strip()
        if revision != RUSTMAGE_REVISION or dirty:
            raise RuntimeError('Comparison requires the clean pinned RustMage revision')
    sources = [Path(__file__), Path('src/patchwork_automaton/backends/rustmage.py'),
               Path('src/patchwork_automaton/baselines/foundations.py'),
               Path('src/patchwork_automaton/data/foundations-aggro.json'),
               Path('src/patchwork_automaton/rl/runner.py'), Path('src/patchwork_automaton/rl/rustmage.py'),
               Path('src/patchwork_automaton/backends/xmage.py'), args.rustmage/'python/rustmage.py']
    report = dict(schema=1, rustmage_revision=RUSTMAGE_REVISION, observation_schema=OBSERVATION_SCHEMA,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest() if library.exists() else None,
        decks=DATA, config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        machine=dict(platform=platform.platform(), processor=platform.processor(), logical_cpus=os.cpu_count()),
        methodology=['One sequential worker, same deterministic policy and exact 60-card FDN decks.',
            'Seat and starting-player independently rotated. Engines interleaved; warmups excluded.',
            'Full natural games; time includes reset/login/deck/table setup, policy, adapters, gameplay and cleanup.',
            'Excludes engine startup, library compilation, inference and training. Failures retained.',
            'RustMage seeded; stock XMage shuffles unseeded. Comparable workload, not identical trajectories.',
            'RustMage uses ctypes/JSON and common Python observation conversion; not a raw Rust rollout benchmark.',
            'Native decision granularity differs; decisions/second is diagnostic, not an equal-work metric.'])
    def persist():
        report.update(games=rows, warmups=warmups,
                      summary={e: summary([r for r in rows if r['engine'] == e]) for e in engine_names})
        tmp = args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(report, indent=2)+'\n'); tmp.replace(args.output)
    with affinity(args.one_cpu, args.output) as cpu:
        report['logical_cpu'] = cpu
        for index in range(args.warmup):
            for engine in engine_names:
                row = await episode(engine, index+100000, args); warmups.append(row)
                print(json.dumps(dict(warmup=index, engine=engine, status=row['status'], seconds=row['seconds'])), flush=True)
                persist()
                if row['status'] != 'completed': raise RuntimeError('Warmup failed: ' + repr(row))
        for index in range(args.games):
            order = engine_names if index % 2 == 0 else engine_names[::-1]
            for engine in order:
                row = await episode(engine, index, args); rows.append(row); persist()
                print(json.dumps(dict(index=index, engine=engine, status=row['status'], seconds=row['seconds'],
                                      turn=row.get('turn'), winner=row.get('winner_color'))), flush=True)
    report['affinity_restored'] = True
    if args.one_cpu:
        recovery = json.loads(args.output.with_suffix('.affinity.json').read_text())
        for pid, default in recovery['defaults'].items():
            for task in Path(f'/proc/{pid}/task').iterdir():
                try:
                    expected = recovery['threads'].get(task.name, default)
                    if sorted(os.sched_getaffinity(int(task.name))) != expected:
                        raise RuntimeError('CPU affinity was not restored for thread ' + task.name)
                except ProcessLookupError: pass
    report['affinity_restoration_verified'] = args.one_cpu
    persist()
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', choices=('both', 'rustmage', 'xmage'), default='both')
    p.add_argument('--games', type=int, default=40, help='Measured attempts per engine')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--seed', type=int, default=801)
    p.add_argument('--limit', type=int, default=3000)
    p.add_argument('--timeout', type=float, default=30)
    p.add_argument('--game-timeout', type=float, default=120)
    p.add_argument('--rustmage', type=Path, default=Path('.local/rustmage'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--one-cpu', action='store_true')
    args = p.parse_args()
    if min(args.games, args.limit, args.timeout, args.game_timeout) <= 0 or args.warmup < 0:
        p.error('Counts/timeouts must be positive and warmup nonnegative')
    asyncio.run(run(args))
