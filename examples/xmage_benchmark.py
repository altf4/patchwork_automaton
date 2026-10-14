"""Sequential 100-decision XMage benchmark; no learner or MTGO connections.

Run with [xmage] installed and the local server already started:
  PYTHONPATH=src:. python -m examples.xmage_benchmark --games 10 --warmup 2
--one-cpu temporarily pins this process and the managed JVMs to one logical CPU.
"""
import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

from examples.xmage_selfplay import smoke_response
from patchwork_automaton.core import CardSpec,DeckSpec
from patchwork_automaton.rl import EpisodeEnd,XMageSelfPlay

ROOT=Path(__file__).resolve().parents[1]


def summary(values):
    ordered=sorted(values)
    return {'mean':statistics.mean(values),'median':statistics.median(values),
            'p95':ordered[math.ceil(.95*len(ordered))-1],
            'min':ordered[0],'max':ordered[-1]}


@contextmanager
def affinity(enabled,output):
    if not enabled:
        yield None
        return
    pids=json.loads((ROOT/'.local/xmage/pids.json').read_text())
    # Identity check: never alter an unrelated reused PID.
    for name,pid in pids.items():
        command=Path(f'/proc/{pid}/cmdline').read_bytes()
        if str(ROOT/'.local/xmage/java/bin/java').encode() not in command:
            raise RuntimeError('Managed JVM identity check failed: '+name)
    defaults={pid:os.sched_getaffinity(pid) for pid in (*pids.values(),os.getpid())}
    cpu=max(set.intersection(*defaults.values()))
    saved={}
    def tasks():
        return [int(p.name) for pid in defaults for p in Path(f'/proc/{pid}/task').iterdir()]
    try:
        # Pin each extant JVM thread; newly created threads inherit this affinity.
        for tid in tasks():
            try:
                saved[tid]=os.sched_getaffinity(tid)
                os.sched_setaffinity(tid,{cpu})
            except ProcessLookupError:pass
        recovery=output.with_suffix('.affinity.json')
        recovery.write_text(json.dumps({'defaults':{str(k):sorted(v) for k,v in defaults.items()},
                                       'threads':{str(k):sorted(v) for k,v in saved.items()}},indent=2))
        yield cpu
    finally:
        for pid,default in defaults.items():
            try:threads=list(Path(f'/proc/{pid}/task').iterdir())
            except FileNotFoundError:continue
            for thread in threads:
                tid=int(thread.name)
                try:os.sched_setaffinity(tid,saved.get(tid,default))
                except ProcessLookupError:pass


async def episode(factory,number,actions):
    start=time.perf_counter()
    runner=await factory.reset()
    turn=await runner.next()
    first_decision=time.perf_counter()
    actors=Counter();callbacks=Counter();latencies=[]
    try:
        for _ in range(actions):
            if isinstance(turn,EpisodeEnd):
                raise RuntimeError('Game ended before the requested decision count')
            actors[turn.actor]+=1
            callbacks[turn.observation.decision.kind]+=1
            answer=smoke_response(turn.observation)
            before=time.perf_counter()
            await runner.submit(answer)
            turn=await runner.next()
            latencies.append(time.perf_counter()-before)
        play_done=time.perf_counter()
        # Explicitly end the game after exactly N gameplay decision submissions.
        # Concession itself is excluded from the N count, included in wall time.
        if not isinstance(turn,EpisodeEnd):
            await runner.games[turn.actor].concede()
        results=await asyncio.gather(*(g.wait_for_result(timeout=30) for g in runner.games.values()))
        if not all(r.terminal and r.outcome and r.outcome.won is not None for r in results):
            raise RuntimeError('Missing terminal result')
        finished=time.perf_counter()
    finally:
        await factory.close()
    end=time.perf_counter()
    return {'episode':number,'actions':runner.steps,'actors':dict(actors),'callbacks':dict(callbacks),
            'setup_seconds':first_decision-start,'play_seconds':play_done-first_decision,
            'finish_seconds':finished-play_done,'cleanup_seconds':end-finished,
            'total_seconds':end-start,'end_reason':'natural' if isinstance(turn,EpisodeEnd) else 'concession',
            'submit_to_next_decision_seconds':summary(latencies)}


async def run(args,cpu):
    deck=DeckSpec((CardSpec('Mountain',24),CardSpec('Lightning Bolt',36)),'Benchmark burn')
    factory=XMageSelfPlay(deck,deck_type='Constructed - Freeform Unlimited',max_decisions=args.actions+10)
    rows=[]
    try:
        for i in range(args.warmup):
            row=await episode(factory,i,args.actions)
            print(json.dumps({'warmup':True,**row}),flush=True)
        begin=time.perf_counter()
        for i in range(args.games):
            row=await episode(factory,i,args.actions)
            rows.append(row)
            print(json.dumps(row),flush=True)
        elapsed=time.perf_counter()-begin
    finally:await factory.close()
    play=sum(r['play_seconds'] for r in rows)
    model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')),platform.processor())
    result={'config':{'games':args.games,'warmup_games':args.warmup,'actions_per_game':args.actions,
                      'concurrent_games':1,'one_logical_cpu':cpu,'cpu_model':model,
                      'python':platform.python_version(),'server':'1.4.62-V1','proxy':'v0.4.7',
                      'deck':'24 Mountain, 36 Lightning Bolt; Freeform Unlimited',
                      'finish':'concede after action limit; await both results; remove table; disconnect'},
            'measured_seconds':elapsed,'games_per_minute':60*len(rows)/elapsed,
            'game_seconds':summary([r['total_seconds'] for r in rows]),
            'setup_seconds':summary([r['setup_seconds'] for r in rows]),
            'play_seconds':summary([r['play_seconds'] for r in rows]),
            'finish_seconds':summary([r['finish_seconds'] for r in rows]),
            'cleanup_seconds':summary([r['cleanup_seconds'] for r in rows]),
            'play_actions_per_second':sum(r['actions'] for r in rows)/play,
            'play_only_100_action_games_per_minute':60*len(rows)/play,
            'episodes':rows}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'summary':{k:v for k,v in result.items() if k!='episodes'}}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games',type=int,default=10)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--actions',type=int,default=100)
    parser.add_argument('--one-cpu',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'.local/xmage/benchmark.json')
    args=parser.parse_args()
    if args.games<1 or args.warmup<0 or args.actions<1:parser.error('Positive games/actions and nonnegative warmup required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with affinity(args.one_cpu,args.output) as cpu:
        asyncio.run(run(args,cpu))
