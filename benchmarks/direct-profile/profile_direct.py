"""CPU Standard self-play collection benchmark; no training or checkpoint writes.

Run from the frozen project root with this directory on PYTHONPATH. Reports child
CPU by exclusive stage, parent-worker CPU, and replica CPU separately. The
replica service spans are elapsed, overlap across replicas, and are not job wall
percentages. Reference and profiled pilots use identical seeds and schedules.
"""
import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import contextmanager
import ctypes
import functools
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('PATCHWORK_OBSERVATION_PIPELINE', 'python')
os.environ.setdefault('PATCHWORK_ROLLOUT_STORAGE', 'pickle')

class Timer:
    def __init__(self):
        self.stack = []
        self.rows = defaultdict(lambda: [0, 0., 0.])

    @contextmanager
    def span(self, name):
        frame = [time.process_time(), 0.]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.process_time() - frame[0]
            self.stack.pop()
            row = self.rows[name]
            row[0] += 1; row[1] += elapsed; row[2] += elapsed - frame[1]
            if self.stack:
                self.stack[-1][1] += elapsed

    def wrap(self, name, f):
        @functools.wraps(f)
        def wrapped(*a, **kw):
            with self.span(name): return f(*a, **kw)
        return wrapped

    def result(self):
        return {k: dict(calls=v[0], inclusive_cpu=v[1], exclusive_cpu=v[2])
                for k, v in self.rows.items()}

TIMER = Timer()

def instrument_worker():
    from patchwork_automaton.rl import standard_imitation as game
    from patchwork_automaton.rl import standard_direct as direct
    from patchwork_automaton.neural.rollout_records import TrajectoryWriter
    from patchwork_automaton.rl.rustmage_processes import _RemoteInference
    lib = direct.library()  # initialize once before the game fork
    native_names = ('ds_new', 'ds_free', 'ds_observe', 'ds_view_free', 'ds_metadata',
                    'ds_history_observe', 'ds_history_take', 'ds_prepare', 'ds_encode',
                    'ds_apply_choice', 'ds_choice_kind', 'ds_string_free',
                    'ds_arrays_views', 'ds_arrays_free')
    for name in native_names:
        setattr(lib, name, TIMER.wrap('native_' + name.removeprefix('ds_'), getattr(lib, name)))
    targets = [(direct, 'arrays', 'numpy_arrays'), (direct, 'string', 'action_label'),
               (TrajectoryWriter, 'write', 'rollout_write'), (TrajectoryWriter, 'close', 'rollout_close'),
               (_RemoteInference, 'predict', 'inference_pipe')]
    for obj, name, label in targets:
        setattr(obj, name, TIMER.wrap(label, getattr(obj, name)))
    original_game = game.play_standard_game
    def play(*a, **kw):
        TIMER.rows.clear()
        started = time.process_time()
        with TIMER.span('game_other'):
            row = original_game(*a, **kw)
        row['profile_child_cpu'] = time.process_time() - started
        row['profile_stages'] = TIMER.result()
        return row
    game.play_standard_game = play


def initialize(connections, counter, metadata, profile):
    from patchwork_automaton.rl.standard_selfplay_processes import _initialize_worker
    _initialize_worker(connections, counter, metadata)
    if profile: instrument_worker()


def profiled_replica(*args):
    # Installed inside each spawned replica, before it creates its batcher.
    from contextlib import nullcontext
    from patchwork_automaton.neural import batching
    import numpy as np
    stages = defaultdict(lambda: [0, 0.])
    @contextmanager
    def span(name):
        start = time.perf_counter()
        try: yield
        finally:
            stages[name][0] += 1
            stages[name][1] += time.perf_counter() - start
    def infer(self, requests):
        with span('batch_packing'):
            items, batch = batching.pack_requests(requests)
        with batching.device_scope(self.device) if self.device else nullcontext():
            with span('model_forward'):
                outputs = self.compiled(batch)
        with span('result_conversion'):
            if self.device: self.output_placements = batching.assert_output_device(outputs, self.device)
            output = {k: np.asarray(v) for k,v in outputs.items()}
            return [{k: (v[i:i+1, :len(items[i]['candidates'])] if k == 'logits' else v[i:i+1])
                     for k,v in output.items()} for i in range(len(requests))]
    old_stats = batching.InferenceBatcher.stats
    def stats(self):
        return dict(old_stats(self), profile_service_stages={k:dict(calls=v[0],wall_seconds=v[1]) for k,v in stages.items()})
    batching.InferenceBatcher._infer = infer
    batching.InferenceBatcher.stats = stats
    from patchwork_automaton.neural.inference_processes import _replica_main
    _replica_main(*args)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def comparison(a, b):
    fields = ('index', 'seed', 'status', 'winner_seat', 'decisions', 'turn', 'final_life', 'actions')
    aa = {r['index']:r for r in a}; bb = {r['index']:r for r in b}
    return dict(games=len(aa), identical=sum(all(aa[i].get(k)==bb[i].get(k) for k in fields) for i in aa if i in bb),
                fields=list(fields))


async def run(args):
    from patchwork_automaton.neural.devices import configure_runtime, device_scope
    from patchwork_automaton.neural.standard_policy import StandardPolicy
    from patchwork_automaton.rl.standard_ppo import schedule, validate_collection
    from patchwork_automaton.rl.standard_selfplay_processes import StandardSelfplayCollector
    from patchwork_automaton.neural.inference_processes import weights_digest
    from examples.train_standard_ppo import source_hashes
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve(strict=True); checkpoint_sha = digest(checkpoint)
    manifest = dict(status='running', checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        script_sha256=digest(__file__), games=args.games, seed=args.seed, workers=args.workers,
        runners=args.runners, learner_replicas=args.learner_replicas, opponent_replicas=args.opponent_replicas,
        notes='Exclusive CPU-seconds are summed across processes; model service wall intervals overlap. Native observe includes RustMage typed observation construction; apply includes engine rules. Replica CPU includes model, batching, IPC and runtime. Control pilots use matched seeds.')
    manifest['source_sha256'] = source_hashes()
    manifest['logical_cpus'] = os.cpu_count()
    manifest['cpu_model'] = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), None)
    atomic(root/'manifest.json', manifest)
    configure_runtime('cpu', hide_gpu=True)
    with device_scope('cpu'): model = StandardPolicy.load(checkpoint).model
    model_digest = weights_digest(model.get_weights())
    async def collect(label, games, profile, planned):
        out = root/label; out.mkdir(); (out/'trajectories').mkdir(); (out/'diagnostics').mkdir()
        cs = [StandardSelfplayCollector(model, model, workers=args.workers//args.runners,
              inference_processes=args.learner_replicas//args.runners,
              opponent_inference_processes=args.opponent_replicas//args.runners) for _ in range(args.runners)]
        if profile:
            from patchwork_automaton.neural import inference_processes
            inference_processes._replica_main = profiled_replica
        for c in cs:
            c.pool._initializer = initialize
            c.pool._initargs = (*c.pool._initargs, profile)
        rows = []
        try:
            await asyncio.gather(*(c.start(version=250, opponent_version=250) for c in cs))
            if any(c.policy_digest != model_digest or c.opponent_digest != model_digest for c in cs):
                raise RuntimeError('Replica weights differ')
            # First games warm up worker imports, fork path, and inference graph.
            warm = schedule(240, seed=args.seed+1000000, round_id=1)[:args.workers]
            for g in warm: g.update(opponent_kind='frozen_neural',opponent_version=250,opponent_checkpoint_sha256=checkpoint_sha)
            async def batch(jobs, callback=None):
                await asyncio.gather(*(c.run_schedule([dict(seed=g['seed'],index=g['index'],game=g) for g in jobs[i::len(cs)]],
                    progress=callback, game_runner='standard_ppo', output=out/'trajectories',
                    diagnostics=out/'diagnostics', game_timeout=60, max_turns=50,
                    opponent_version=250, opponent_sha256=checkpoint_sha, greedy=False)
                    for i,c in enumerate(cs)))
            await batch(warm)
            before = [c.stats() for c in cs]
            started = time.perf_counter()
            coordinator_cpu = time.process_time()
            with (out/'games.jsonl').open('w') as stream:
                def progress(row):
                    rows.append(row)
                    stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                    if len(rows)%100 == 0:
                        stream.flush()
                        atomic(root/'progress.json',dict(phase=label,completed=len(rows),total=games,
                            seconds=time.perf_counter()-started))
                for i in range(0, games, 1000): await batch(planned[i:i+1000], progress)
            elapsed = time.perf_counter()-started
            coordinator_cpu = time.process_time()-coordinator_cpu
            validate_collection(rows, planned, 250)
            after = [c.stats() for c in cs]
            aggregate = defaultdict(lambda: [0,0.,0.])
            for row in rows:
                for name, v in row.get('profile_stages', {}).items():
                    z=aggregate[name]; z[0]+=v['calls']; z[1]+=v['inclusive_cpu']; z[2]+=v['exclusive_cpu']
            metrics = dict(games=len(rows), wall_seconds=elapsed, games_per_minute=60*len(rows)/elapsed,
                outcomes=dict(Counter(r['status'] for r in rows)), draw_reasons=dict(Counter(r.get('reason') for r in rows if r['status']=='draw')),
                decisions=sum(r['decisions'] for r in rows), coordinator_cpu_seconds=coordinator_cpu, parent_worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in rows),
                child_cpu_seconds=sum(r.get('profile_child_cpu',0) for r in rows),
                inference_wait_seconds=sum(r['worker_inference_seconds']+r['opponent_inference_seconds'] for r in rows),
                stages={k:dict(calls=v[0],inclusive_cpu=v[1],exclusive_cpu=v[2]) for k,v in aggregate.items()},
                replica_before=before, replica_after=after)
            for role in ('learner','opponent'):
                metrics[role+'_replica_cpu_seconds'] = sum(y[role]['replica_cpu_seconds']-x[role]['replica_cpu_seconds'] for x,y in zip(before,after))
                metrics[role+'_service_wall_seconds'] = sum(y[role]['inference_seconds']-x[role]['inference_seconds'] for x,y in zip(before,after))
                metrics[role+'_requests'] = sum(y[role]['requests']-x[role]['requests'] for x,y in zip(before,after))
                metrics[role+'_calls'] = sum(y[role]['calls']-x[role]['calls'] for x,y in zip(before,after))
            if profile:
                service = defaultdict(float)
                for x,y in zip(before,after):
                    for role in ('learner','opponent'):
                        for old,new in zip(x[role]['replicas'], y[role]['replicas']):
                            for key,value in new['profile_service_stages'].items():
                                service[key] += value['wall_seconds'] - old.get('profile_service_stages', {}).get(key, {}).get('wall_seconds', 0.)
                metrics['service_stage_wall_seconds'] = dict(service)
            atomic(out/'summary.json',metrics)
            print(json.dumps(dict(phase=label,wall_seconds=elapsed,games=len(rows))),flush=True)
            return rows, metrics
        finally:
            await asyncio.gather(*(c.close() for c in cs))
    def planned(rounds):
        result = []
        for n in range(1, rounds+1): result.extend(schedule(1000,seed=args.seed,round_id=n))
        for g in result: g.update(opponent_kind='frozen_neural',opponent_version=250,opponent_checkpoint_sha256=checkpoint_sha)
        return result
    try:
        main = planned(args.games//1000)
        pilot = main[:args.pilot_games]
        a, am = await collect('control',len(pilot),False,pilot)
        b, bm = await collect('profile-pilot',len(pilot),True,pilot)
        check = comparison(a,b)
        if check['identical'] != len(pilot): raise RuntimeError(f'Pilot parity failed: {check}')
        manifest['pilot_comparison'] = check
        manifest['pilot_profile_wall_ratio'] = bm['wall_seconds']/am['wall_seconds']
        atomic(root/'manifest.json',manifest)
        _, result = await collect('profile',len(main),True,main)
        if digest(checkpoint)!=checkpoint_sha: raise RuntimeError('Checkpoint changed')
        if source_hashes()!=manifest['source_sha256']: raise RuntimeError('Frozen source changed')
        manifest.update(status='complete', result=result)
        atomic(root/'manifest.json',manifest)
    except BaseException as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}')
        atomic(root/'manifest.json',manifest)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--games',type=int,default=10000); p.add_argument('--pilot-games',type=int,default=240)
    p.add_argument('--seed',type=int,default=962610310000); p.add_argument('--workers',type=int,default=48)
    p.add_argument('--runners',type=int,default=2); p.add_argument('--learner-replicas',type=int,default=2)
    p.add_argument('--opponent-replicas',type=int,default=2)
    a=p.parse_args()
    if a.games<1000 or a.games%1000 or not 1<=a.pilot_games<=a.games or any(x%a.runners for x in (a.workers,a.learner_replicas,a.opponent_replicas)):
        p.error('Games must be a multiple of 1000; workers and replicas must divide evenly among runners')
    asyncio.run(run(a))

if __name__ == '__main__': main()
