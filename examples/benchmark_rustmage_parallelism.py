"""Measure CPU parallelism with fixed checkpoints, games and PPO rollouts.

Run each configuration in a fresh process: TensorFlow thread pools are fixed at
initialization. Collection and updates are separate, avoiding contention between
benchmark trials. No input checkpoint is changed. Use uv run --no-sync python -m
examples.benchmark_rustmage_parallelism --help for the two modes.
"""
import argparse
import asyncio
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class InferenceServices:
    """Benchmark routing: keep each game's requests on one frozen service."""
    def __init__(self, services):
        self.services, self.routes = services, {}
        self.lock = threading.Lock()

    def predict(self, batch, *, identity, **options):
        key = identity[:2]
        with self.lock:
            if key not in self.routes:
                self.routes[key] = len(self.routes) % len(self.services)
            service = self.services[self.routes[key]]
        return service.predict(batch, identity=identity, **options)


def metadata(args):
    import tensorflow as tf
    return dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        checkpoint_sha256=digest(args.checkpoint), platform=platform.platform(),
        cpu_model=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                       if line.startswith('model name')),
        available_cpus=sorted(os.sched_getaffinity(0)), tensorflow=tf.__version__,
        runtime={k: os.environ.get(k) for k in ('TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS',
                                              'TF_ENABLE_ONEDNN_OPTS', 'CUDA_VISIBLE_DEVICES')},
        source_sha256={str(p): digest(p) for p in [Path(__file__),
            *Path('src/patchwork_automaton/neural').glob('*.py'),
            *Path('src/patchwork_automaton/rl').glob('rustmage*.py')]})


def weights_hash(model):
    import numpy as np
    return hashlib.sha256(b''.join(np.asarray(v).tobytes() for v in model.weights)).hexdigest()


async def collect(args, report):
    from patchwork_automaton.neural.batching import InferenceBatcher
    from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
    from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector
    services = []
    collector = None
    timer = None
    try:
        if args.inference_processes > 1:
            model = RustMagePolicy.load(args.checkpoint).model
            collector = ProcessGameCollector(model,workers=args.workers,
                inference_processes=args.inference_processes,wait_ms=args.wait_ms)
            await collector.start()
            replica_source_hash = weights_hash(model)
        else:
            for _ in range(args.inference_services):
                model = RustMagePolicy.load(args.checkpoint).model
                if args.profile:
                    from .profile_rustmage import ProfiledBatcher, StageTimer
                    if timer is None: timer = StageTimer()
                    service = ProfiledBatcher(model, timer, max_batch=args.max_batch, wait_ms=args.wait_ms)
                else:
                    service = InferenceBatcher(model, max_batch=args.max_batch, wait_ms=args.wait_ms)
                services.append(await service.start())
            collector = ProcessGameCollector(services[0].model, InferenceServices(services), workers=args.workers)
        initial_hashes = [weights_hash(s.model) for s in services]
        def inference_stats():
            return collector.stats()['replicas'] if args.inference_processes > 1 else [s.stats() for s in services]
        warm = []
        for attempt in range(4):
            warm.extend(await collector.run_games(games=args.workers,
                seed_base=args.seed + 100000 + attempt*args.workers, mode='selfplay', max_own_turns=60))
            if any(r['status'] not in ('win', 'loss', 'draw') for r in warm):
                raise RuntimeError('Warm-up did not finish naturally')
            if len({r['worker_pid'] for r in warm}) == args.workers: break
        else: raise RuntimeError('Not every worker completed warm-up')
        report['warmup'] = dict(games=len(warm), initialized_workers=len({r['worker_pid'] for r in warm}))
        if args.start_signal:
            # A benchmark coordinator can release separately warmed collectors
            # together. No game/policy state is shared across these processes.
            (args.output/'ready').touch()
            deadline = time.monotonic()+120
            while not args.start_signal.exists():
                if time.monotonic() >= deadline: raise TimeoutError('Collection start signal timed out')
                await asyncio.sleep(.05)
        before = inference_stats()
        if timer:
            for state in timer.threads: state.rows.clear()
        started = time.perf_counter(); cpu = time.process_time()
        rows = await collector.run_games(games=args.games, seed_base=args.seed, index_base=args.index_base, mode='selfplay',
            max_own_turns=60, output=args.output/'rollouts',
            progress=lambda r: print(json.dumps(dict(stage='game', index=r['index'], status=r['status'])), flush=True))
        seconds = time.perf_counter() - started
        parent_cpu = time.process_time() - cpu
        worker_cpu = sum(r['worker_cpu_seconds'] for r in rows)
        after = inference_stats()
        replica_cpu = sum(b.get('process_cpu_seconds',0)-a.get('process_cpu_seconds',0) for a,b in zip(before,after))
        requests = sum(b['requests']-a['requests'] for a,b in zip(before,after))
        calls = sum(b['calls']-a['calls'] for a,b in zip(before,after))
        inference_seconds = sum(b['inference_seconds']-a['inference_seconds'] for a,b in zip(before,after))
        waiting = sum(r['worker_inference_seconds'] for r in rows)
        worker_wall = sum(r['seconds'] for r in rows)
        if [weights_hash(s.model) for s in services] != initial_hashes:
            raise AssertionError('Collection changed model weights')
        if args.inference_processes > 1:
            if weights_hash(model) != replica_source_hash: raise AssertionError('Collection changed learner weights')
            report['replica_policies'] = [dict(pid=s['pid'],engines=s['engines'],version=s['version'],
                weights_sha256=s['weights_sha256']) for s in after]
        report.update(seconds=seconds, games_per_minute=60*len(rows)/seconds,
            collection_started_monotonic=started, collection_finished_monotonic=started+seconds,
            choices_per_second=sum(r['neural_choices'] for r in rows)/seconds,
            parent_cpu_seconds=parent_cpu, worker_cpu_seconds=worker_cpu,
            inference_replica_cpu_seconds=replica_cpu,
            mean_cpu_cores=(parent_cpu+worker_cpu+replica_cpu)/seconds,
            inference=dict(requests=requests, calls=calls, mean_batch=requests/max(calls,1),
                summed_service_seconds=inference_seconds,
                service_busy_fraction=inference_seconds/(seconds*len(after)),
                worker_wait_seconds=waiting, summed_worker_wall_seconds=worker_wall,
                worker_wait_fraction=waiting/worker_wall,
                graph_traces=[s['graph_traces'] for s in after]),
            games=rows, stages=timer.summary() if timer else {},
            scope='Warmed full self-play collection, including trajectory writing; startup excluded. '
                  'Worker waits overlap and cannot be added to job wall time.')
        report['all_natural'] = all(r['status'] in ('win', 'loss', 'draw') for r in rows)
        if not report['all_natural']: raise RuntimeError('Timed games did not finish naturally')
    finally:
        if collector: await collector.close()
        for service in services: await service.close()


def update(args, report):
    import numpy as np
    from patchwork_automaton.neural.batching import example_batch
    from patchwork_automaton.neural.experiment import make_trainer
    from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
    checkpoint_files = [p for p in args.checkpoint.parent.iterdir() if p.is_file()]
    report['input_checkpoint_sha256'] = {str(p): digest(p) for p in checkpoint_files}
    model = RustMagePolicy.load(args.checkpoint).model
    state = json.loads(args.checkpoint.with_name('state.json').read_text())
    trainer = make_trainer(model, dict(state['config'], concurrent_updates=args.concurrent_updates,
        prefetch_batches=args.prefetch_batches,measurement_workers=args.measurement_workers))
    trainer.compile()
    for optimizer, filename in [(trainer.optimizer, 'optimizer.npz'), (trainer.critic_optimizer, 'critic-optimizer.npz')]:
        with np.load(args.checkpoint.with_name(filename), allow_pickle=False) as archive:
            if len(archive.files) != len(optimizer.variables): raise ValueError('Optimizer state mismatch')
            for i, variable in enumerate(optimizer.variables): variable.assign(archive[f'v{i}'])
    trainer.rng.bit_generator.state = state['trainer_rng_state']
    source = json.loads(args.rollout_report.read_text())
    round_ = next(r for r in source['rounds'] if r['round'] == args.round)
    if state['round'] != args.round-1: raise ValueError('Use the checkpoint preceding the selected rollout')
    trajectories = [t for game in round_['games'] for t in game['trajectories']]
    report.update(rollout_report_sha256=digest(args.rollout_report),
        trajectory_sha256={t['path']: digest(t['path']) for t in trajectories},
        original_update=round_['update'])
    variables = list(model.variables) + list(trainer.optimizer.variables) + list(trainer.critic_optimizer.variables)
    saved = [np.asarray(v).copy() for v in variables]
    batch = example_batch(model.memory_size, model.feature_spec)
    zero = np.zeros(1, np.float32)
    trainer.step(batch, np.zeros(1, np.int32), zero, zero, zero)
    trainer.critic_step(batch, zero)
    for variable, value in zip(variables, saved): variable.assign(value)
    stages = defaultdict(lambda: dict(calls=0, seconds=0., cpu_seconds=0., calling_thread_cpu_seconds=0.))
    intervals = []
    timer_lock = threading.Lock()
    parallel_calls = args.concurrent_updates or args.measurement_workers > 1
    def timed(name, function):
        def wrapped(*a, **kw):
            start = time.perf_counter(); cpu = time.process_time(); thread_cpu = time.thread_time()
            try:
                return function(*a, **kw)
            finally:
                end = time.perf_counter()
                process_elapsed = time.process_time()-cpu
                thread_elapsed = time.thread_time()-thread_cpu
                with timer_lock:
                    row = stages[name]
                    row['seconds'] += end-start
                    # Process CPU includes the other optimizer during overlap;
                    # attributing it to this individual stage would double count.
                    row['cpu_seconds'] = None if parallel_calls else row['cpu_seconds'] + process_elapsed
                    row['calling_thread_cpu_seconds'] += thread_elapsed
                    row['calls'] += 1
                    intervals.append((start,end))
        return wrapped
    for name in ('step', 'critic_step', 'measure_with_drift', 'measure_actor_with_drift', 'snapshot', 'restore'):
        setattr(trainer, name, timed(name, getattr(trainer, name)))
    started = time.perf_counter(); cpu = time.process_time()
    result = trainer.update_files(trajectories)
    seconds = time.perf_counter()-started; cpu = time.process_time()-cpu
    covered = 0.; right = started
    for left, end in sorted(intervals):
        covered += max(0., end-max(left,right)); right = max(right,end)
    overlap = sum(r['seconds'] for r in stages.values())-covered
    stages['outside_graph_calls'] = dict(seconds=max(0.,seconds-covered),
        cpu_seconds=None if parallel_calls else cpu-sum(r['cpu_seconds'] for r in stages.values()))
    report.update(seconds=seconds, cpu_seconds=cpu, mean_cpu_cores=cpu/seconds,
        games_per_minute=60*len(round_['games'])/seconds, result=result, stages=dict(stages),
        graph_call_overlap_seconds=overlap,
        trainer_rng_state=trainer.rng.bit_generator.state,
        stage_time_semantics='Per-stage wall durations overlap in concurrent mode; outside_graph_calls '
            'uses the union of timed graph intervals. It is not total data-preparation cost, '
            'because preparation can overlap a graph on the other thread. Only total process '
            'CPU accounts for all TensorFlow threads without double counting. graph_call_overlap_seconds '
            'is summed call durations minus their interval union; it can exceed elapsed time with '
            'three or more simultaneous calls.',
        scope='One full PPO update on identical saved rollouts, restored model/Adam/RNG. '
              'Model and gradient graphs warmed; rollout reads, packing and snapshot overhead included; '
              'input checkpoints unchanged.')
    np.savez(args.output/'updated-weights.npz', *model.get_weights())
    for name in ('optimizer','critic_optimizer'):
        np.savez(args.output/f'updated-{name}.npz', *[np.asarray(v) for v in getattr(trainer,name).variables])
    if {str(p): digest(p) for p in checkpoint_files} != report['input_checkpoint_sha256']:
        raise AssertionError('Input checkpoint or optimizer state changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('collect', 'update'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--intra', type=int, default=2)
    parser.add_argument('--inter', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--inference-services', type=int, default=1)
    parser.add_argument('--inference-processes',type=int,default=1,
        help='Independent inference processes sharing the total game-worker count')
    parser.add_argument('--max-batch', type=int, default=8)
    parser.add_argument('--wait-ms', type=float, default=2)
    parser.add_argument('--games', type=int, default=48)
    parser.add_argument('--seed', type=int, default=71000000)
    parser.add_argument('--index-base',type=int,default=0,
        help='First global game index for a disjoint collection shard')
    parser.add_argument('--start-signal',type=Path,
        help='Collection benchmark waits for this file after publishing output/ready')
    parser.add_argument('--rollout-report', type=Path)
    parser.add_argument('--round', type=int, default=40)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--concurrent-updates', action='store_true',
        help='Overlap independent actor/critic updates in update mode')
    parser.add_argument('--prefetch-batches',type=int,default=0)
    parser.add_argument('--measurement-workers',type=int,default=1)
    args = parser.parse_args()
    if min(args.intra,args.inter,args.workers,args.inference_services,args.max_batch,args.games,args.round) < 1 or args.wait_ms < 0:
        parser.error('Counts must be positive; wait-ms must be nonnegative')
    if args.mode == 'update' and not args.rollout_report: parser.error('update requires --rollout-report')
    if args.mode != 'update' and args.concurrent_updates: parser.error('concurrent-updates requires update mode')
    if args.prefetch_batches < 0 or args.measurement_workers < 1: parser.error('Invalid update pipeline limits')
    if args.mode != 'update' and (args.prefetch_batches or args.measurement_workers != 1):
        parser.error('Update pipeline options require update mode')
    if args.index_base < 0: parser.error('index-base must be nonnegative')
    if args.mode != 'collect' and (args.index_base or args.start_signal):
        parser.error('Collection coordination options require collect mode')
    if not 1 <= args.inference_processes <= args.workers:
        parser.error('Inference process count must be between one and workers')
    if args.inference_processes > 1:
        if args.mode != 'collect' or args.inference_services != 1 or args.profile:
            parser.error('Replicated inference requires collect mode, one service per process and no per-stage profile')
        if args.max_batch != args.workers:
            parser.error('For replicated inference set max-batch to workers; each replica batches its assigned engines')
    os.environ.update(KERAS_BACKEND='tensorflow', CUDA_VISIBLE_DEVICES='-1',
        TF_NUM_INTRAOP_THREADS=str(args.intra), TF_NUM_INTEROP_THREADS=str(args.inter),
        TF_ENABLE_ONEDNN_OPTS='0', TF_CPP_MIN_LOG_LEVEL='2')
    args.output.mkdir(parents=True, exist_ok=False)
    from patchwork_automaton.neural.experiment import atomic_json
    report = metadata(args)
    try:
        if args.mode == 'collect': asyncio.run(collect(args, report))
        else: update(args, report)
        if digest(args.checkpoint) != report['checkpoint_sha256']: raise AssertionError('Input checkpoint changed')
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'error'; report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output/'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('games','source_sha256','trajectory_sha256','original_update')}), flush=True)


if __name__ == '__main__': main()
