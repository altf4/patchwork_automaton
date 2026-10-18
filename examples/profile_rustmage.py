"""Profile the frozen mono-red collection pipeline, without updating weights.

Run with uv run python -m examples.profile_rustmage --checkpoint ... --output ...
Stage CPU timings are exclusive and thread-local. Wall timings from different
workers overlap and must never be interpreted as fractions of job wall time.
"""
import argparse
import asyncio
from collections import defaultdict
from contextlib import contextmanager, ExitStack
import ctypes
import cProfile
import functools
import hashlib
import json
import os
from pathlib import Path
import pstats
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import zlib

from . import train_rustmage  # CPU settings before TensorFlow imports
import numpy as np
import tensorflow as tf

from patchwork_automaton.backends import rustmage as backend
from patchwork_automaton.neural import batching, rustmage_features, rustmage_policy
from patchwork_automaton.neural.experiment import TrajectoryWriter, atomic_json
from patchwork_automaton.neural.history import History
from patchwork_automaton.rl import rustmage_experiment as experiment


class StageTimer:
    """Nested exclusive timers; each thread owns its stack and accumulators."""
    def __init__(self):
        self.local = threading.local()
        self.threads = []
        self.lock = threading.Lock()

    def state(self):
        if not hasattr(self.local, 'state'):
            state = SimpleNamespace(stack=[], rows=defaultdict(lambda: [0, 0., 0., 0., 0.]))
            with self.lock:
                self.threads.append(state)
            self.local.state = state
        return self.local.state

    @contextmanager
    def span(self, name):
        state = self.state()
        frame = [name, time.perf_counter(), time.thread_time(), 0., 0.]
        state.stack.append(frame)
        try:
            yield
        finally:
            cpu = time.thread_time()-frame[2]
            wall = time.perf_counter()-frame[1]
            assert state.stack.pop() is frame
            row = state.rows[name]
            row[0] += 1; row[1] += cpu; row[2] += wall
            row[3] += cpu-frame[3]; row[4] += wall-frame[4]
            if state.stack:
                state.stack[-1][3] += cpu
                state.stack[-1][4] += wall

    def wrap(self, name, function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with self.span(name):
                return function(*args, **kwargs)
        return wrapped

    def summary(self):
        rows = defaultdict(lambda: [0, 0., 0., 0., 0.])
        for state in self.threads:
            assert not state.stack
            for key, row in state.rows.items():
                rows[key] = [a+b for a,b in zip(rows[key], row)]
        names = ('calls', 'inclusive_thread_cpu_seconds', 'inclusive_wall_seconds',
                 'exclusive_thread_cpu_seconds', 'exclusive_wall_seconds')
        return {key: dict(zip(names, value)) for key,value in rows.items()}


def native_threads():
    """Linux CPU counters include TensorFlow's native worker pool."""
    root = Path('/proc/self/task')
    if not root.is_dir():
        return {}
    hz = os.sysconf('SC_CLK_TCK')
    result = {}
    for item in root.iterdir():
        try:
            fields = (item/'stat').read_text().rsplit(')',1)[1].split()
            result[int(item.name)] = (int(fields[11])+int(fields[12]))/hz
        except FileNotFoundError:
            pass
    return result


class ProfiledBatcher(batching.InferenceBatcher):
    def __init__(self, model, timer, **kwargs):
        super().__init__(model, **kwargs)
        self.timer = timer

    def _infer(self, requests):
        # Same numerical operations as the production batcher. Explicit tensor
        # conversion separates NumPy handoff from the compiled forward call.
        with self.timer.span('inference_other'):
            with self.timer.span('inference_batch_packing'):
                items, batch = batching.pack_requests(requests)
            with self.timer.span('numpy_to_tensorflow'):
                tensors = {key: tf.convert_to_tensor(value) for key,value in batch.items()}
            with self.timer.span('tensorflow_forward'):
                tensors_out = self.compiled(tensors)
            with self.timer.span('tensorflow_to_numpy'):
                output = {key: np.asarray(value) for key,value in tensors_out.items()}
            return [{k: (v[i:i+1, :len(items[i]['candidates'])] if k == 'logits'
                         else v[i:i+1]) for k,v in output.items()} for i in range(len(requests))]


@contextmanager
def instrument(timer, trace_dir=None):
    """Temporary, local instrumentation; restore all production functions."""
    original_loader = backend.load_binding
    traces = []
    wire = defaultdict(lambda: dict(calls=0, bytes=0))

    def load_binding(*args, **kwargs):
        module = original_loader(*args, **kwargs)
        cls = module.Game
        original_init = cls.__init__

        def initialize(self, decks, seed=0, starting_player='P0',
                       deck_rules='Constructed', *, library=None):
            self._profile_trace = None
            if trace_dir:
                trace_dir.mkdir(parents=True, exist_ok=True)
                self._profile_trace = (trace_dir/f'game-{seed}.jsonl').open('w')
                traces.append(self._profile_trace)
                config = dict(decks=decks, seed=seed, starting_player=starting_player,
                              deck_rules=deck_rules)
                self._profile_trace.write(json.dumps(dict(config=config))+'\n')
            with timer.span('native_game_initialization'):
                original_init(self, decks, seed, starting_player, deck_rules, library=library)

        def request(self, op, **fields):
            with timer.span('ffi_python_other'):
                with self._lock:
                    if not self._handle:
                        raise module.EngineError('game is closed')
                    with timer.span('request_json_encoding'):
                        payload = json.dumps(dict(op=op, **fields)).encode()
                    with timer.span('native_ffi/'+op):
                        pointer = self._lib.rm_request(self._handle, payload)
                    if not pointer:
                        raise module.EngineError('null native response')
                    try:
                        with timer.span('response_copy_utf8'):
                            raw = ctypes.string_at(pointer)
                            text = raw.decode('utf-8')
                    finally:
                        with timer.span('native_response_free'):
                            self._lib.rm_string_free(pointer)
                    with timer.span('response_json_decoding'):
                        response = module.json.loads(text)
                    wire[op]['calls'] += 1
                    wire[op]['bytes'] += len(raw)
                    if self._profile_trace:
                        with timer.span('trace_recording'):
                            self._profile_trace.write(json.dumps(dict(request=payload.decode(),
                                response_bytes=len(raw), response_adler32=zlib.adler32(raw)))+'\n')
                    if response.get('error'):
                        if response.get('capacity'):
                            raise module.CapacityError(response['error'], response['capacity']['resource'])
                        raise module.EngineError(response['error'])
                    return response

        cls.__init__, cls.request = initialize, request
        return module

    targets = (
        (experiment, 'play_game', 'worker_other'),
        (experiment, 'learning_view', 'python_observation_objects'),
        (rustmage_features.RustMageEncoder, 'encode', 'numerical_features'),
        (rustmage_features, 'expand_options', 'legal_action_encoding'),
        (rustmage_policy, 'expand_options', 'legal_action_encoding'),
        (rustmage_policy, 'collate', 'single_decision_batch'),
        (rustmage_policy.RustMagePolicy, 'act', 'policy_other'),
        (History, 'observe', 'history_tracking'),
        (History, 'take', 'history_arrays'),
        (TrajectoryWriter, 'write', 'trajectory_writing'),
        (TrajectoryWriter, 'close', 'trajectory_writing'),
        (batching.InferenceBatcher, 'predict', 'worker_inference_wait'),
        (experiment, 'visible_state_fingerprint', 'repeat_guard_fingerprint'),
    )
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(backend, 'load_binding', load_binding))
            for obj, name, label in targets:
                stack.enter_context(patch.object(obj, name, timer.wrap(label, getattr(obj, name))))
            yield wire
    finally:
        for trace in traces:
            trace.close()


def compare_games(first, second):
    fields = ('seed','seat','starting_player','status','winner','decisions',
              'neural_choices','own_turns_observed','final_life','actions')
    assert len(first) == len(second)
    matching = sum(all(a.get(k)==b.get(k) for k in fields) for a,b in zip(first,second))
    if matching != len(first):
        raise RuntimeError(f'Profile altered game records: {matching}/{len(first)} match')
    return matching


async def run(args):
    if min(args.games, args.workers, args.trace_games) < 1:
        raise ValueError('Counts must be positive')
    if args.resume and not (args.output/'report.json').is_file():
        raise FileNotFoundError('Resume requires an existing report.json')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    policy = rustmage_policy.RustMagePolicy.load(args.checkpoint)
    tf.config.experimental.set_synchronous_execution(True)
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha,
        games_per_pass=args.games, workers=args.workers, seed=args.seed,
        logical_cpus=os.cpu_count(), tf_intra_threads=os.environ['TF_NUM_INTRAOP_THREADS'],
        tf_inter_threads=os.environ['TF_NUM_INTEROP_THREADS'],
        timing_notes='Exclusive per-thread CPU; overlapping wall times cannot be summed into job wall percentages. Persistent native-runtime worker CPU is measured separately from Linux /proc counters (usually 10 ms resolution). Startup and warmup excluded; no PPO updates. cProfile is used only for a separate call-count diagnostic, never CPU or wall-time attribution.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), passes={})
    if args.resume:
        saved = json.loads((args.output/'report.json').read_text())
        for key in ('checkpoint_sha256','games_per_pass','workers','seed'):
            if saved[key]!=report[key]:
                raise ValueError(f'Resume configuration changed: {key}')
        saved.setdefault('resume_source_sha256',[]).append(report['source_sha256'])
        saved['timing_notes'] = report['timing_notes']
        if args.redo_call_profile:
            saved['passes'].pop('call_profile',None)
            saved.pop('call_profile_top_self_cpu',None)
            saved.pop('call_profile_top_call_counts',None)
            saved.pop('matching_call_profile_games',None)
        report = saved

    async def collect(label, *, profiled=False, trace=False, games=None, call_profile=False):
        if label in report['passes']:
            print(json.dumps(dict(stage=label,reused=True)),flush=True)
            return report['passes'][label]['games']
        timer = StageTimer()
        service = (ProfiledBatcher(policy.model,timer,max_batch=args.workers,wait_ms=2)
                   if profiled else batching.InferenceBatcher(policy.model,max_batch=args.workers,wait_ms=2))
        await service.start()
        try:
            await experiment.run_games(games=4,workers=args.workers,seed_base=args.seed+100000,
                mode='selfplay',model=policy.model,batcher=service,max_own_turns=60)
            # Discard warmup timings, including inference instrumentation.
            if profiled:
                timer = StageTimer(); service.timer = timer
            count = games or args.games
            before = service.stats()
            def completed(row):
                print(json.dumps(dict(stage=label,index=row['index'],status=row['status'])),flush=True)
            options = dict(games=count,workers=args.workers,seed_base=args.seed,
                mode='selfplay',model=policy.model,batcher=service,max_own_turns=60,
                output=args.output/label/'rollouts',progress=completed)
            if call_profile:
                # CPython's cProfile monitoring slot cannot be enabled by two
                # independent Profile objects concurrently. This is a separate
                # diagnostic, never a throughput sample.
                options['workers'] = 1
            inference_tid = await service.loop.run_in_executor(service.executor, threading.get_native_id)
            initial_threads = native_threads()
            start = time.perf_counter(); cpu = time.process_time(); event_cpu = time.thread_time()
            if profiled or trace:
                with instrument(timer,args.output/'native-traces' if trace else None) as wire:
                    rows = await experiment.run_games(**options)
            elif call_profile:
                original = experiment.play_game
                def detailed_game(**kw):
                    # The runtime's global profiling callbacks can cross native
                    # threads. A per-thread clock then has incomparable origins.
                    # Keep default timing internally, but report call counts only.
                    profiler = cProfile.Profile()
                    try:
                        return profiler.runcall(original, **kw)
                    finally:
                        profiler.dump_stats(str(args.output/f"game-{kw['seed']}.pstats"))
                with patch.object(experiment, 'play_game', detailed_game):
                    rows = await experiment.run_games(**options)
                wire = {}
            else:
                rows = await experiment.run_games(**options); wire = {}
            event_cpu = time.thread_time()-event_cpu
            cpu = time.process_time()-cpu; wall = time.perf_counter()-start
            final_threads = native_threads()
            # Game workers have exited. Persistent threads other than the main
            # loop and inference executor belong to native runtimes (TensorFlow
            # and any idle NumPy pools). /proc resolution is usually 10 ms.
            runtime_cpu = sum(final_threads[tid]-initial_threads[tid]
                for tid in initial_threads.keys() & final_threads.keys()
                if tid not in (threading.get_native_id(),inference_tid))
            if any(row['status'] not in ('win','loss','draw') for row in rows):
                atomic_json(args.output/(label+'-failed.json'), rows)
                raise RuntimeError('Profile requires naturally terminated games')
            after = service.stats()
            calls = after['calls']-before['calls']; requests = after['requests']-before['requests']
            result = dict(seconds=wall,process_cpu_seconds=cpu,mean_cores=cpu/wall,
                main_event_loop_cpu_seconds=event_cpu,
                persistent_native_runtime_cpu_seconds=runtime_cpu,
                games_per_minute=60*count/wall,games=rows,wire=dict(wire),
                inference=dict(calls=calls,requests=requests,mean_batch=requests/max(calls,1),
                    service_seconds=after['inference_seconds']-before['inference_seconds'],
                    aggregate_request_queue_seconds=(after['mean_queue_ms']*after['requests']
                        -before['mean_queue_ms']*before['requests'])/1000),
                stages=timer.summary() if profiled or trace else {})
            report['passes'][label] = result
            atomic_json(args.output/'report.json', report)
            print(json.dumps(dict(stage=label,seconds=wall,cpu_seconds=cpu)),flush=True)
            return rows
        finally:
            await service.close()

    before = await collect('reference_before')
    measured = await collect('profiled',profiled=True)
    after = await collect('reference_after')
    report['matching_profiled_games'] = compare_games(before,measured)
    report['matching_reference_games'] = compare_games(before,after)
    reference_wall = (report['passes']['reference_before']['seconds']+report['passes']['reference_after']['seconds'])/2
    report['instrumentation_wall_ratio'] = report['passes']['profiled']['seconds']/reference_wall
    if args.trace_games:
        traced = await collect('trace_capture',trace=True,games=min(args.trace_games,args.games))
        report['matching_traced_games'] = compare_games(before[:len(traced)],traced)
    if args.cprofile_games:
        detailed = await collect('call_profile',games=min(args.cprofile_games,args.games),call_profile=True)
        report['matching_call_profile_games'] = compare_games(before[:len(detailed)],detailed)
        stats = pstats.Stats(*map(str, sorted(args.output.glob('game-*.pstats'))))
        report['call_profile_top_call_counts'] = [dict(file=key[0],line=key[1],function=key[2],
            primitive_calls=value[0],calls=value[1])
            for key,value in sorted(stats.stats.items(),key=lambda item:-item[1][1])[:60]]
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==sha
    stages = report['passes']['profiled']['stages']
    accounted = sum(row['exclusive_thread_cpu_seconds'] for row in stages.values())
    report['profiled_accounted_thread_cpu_seconds'] = accounted
    report['profiled_unattributed_process_cpu_seconds'] = report['passes']['profiled']['process_cpu_seconds']-accounted
    report['profiled_unattributed_after_runtime_and_event_loop_seconds'] = (
        report['profiled_unattributed_process_cpu_seconds']
        -report['passes']['profiled']['persistent_native_runtime_cpu_seconds']
        -report['passes']['profiled']['main_event_loop_cpu_seconds'])
    atomic_json(args.output/'report.json', report)
    print(json.dumps(dict(stage='complete',report=str(args.output/'report.json'),
        instrumentation_wall_ratio=report['instrumentation_wall_ratio'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--games',type=int,default=16)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--trace-games',type=int,default=4)
    parser.add_argument('--cprofile-games',type=int,default=2,
        help='Separate slow call-profiler diagnostic; excluded from stage timing and throughput')
    parser.add_argument('--seed',type=int,default=23000000)
    parser.add_argument('--resume',action='store_true',help='Reuse completed passes after a diagnostic failure')
    parser.add_argument('--redo-call-profile',action='store_true',help='With --resume, repeat only the optional call-count diagnostic')
    asyncio.run(run(parser.parse_args()))
