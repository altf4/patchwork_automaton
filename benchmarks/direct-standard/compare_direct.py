"""Matched Python-versus-direct-Rust Standard self-play benchmark.

Runs alternating configurations on identical 1,000-game schedules. No model
training occurs. Each phase has fresh, separately warmed workers and replicas.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
import itertools

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('PATCHWORK_OBSERVATION_PIPELINE', 'python')
os.environ.setdefault('PATCHWORK_ROLLOUT_STORAGE', 'pickle')


def initialize(connections, counter, metadata, native):
    from patchwork_automaton.rl.standard_selfplay_processes import _initialize_worker
    _initialize_worker(connections, counter, metadata)
    if native:
        from patchwork_automaton.rl import standard_imitation
        import runtime
        runtime.native()  # load/configure before the single-threaded game fork
        standard_imitation.play_standard_game = runtime.play


def compare_trajectories(a, b):
    import numpy as np
    from patchwork_automaton.neural.rollout_records import read_trajectory
    aa={r['index']:r for r in a}; bb={r['index']:r for r in b}
    records=0; compared_games=0; skipped=[]; feature_mismatch=[]; value_mismatch=[]; max_value_error=0.; max_logp_error=0.; max_memory_error=0.; memory_mismatch=[]
    for i in aa:
        left,right=aa[i],bb[i]
        if left.get('adjudicated') or right.get('adjudicated'):
            skipped.append(i); continue
        lt,rt=left['trajectories'],right['trajectories']
        if len(lt)!=1 or len(rt)!=1: raise ValueError(f'Missing trajectory for {i}')
        for idx,(x,y) in enumerate(itertools.zip_longest(read_trajectory(lt[0]['path']),read_trajectory(rt[0]['path']))):
            if x is None or y is None:
                feature_mismatch.append(dict(game=i,record=idx,reason='length'));break
            records+=1
            if x.action!=y.action or x.advance!=y.advance or set(x.arrays)!=set(y.arrays):
                feature_mismatch.append(dict(game=i,record=idx,reason='action_or_keys'));continue
            for k in x.arrays:
                u,v=x.arrays[k],y.arrays[k]
                if u.dtype!=v.dtype or u.shape!=v.shape or not np.array_equal(u,v):
                    feature_mismatch.append(dict(game=i,record=idx,reason=k));break
            if x.memory.shape!=y.memory.shape:
                memory_mismatch.append(dict(game=i,record=idx))
            else:
                max_memory_error=max(max_memory_error,float(np.max(np.abs(x.memory-y.memory))))
                if not np.allclose(x.memory,y.memory,rtol=1e-5,atol=1e-6):
                    memory_mismatch.append(dict(game=i,record=idx))
            e=abs(x.value-y.value); max_value_error=max(max_value_error,e)
            if e>1e-6+1e-5*abs(x.value): value_mismatch.append(dict(game=i,record=idx,reason='value'))
            if x.log_probs.shape!=y.log_probs.shape:
                value_mismatch.append(dict(game=i,record=idx,reason='logp_shape'))
            else:
                e=float(np.max(np.abs(x.log_probs-y.log_probs)))
                max_logp_error=max(max_logp_error,e)
                if not np.allclose(x.log_probs,y.log_probs,rtol=1e-5,atol=1e-6): value_mismatch.append(dict(game=i,record=idx,reason='logp'))
        compared_games+=1
    return dict(compared_games=compared_games,records=records,skipped_adjudicated=skipped,
        feature_mismatches=feature_mismatch[:20],feature_mismatch_count=len(feature_mismatch),
        memory_mismatch_count=len(memory_mismatch),output_mismatch_count=len(value_mismatch),
        max_abs_memory_error=max_memory_error,max_abs_value_error=max_value_error,max_abs_logp_error=max_logp_error,
        output_tolerance={'absolute':1e-6,'relative':1e-5})


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def save(path, data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def compare(a,b):
    fields=('index','seed','status','winner_seat','decisions','turn','final_life','actions')
    first={r['index']:r for r in a}; second={r['index']:r for r in b}
    if set(first)!=set(second): raise ValueError('Matched schedules differ')
    differing=[i for i in first if any(first[i].get(k)!=second[i].get(k) for k in fields)]
    return dict(games=len(first),matching=len(first)-len(differing),
                differing_indices=differing, fields=list(fields))


async def run(args):
    from examples.train_standard_ppo import source_hashes
    from patchwork_automaton.neural.devices import configure_runtime, device_scope
    from patchwork_automaton.neural.inference_processes import weights_digest
    from patchwork_automaton.neural.standard_policy import StandardPolicy
    from patchwork_automaton.rl.standard_ppo import schedule, validate_collection
    from patchwork_automaton.rl.standard_selfplay_processes import StandardSelfplayCollector
    root=args.output.resolve(); root.mkdir(parents=True,exist_ok=False)
    checkpoint=args.checkpoint.resolve(strict=True); checkpoint_sha=sha(checkpoint)
    sources=source_hashes()
    candidate='python_repeat' if args.baseline_repeat else 'direct'
    manifest=dict(status='running', checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        source_sha256=sources, driver_sha256=sha(Path(__file__)), games_per_phase=args.games_per_phase,
        pairs=args.pairs, seed=args.seed, workers=48, runners=2, opponent_replicas=2,
        configurations={'python': 'python',candidate: 'python' if args.baseline_repeat else 'direct'}, baseline_repeat=args.baseline_repeat, native_library_sha256=sha(Path(os.environ['DIRECT_STANDARD_LIBRARY'])),
        phase_order=[f'python-{candidate}' if i%2 else f'{candidate}-python' for i in range(1,args.pairs+1)], cpu_model=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),None),
        logical_cpus=os.cpu_count(), phases=[], comparisons=[])
    save(root/'manifest.json',manifest)
    configure_runtime('cpu',hide_gpu=True)
    with device_scope('cpu'): model=StandardPolicy.load(checkpoint).model
    model_digest=weights_digest(model.get_weights())

    async def phase(label, mode, jobs, run_number):
        os.environ['PATCHWORK_OBSERVATION_PIPELINE'] = 'python'
        out=root/f'pair-{run_number:02d}'/label; out.mkdir(parents=True)
        (out/'trajectories').mkdir(); (out/'diagnostics').mkdir()
        cs=[StandardSelfplayCollector(model,model,workers=24,inference_processes=1,
            opponent_inference_processes=1) for _ in range(2)]
        for collector in cs:
            collector.pool._initializer = initialize
            collector.pool._initargs = (*collector.pool._initargs, mode == 'direct')
        rows=[]
        try:
            await asyncio.gather(*(c.start(version=250,opponent_version=250) for c in cs))
            if any(c.policy_digest!=model_digest or c.opponent_digest!=model_digest for c in cs):
                raise RuntimeError('Inference replica weights mismatch')
            async def batch(games, callback=None):
                await asyncio.gather(*(c.run_schedule([dict(seed=g['seed'],index=g['index'],game=g) for g in games[i::2]],
                    progress=callback,game_runner='standard_ppo',output=out/'trajectories',
                    diagnostics=out/'diagnostics',game_timeout=60,max_turns=50,
                    opponent_version=250,opponent_sha256=checkpoint_sha,greedy=False)
                    for i,c in enumerate(cs)))
            warm=schedule(240,seed=args.seed+1000000+run_number*10000,round_id=1)[:48]
            for game in warm:
                game.update(opponent_kind='frozen_neural',opponent_version=250,opponent_checkpoint_sha256=checkpoint_sha)
            await batch(warm)
            before=[c.stats() for c in cs]
            start=time.perf_counter(); coord=time.process_time()
            with (out/'games.jsonl').open('w') as stream:
                def progress(row):
                    rows.append(row)
                    stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                    if len(rows)%100==0:
                        stream.flush()
                        save(root/'progress.json',dict(pair=run_number,configuration=label,
                            completed=len(rows),total=len(jobs),seconds=time.perf_counter()-start))
                await batch(jobs,progress)
            elapsed=time.perf_counter()-start; coord=time.process_time()-coord
            validate_collection(rows,jobs,250)
            after=[c.stats() for c in cs]
            result=dict(pair=run_number,configuration=label,pipeline_mode=mode,games=len(rows),wall_seconds=elapsed,
                games_per_minute=60*len(rows)/elapsed,decisions=sum(r['decisions'] for r in rows),
                outcomes=dict(Counter(r['status'] for r in rows)),
                draw_reasons=dict(Counter(r.get('reason') for r in rows if r['status']=='draw')),
                parent_worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in rows),coordinator_cpu_seconds=coord,
                replica_before=before,replica_after=after)
            for role in ('learner','opponent'):
                for key in ('replica_cpu_seconds','inference_seconds','requests','calls'):
                    result[role+'_'+key]=sum(b[role][key]-a[role][key] for a,b in zip(before,after))
                requests=result[role+'_requests']
                result[role+'_mean_queue_ms']=sum(b[role]['requests']*b[role]['mean_queue_ms']-
                    a[role]['requests']*a[role]['mean_queue_ms'] for a,b in zip(before,after))/max(requests,1)
            save(out/'summary.json',result)
            print(json.dumps(dict(event='phase_complete',pair=run_number,configuration=label,
                wall_seconds=elapsed,games_per_minute=result['games_per_minute'],outcomes=result['outcomes'])),flush=True)
            return rows,result
        finally:
            await asyncio.gather(*(c.close() for c in cs))

    try:
        for run_number in range(1,args.pairs+1):
            jobs=schedule(1000,seed=args.seed,round_id=run_number)[:args.games_per_phase]
            for game in jobs:
                game.update(opponent_kind='frozen_neural',opponent_version=250,opponent_checkpoint_sha256=checkpoint_sha)
            labels=('python',candidate) if run_number%2 else (candidate,'python')
            rows={}; results={}
            for label in labels:
                rows[label],results[label]=await phase(label,manifest['configurations'][label],jobs,run_number)
                manifest['phases'].append(results[label]); save(root/'manifest.json',manifest)
            parity=compare(rows['python'],rows[candidate])
            trajectory=compare_trajectories(rows['python'],rows[candidate])
            manifest['comparisons'].append(dict(pair=run_number,trajectory=trajectory,**parity))
            print(json.dumps(dict(event='pair_checked',pair=run_number,game_match=parity['matching'],trajectory=trajectory)),flush=True)
            save(root/'manifest.json',manifest)
        if sha(checkpoint)!=checkpoint_sha or source_hashes()!=sources or sha(Path(os.environ['DIRECT_STANDARD_LIBRARY'])) != manifest['native_library_sha256']:
            raise RuntimeError('Frozen checkpoint or source changed')
        manifest['status']='complete';save(root/'manifest.json',manifest)
    except BaseException as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}')
        save(root/'manifest.json',manifest);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=1022610310000);p.add_argument('--pairs',type=int,default=2)
    p.add_argument('--games-per-phase',type=int,default=1000)
    p.add_argument('--baseline-repeat',action='store_true',help='Run Python versus Python to quantify inference nondeterminism')
    a=p.parse_args()
    if a.pairs not in (1,2) or not 48<=a.games_per_phase<=1000:
        p.error('One or two pairs, with 48–1000 games per phase, required')
    asyncio.run(run(a))
if __name__=='__main__':main()
