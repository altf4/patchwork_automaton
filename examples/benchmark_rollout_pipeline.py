"""Sequential CPU-only full-session cache/storage comparisons.

One frozen-policy collection round keeps the self-play workload identical.
Full wall time includes imports, startup, evaluation, collection, PPO updates,
validation, checkpointing and process shutdown. Do not run competing benchmarks.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def tree_rss(pid):
    """Sample summed resident bytes; shared pages count in each process."""
    pending, seen, rss = [pid], set(), 0
    while pending:
        current = pending.pop()
        if current in seen: continue
        seen.add(current)
        try:
            proc = Path('/proc')/str(current)
            rss += int((proc/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
            for thread in (proc/'task').iterdir():
                try: pending.extend(map(int,(thread/'children').read_text().split()))
                except (FileNotFoundError, ProcessLookupError): pass
        except (FileNotFoundError, ProcessLookupError): pass
    return rss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+', default=['256:pickle','512:pickle','1024:pickle','1024:pickle','512:pickle','256:pickle'],
                   help='Ordered CACHE_MIB:STORAGE pairs, e.g. 512:pickle 512:indexed')
    p.add_argument('--games', type=int, default=64)
    p.add_argument('--seed', type=int, default=721)
    args = p.parse_args()
    cases = []
    for value in args.cases:
        cache,storage=value.split(':');cache=int(cache)
        if cache < 0 or storage not in ('pickle','indexed'): p.error('Invalid cache/storage case')
        cases.append((cache,storage))
    root=Path(__file__).resolve().parents[1]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    sources=list((root/'src/patchwork_automaton/neural').glob('*.py'))
    sources+=list((root/'src/patchwork_automaton/rl').glob('rustmage*.py'))
    sources+=[root/'src/patchwork_automaton/backends/rustmage.py',root/'examples/train_rustmage.py',Path(__file__).resolve()]
    hashes={str(p.relative_to(root)):sha(p) for p in sources}
    library=root/'.local/rustmage/target/release/librustmage.so'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',KERAS_BACKEND='tensorflow',
        PATCHWORK_OBSERVATION_PIPELINE='python',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='4',TF_ENABLE_ONEDNN_OPTS='0',
        MPLCONFIGDIR=str(root/'.local/matplotlib'),KERAS_HOME=str(root/'.local/keras'))
    report=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),methodology=__doc__,
        config=dict(games=args.games,seed=args.seed,cases=args.cases),platform=platform.platform(),
        cpu=subprocess.check_output(['lscpu'],text=True),packages={k:version(k) for k in ('tensorflow-cpu','keras','numpy','orjson')},
        runtime={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','TF_ENABLE_ONEDNN_OPTS')},
        source_sha256=hashes,engine_sha256=sha(library),memory_metric='1-second sampled sum of process-tree RSS; shared pages counted repeatedly',trials=[])
    atomic(output/'summary.json',report)
    for index,(cache,storage) in enumerate(cases,1):
        name=f'{index:02d}-{cache}-{storage}';directory=output/name
        command=[sys.executable,'-m','examples.train_rustmage','--output',str(directory),'--rounds','1',
            '--games-per-round',str(args.games),'--eval-games','8','--workers','8','--inference-processes','2',
            '--seed',str(args.seed),'--collector','processes','--observation-pipeline','python','--architecture','separate',
            '--train-batch','32','--ppo-epochs','2','--validation-fraction','.2','--concurrent-updates',
            '--prefetch-batches','2','--measurement-workers','4','--value-max-steps','500','--rollout-cache-mib',str(cache)]
        if storage!='pickle':command+=['--rollout-storage',storage]
        print(json.dumps(dict(stage='starting',trial=index,cache_mib=cache,storage=storage)),flush=True)
        before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.perf_counter();peak=[0];stop=threading.Event()
        with (output/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            def sample_memory():
                while not stop.is_set():
                    peak[0]=max(peak[0],tree_rss(child.pid))
                    stop.wait(1)
            sampler=threading.Thread(target=sample_memory,name='benchmark-rss');sampler.start()
            try:
                child.wait(timeout=1800)
                elapsed=time.perf_counter()-started
            except BaseException:
                child.send_signal(2)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                report.update(status='interrupted',failed_trial=index);atomic(output/'summary.json',report)
                raise
            finally:
                stop.set();sampler.join()
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        if child.returncode:
            report.update(status='failed',failed_trial=index,exit_code=child.returncode);atomic(output/'summary.json',report)
            raise RuntimeError(f'Training failed: {output/(name+".log")}')
        training=json.loads((directory/'report.json').read_text());rounds=training['rounds']
        if training['status']!='complete':raise ValueError('Incomplete training')
        games=[g for r in rounds for g in r['games']]
        if any(g['status'] in ('error','cutoff') for g in games):raise ValueError('Nonterminal self-play invalidates workload comparison')
        fields=('seed','index','status','decisions','neural_choices','actions','final_life','winner')
        records=[{k:g.get(k) for k in fields} for g in games]
        trial=dict(trial=index,case=f'{cache}:{storage}',cache_mib=cache,storage=storage,
            full_wall_seconds=elapsed,training_loop_seconds=training['seconds_this_invocation'],
            collection_seconds=sum(r['collection_seconds'] for r in rounds),update_seconds=sum(r['update_seconds'] for r in rounds),
            validation_seconds=sum(r['validation_seconds'] for r in rounds),
            process_tree_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
            peak_sum_rss_bytes=peak[0],initial_weights_sha256=rounds[0]['collection_policy']['weights_sha256'],
            primary_rollout_bytes=sum(p.stat().st_size for p in (directory/'rollouts').iterdir() if p.is_file()),
            game_records_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
            decisions=sum(g['decisions'] for g in games),neural_choices=sum(g['neural_choices'] for g in games),
            updates=[{k:r['update'].get(k) for k in ('samples','accepted_steps','critic_steps','rejected_epochs','exact_kl','max_state_kl','cache','storage')} for r in rounds],
            report=str(directory/'report.json'),report_sha256=sha(directory/'report.json'))
        report['trials'].append(trial);atomic(output/'summary.json',report)
        print(json.dumps(dict(stage='completed',**trial)),flush=True)
    for path,digest in hashes.items():
        if sha(root/path)!=digest:raise ValueError('Source changed during timing: '+path)
    if sha(library)!=report['engine_sha256']:raise ValueError('Engine library changed')
    for field in ('initial_weights_sha256','game_records_sha256','decisions','neural_choices'):
        if len({t[field] for t in report['trials']})!=1:raise ValueError('Mismatched workloads: '+field)
    for field in ('samples','accepted_steps','critic_steps','rejected_epochs'):
        if len({tuple(r[field] for r in t['updates']) for t in report['trials']})!=1:raise ValueError('Mismatched updates: '+field)
    metrics=('full_wall_seconds','training_loop_seconds','collection_seconds','update_seconds','validation_seconds','process_tree_cpu_seconds','peak_sum_rss_bytes')
    report['medians']={case:{k:statistics.median(t[k] for t in report['trials'] if t['case']==case) for k in metrics} for case in dict.fromkeys(args.cases)}
    base=report['medians'][args.cases[0]]
    report['time_reduction_percent']={case:{k:100*(1-values[k]/base[k]) for k in ('full_wall_seconds','update_seconds')} for case,values in report['medians'].items()}
    report.update(status='complete',identical_workload=True,finished_utc=datetime.now(timezone.utc).isoformat());atomic(output/'summary.json',report)
    print(json.dumps(dict(stage='result',medians=report['medians'],time_reduction_percent=report['time_reduction_percent'])),flush=True)


if __name__=='__main__':main()
