"""Run matched complete CPU training sessions in an alternating pipeline order.

Includes startup, baseline/before/after evaluation, self-play, held-out validation,
PPO updates, checkpoint writing, model synchronization and shutdown in wall time.
Runs are sequential; do not compile or run other benchmarks alongside them.
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
import time


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--order', nargs='+', choices=('python','rust'), default=['python','rust','rust','python'])
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--games-per-round', type=int, default=32)
    p.add_argument('--eval-games', type=int, default=8)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--inference-processes', type=int, default=2)
    p.add_argument('--seed', type=int, default=721)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    sources = list((root/'native/observation-pipeline/src').glob('*.rs'))
    sources += list((root/'src/patchwork_automaton/neural').glob('*.py'))
    sources += list((root/'src/patchwork_automaton/rl').glob('rustmage*.py'))
    sources += [root/'src/patchwork_automaton/backends/rustmage.py', root/'examples/train_rustmage.py', Path(__file__).resolve(),
                root/'native/observation-pipeline/Cargo.toml', root/'native/observation-pipeline/Cargo.lock']
    hashes = {str(path.relative_to(root)): sha(path) for path in sources}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', KERAS_BACKEND='tensorflow',
               TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='4', TF_ENABLE_ONEDNN_OPTS='0',
               MPLCONFIGDIR=str(root/'.local/matplotlib'), KERAS_HOME=str(root/'.local/keras'))
    report = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  methodology=__doc__, config={k:v for k,v in vars(args).items() if k!='output'},
                  platform=platform.platform(), cpu=subprocess.check_output(['lscpu'], text=True),
                  packages={name:version(name) for name in ('numpy','keras','tensorflow-cpu','orjson')},
                  source_sha256=hashes,
                  library_sha256={name:sha(root/name) for name in (
                      '.local/rustmage/target/release/librustmage.so',
                      '.local/observation-target/release/libpatchwork_observation_pipeline.so')},
                  runtime={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','TF_ENABLE_ONEDNN_OPTS')},
                  trials=[])
    atomic(output/'summary.json', report)
    for index, mode in enumerate(args.order):
        run_dir = output/f'{index+1:02d}-{mode}'
        command = [sys.executable, '-m', 'examples.train_rustmage', '--output', str(run_dir),
                   '--observation-pipeline', mode, '--rounds', str(args.rounds),
                   '--games-per-round', str(args.games_per_round), '--eval-games', str(args.eval_games),
                   '--workers', str(args.workers), '--inference-processes', str(args.inference_processes),
                   '--seed', str(args.seed), '--collector', 'processes', '--architecture', 'separate',
                   '--train-batch', '32', '--ppo-epochs', '2', '--validation-fraction', '.2',
                   '--concurrent-updates', '--prefetch-batches', '2', '--measurement-workers', '4',
                   '--value-max-steps', '500']
        print(json.dumps(dict(stage='starting', trial=index+1, pipeline=mode, command=command)), flush=True)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.perf_counter()
        with (output/f'{index+1:02d}-{mode}.log').open('w') as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        elapsed = time.perf_counter()-started
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        if result.returncode:
            report.update(status='failed', failed_trial=index+1, exit_code=result.returncode)
            atomic(output/'summary.json', report)
            raise RuntimeError(f'Training failed; inspect {output}/{index+1:02d}-{mode}.log')
        training = json.loads((run_dir/'report.json').read_text())
        if training['status'] != 'complete': raise ValueError('Incomplete training session')
        rows = [g for r in training['rounds'] for g in r['games']]
        if any(g['status']=='error' for g in rows): raise ValueError('Collection errors invalidate the benchmark')
        behavior = [{k:g.get(k) for k in ('seed','index','status','decisions','neural_choices','actions','final_life','winner')}
                    for g in rows]
        trial = dict(pipeline=mode, trial=index+1, full_wall_seconds=elapsed,
                     process_tree_cpu_seconds=(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime),
                     initial_weights_sha256=training['rounds'][0]['collection_policy']['weights_sha256'],
                     game_records_sha256=hashlib.sha256(json.dumps(behavior,sort_keys=True).encode()).hexdigest(),
                     report=str(run_dir/'report.json'), training_loop_seconds=training['seconds_this_invocation'],
                     collection_seconds=sum(r['collection_seconds'] for r in training['rounds']),
                     update_seconds=sum(r['update_seconds'] for r in training['rounds']),
                     validation_seconds=sum(r['validation_seconds'] for r in training['rounds']),
                     collection_worker_cpu_seconds=sum(g.get('worker_cpu_seconds',0) for g in rows),
                     outcomes={s:sum(g['status']==s for g in rows) for s in ('win','loss','draw','cutoff','error')},
                     decisions=sum(g['decisions'] for g in rows), neural_choices=sum(g['neural_choices'] for g in rows),
                     updates=[{k:r['update'].get(k) for k in ('samples','accepted_steps','critic_steps','rejected_epochs','exact_kl','max_state_kl')}
                              for r in training['rounds']], throughput=training['throughput'])
        report['trials'].append(trial)
        atomic(output/'summary.json', report)
        print(json.dumps(dict(stage='completed', **trial)), flush=True)
    for path, expected in hashes.items():
        if sha(root/path) != expected: raise ValueError('Sources changed during timing: '+path)
    for path, expected in report['library_sha256'].items():
        if sha(root/path) != expected: raise ValueError('Library changed during timing: '+path)
    if len({t['initial_weights_sha256'] for t in report['trials']}) != 1:
        raise ValueError('Initial weights differ')
    report['identical_game_records'] = len({t['game_records_sha256'] for t in report['trials']}) == 1
    metrics = ('full_wall_seconds','training_loop_seconds','collection_seconds','update_seconds',
               'validation_seconds','process_tree_cpu_seconds','collection_worker_cpu_seconds')
    report['medians'] = {mode:{key:statistics.median(t[key] for t in report['trials'] if t['pipeline']==mode)
                              for key in metrics} for mode in set(args.order)}
    if set(report['medians']) == {'python','rust'}:
        report['speedups'] = {key:report['medians']['python'][key]/report['medians']['rust'][key] for key in metrics}
    report.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    atomic(output/'summary.json', report)
    print(json.dumps(dict(stage='result', **{k:report[k] for k in ('medians','speedups','identical_game_records') if k in report})), flush=True)


if __name__ == '__main__': main()
