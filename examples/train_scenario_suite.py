"""Run a fixed CPU scenario recipe across independent clean and noisy seeds.

Each worker is a separate uv-launched Python process. No policy weights or
random streams are shared between seeds, and failed gates remain in the report.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess


def write_json(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def run(args):
    args.directory.mkdir(parents=True,exist_ok=False)
    recipe=['--algorithm','ppo','--candidate-interactions','--player-readout','--fact-readout',
            '--stable-heads','--standardize-inputs','--trust-guard',
            '--learning-rate','.0003','--advantage-scale-floor',str(args.advantage_scale_floor),
            '--target-kl','.01','--max-state-kl','.05','--batch','96',
            '--updates',str(args.updates),'--test-cases','500',
            '--distribution',args.distribution,'--identity-dropout',str(args.identity_dropout),
            '--retention-checks','3','--require-pass']
    if args.isolate_value:recipe.append('--isolate-value')
    if args.final_learning_rate is not None:recipe.extend(['--final-learning-rate',str(args.final_learning_rate)])
    jobs=[('clean',seed,0.) for seed in args.clean_seeds]+[('noisy',seed,.2) for seed in args.noisy_seeds]
    report=dict(recipe=recipe,workers=args.workers,jobs=[dict(kind=k,seed=s,outcome_noise=n) for k,s,n in jobs],runs=[])
    write_json(args.directory/'plan.json',report)
    uv=str(args.uv.resolve())
    def execute(job):
        kind,seed,noise=job;name=f'{kind}-seed-{seed}'
        directory=args.directory/name
        command=[uv,'run','--no-sync','python','-m','examples.train_scenarios',*recipe,
                 '--seeds',str(seed),'--outcome-noise',str(noise),'--directory',str(directory)]
        environment=os.environ|{'CUDA_VISIBLE_DEVICES':'-1','TF_NUM_INTRAOP_THREADS':'2',
                                'TF_NUM_INTEROP_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                                'TF_ENABLE_ONEDNN_OPTS':'0'}
        with (args.directory/f'{name}.log').open('w') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=environment,check=False)
        result=dict(kind=kind,seed=seed,outcome_noise=noise,returncode=completed.returncode,source=str(directory/'report.json'))
        if (directory/'report.json').exists():
            result['report']=json.loads((directory/'report.json').read_text())
            result['passed']=completed.returncode==0 and all(r['passed'] for r in result['report']['runs'])
        else:
            result.update(passed=False,error='No complete report; inspect the case log')
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(execute,job):job for job in jobs}
        for future in as_completed(pending):
            try:result=future.result()
            except Exception as error:
                kind,seed,noise=pending[future]
                result=dict(kind=kind,seed=seed,outcome_noise=noise,passed=False,error=str(error))
            report['runs'].append(result)
            report['runs'].sort(key=lambda r:(r['kind'],r['seed']))
            write_json(args.directory/'report.json',report)
            print(json.dumps({k:result[k] for k in ('kind','seed','passed')}),flush=True)
    report['passed']=len(report['runs'])==len(jobs) and all(r['passed'] for r in report['runs'])
    write_json(args.directory/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--uv',type=Path,default=Path('.local/uv/uv'))
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--updates',type=int,default=600)
    parser.add_argument('--advantage-scale-floor',type=float,default=1.)
    parser.add_argument('--distribution',choices=('legacy','diverse','native'),default='diverse')
    parser.add_argument('--identity-dropout',type=float,default=.5)
    parser.add_argument('--isolate-value',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--final-learning-rate',type=float,default=.000003)
    parser.add_argument('--clean-seeds',type=int,nargs='+',default=[81,82,83,84,85])
    parser.add_argument('--noisy-seeds',type=int,nargs='+',default=[91,92,93])
    args=parser.parse_args()
    if args.final_learning_rate is not None and not 0<args.final_learning_rate<=.0003:parser.error('Final learning rate must be in (0, .0003]')
    if not 0<=args.identity_dropout<=1:parser.error('Identity dropout must be in [0, 1]')
    if args.advantage_scale_floor<=0:parser.error('Advantage scale floor must be positive')
    if args.workers<1 or args.updates<300:parser.error('Use positive workers and at least 300 updates for retention checks')
    if any(len(set(seeds))!=len(seeds) for seeds in (args.clean_seeds,args.noisy_seeds)):
        parser.error('Seeds within each condition must be unique')
    if not run(args)['passed']:raise SystemExit(1)
