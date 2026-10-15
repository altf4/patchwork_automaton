"""Reproduce a completed scenario suite from saved models, without training."""
from examples.train_scenarios import evaluate
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from patchwork_automaton.neural.policy import NeuralPolicy
from patchwork_automaton.neural.scenarios import sample
from patchwork_automaton.neural.batching import example_batch,input_signature


def verify(directory):
    suite=json.loads((directory/'report.json').read_text())
    expected={(r['kind'],r['seed']) for r in suite['jobs']}
    actual={(r['kind'],r['seed']) for r in suite['runs']}
    if actual!=expected:raise ValueError('Suite is not complete')
    current={name:hashlib.sha256(Path(importlib.import_module('patchwork_automaton.neural.'+name).__file__).read_bytes()).hexdigest()
             for name in ('features','model','ppo','scenarios','augmentation','policy','native_layout')}
    current['runner']=hashlib.sha256(Path(importlib.import_module('examples.train_scenarios').__file__).read_bytes()).hexdigest()
    for name in ('baselines.burn_mulligan','backends.xmage_prompts','backends.xmage_choices','backends.burn_actions'):
        current[name]=hashlib.sha256(Path(importlib.import_module('patchwork_automaton.'+name).__file__).read_bytes()).hexdigest()
    current['native_layout_data']=hashlib.sha256(Path('src/patchwork_automaton/data/burn-opening-layout.json').read_bytes()).hexdigest()
    results=[]
    for case in suite['runs']:
        report=case['report'];config=report['config'];run=report['runs'][0];seed=run['seed']
        if report.get('source_sha256')!=current:
            raise ValueError(f'Current sources do not match the frozen run for seed {seed}')
        model_path=Path(config['directory'])/f'seed-{seed}.keras'
        policy=NeuralPolicy.load(model_path);model=policy.model
        for key in ('candidate_interactions','player_readout','stable_heads','standardize_inputs','isolate_value','fact_readout'):
            if getattr(model,key)!=config.get(key,False):raise ValueError(f'Checkpoint flag mismatch: {key}')
        forward=tf.function(model.call,input_signature=[input_signature(example_batch(model.memory_size))],autograph=False)
        verified=0;max_difference=0.
        for split,rng_seed,heldout in [('final',(91000 if config.get('development_only') else 90000)+seed,not config.get('development_only',False)),
                                      ('final_iid',80000+seed,False)]:
            rng=np.random.default_rng(rng_seed)
            states=[sample(task,rng,heldout=heldout,distribution=config.get('distribution','legacy'))
                    for task in config['tasks'] for _ in range(config['test_cases'])]
            reproduced=evaluate(states,policy.encoder,model,forward,outcome_noise=config.get('outcome_noise',0.),runtime_history=config.get('runtime_history',False))
            if reproduced.keys()!=run[split].keys():raise ValueError('Evaluation branches changed')
            for branch,metrics in reproduced.items():
                for key,value in run[split][branch].items():
                    actual_value=metrics[key]
                    np.testing.assert_allclose(actual_value,value,rtol=1e-5,atol=1e-6,err_msg=f'{seed}/{split}/{branch}/{key}')
                    max_difference=max(max_difference,abs(actual_value-value))
            verified+=len(states)
        result=dict(kind=case['kind'],seed=seed,checkpoint=str(model_path),states=verified,
                    max_metric_difference=max_difference,source_hashes_match=True,passed=True)
        results.append(result);print(json.dumps(result),flush=True)
        tf.keras.backend.clear_session()
    return dict(training_updates=0,passed=True,states=sum(r['states'] for r in results),runs=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=verify(args.directory)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
