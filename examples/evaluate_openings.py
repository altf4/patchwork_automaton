"""Exercise saved policies through complete synthetic bottom-card sequences."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('KERAS_BACKEND','tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import argparse
import json
from pathlib import Path
import numpy as np
from patchwork_automaton.neural.policy import NeuralPolicy
from patchwork_automaton.neural.scenarios import sample,advance_bottom


class GreedySampler:
    def choice(self, count, *, p): return int(np.argmax(p))


def evaluate(path, *, seed=501, episodes=100):
    policy=NeuralPolicy.load(path,seed=seed)
    result=dict(checkpoint=str(path),optimizer_steps=0,runs=[])
    for heldout in (False,True):
        for greedy in (False,True):
            policy.rng=GreedySampler() if greedy else np.random.default_rng(seed+1)
            rng=np.random.default_rng(seed+int(heldout));branches={}
            for _ in range(episodes):
                scenario=sample('mulligan_bottom',rng,heldout=heldout)
                branch=branches.setdefault(scenario.variant,dict(episodes=0,wins=0,decisions=0))
                policy.reset();success=True
                while scenario is not None:
                    record=policy.act(scenario.observation)
                    success &= record.response.option in scenario.winning_options
                    branch['decisions']+=1
                    scenario=advance_bottom(scenario,record.response.option)
                branch['episodes']+=1;branch['wins']+=int(success)
            for branch in branches.values():branch['success_rate']=branch['wins']/branch['episodes']
            result['runs'].append(dict(heldout=heldout,greedy=greedy,branches=branches))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seed',type=int,default=501)
    parser.add_argument('--episodes',type=int,default=100)
    args=parser.parse_args()
    if args.episodes<1:parser.error('Episodes must be positive')
    result=evaluate(args.load,seed=args.seed,episodes=args.episodes)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
