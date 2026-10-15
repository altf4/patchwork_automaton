"""Replay independent synthetic puzzles and identify the first teacher deviation.

Teacher labels are compared after a network action; they never select an action.
This is diagnostic evaluation, with no optimizer or model updates.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import numpy as np
from patchwork_automaton.neural.curriculum import TASKS,ALL_TASKS,make_episode
from patchwork_automaton.neural.policy import NeuralPolicy
from patchwork_automaton.neural.batching import InferenceBatcher
from patchwork_automaton.backends.xmage_prompts import prompt_metadata
from patchwork_automaton.neural.experiment import atomic_json
from examples.train_curriculum import PuzzlePolicy


async def run(args):
    model=NeuralPolicy.load(args.load).model
    batcher=await InferenceBatcher(model,max_batch=8,wait_ms=2).start()
    executor=ThreadPoolExecutor(max_workers=8)
    jobs=asyncio.Queue()
    for i in range(args.cases*len(args.tasks)):jobs.put_nowait(i)
    results=[]
    async def worker():
        while not jobs.empty():
            index=jobs.get_nowait();task=args.tasks[index%len(args.tasks)];seed=args.seed+index*7919
            episode=make_episode(task,np.random.default_rng(seed),heldout=True)
            policy=PuzzlePolicy(model,batcher,version=0,seed=seed+1,mode=args.mode)
            first=None
            while not episode.terminal and episode.steps<32:
                obs=episode.observation()
                expected=episode.oracle_options() if first is None else set()
                action=await asyncio.get_running_loop().run_in_executor(executor,policy.act,obs)
                if first is None and action.response.option not in expected:
                    first=dict(step=episode.steps+1,purpose=prompt_metadata(obs.decision.kind,obs.decision.native)['purpose'],
                        chosen=action.response.option,expected=sorted(expected),
                        options=[dict(id=o.id,label=o.label) for o in obs.decision.options],
                        hand=[o.name for o in obs.objects.values() if o.zone=='hand'],
                        hand_size=next(p.hand_count for p in obs.players if p.id==obs.actor))
                episode.step(action.response)
            results.append(dict(task=task,seed=seed,branch=episode.branch,steps=episode.steps,
                success=episode.terminal and episode.reward==1.,timeout=not episode.terminal,first_deviation=first))
    workers=[asyncio.create_task(worker()) for _ in range(8)]
    try:await asyncio.gather(*workers)
    finally:
        for task in workers:task.cancel()
        await asyncio.gather(*workers,return_exceptions=True)
        await asyncio.to_thread(executor.shutdown,wait=True)
        await batcher.close()
    summary={}
    for row in results:
        g=summary.setdefault(row['task'],dict(cases=0,wins=0,timeouts=0,first_failure_purpose={}))
        g['cases']+=1;g['wins']+=row['success'];g['timeouts']+=row['timeout']
        if not row['success']:
            purpose=(row['first_deviation'] or {}).get('purpose','timeout')
            g['first_failure_purpose'][purpose]=g['first_failure_purpose'].get(purpose,0)+1
    report=dict(checkpoint=str(args.load),mode=args.mode,seed=args.seed,optimizer_steps=0,
        summary=summary,episodes=results)
    atomic_json(args.output,report);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--load',type=Path,required=True)
    p.add_argument('--tasks',nargs='+',choices=ALL_TASKS,default=list(TASKS))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',type=int,default=100)
    p.add_argument('--seed',type=int,default=180001)
    p.add_argument('--mode',choices=('greedy','sampled'),default='greedy')
    args=p.parse_args()
    if args.cases<1:p.error('Cases must be positive')
    asyncio.run(run(args))
