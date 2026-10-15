"""Batched CPU PPO on scripted multi-decision Burn tasks; terminal task rewards."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('KERAS_BACKEND','tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from patchwork_automaton.neural.policy import NeuralPolicy,untrained_policy
from patchwork_automaton.neural.batching import BatchedPolicy,InferenceBatcher
from patchwork_automaton.neural.experiment import TrajectoryWriter,make_trainer,save_checkpoint,atomic_json
from patchwork_automaton.neural.scenarios import TASKS
from patchwork_automaton.neural.episodes import ScenarioEpisode,CastBoltEpisode

EPISODE_TASKS=(*TASKS,'pay_bolt','pay_charm','cast_bolt','pass_unpayable')


async def collect(count,phase,model,batcher,executor,args,rng,*,heldout=False):
    jobs=asyncio.Queue()
    for i in range(count):jobs.put_nowait((i,int(rng.integers(0,2**31-1))))
    results=[]
    async def worker():
        while not jobs.empty():
            index,seed=jobs.get_nowait();task=args.tasks[index%len(args.tasks)]
            local=np.random.default_rng(seed)
            episode=(CastBoltEpisode(local,heldout=heldout,unpayable=task=='pass_unpayable',payment_only=task in ('pay_bolt','pay_charm'),
                                     mana_color=str(local.choice(['red','white'])) if task=='pay_charm' else 'red',charm=task=='pay_charm')
                     if task in ('pay_bolt','pay_charm','cast_bolt','pass_unpayable') else ScenarioEpisode(task,local,heldout=heldout))
            branch=episode.scenario.variant if isinstance(episode,ScenarioEpisode) else task+':'+episode.wanted+':'+episode.land_name
            writer=TrajectoryWriter(args.directory/'rollouts'/phase/f'{index:05d}.pkl.gz')
            policy=BatchedPolicy(model,batcher,version=batcher.version,seed=seed+1,sink=writer)
            try:
                while not episode.terminal:
                    if episode.steps>=16:raise RuntimeError('Synthetic episode exceeded its decision bound')
                    obs=episode.observation()
                    record=await asyncio.get_running_loop().run_in_executor(executor,policy.act,obs)
                    reward=episode.step(record.response)
                    if not episode.terminal and reward is not None:raise RuntimeError('Intermediate reward is forbidden')
                results.append(dict(task=task,branch=branch,seed=seed,steps=episode.steps,
                    path=str(writer.path),samples=writer.count,reward=episode.reward))
            finally:writer.close()
    await asyncio.gather(*(worker() for _ in range(min(args.workers,count))))
    return results


def summary(results):
    groups={}
    for row in results:
        item=groups.setdefault(row['branch'],dict(episodes=0,wins=0,decisions=0))
        item['episodes']+=1;item['wins']+=row['reward']>0;item['decisions']+=row['steps']
    for item in groups.values():item['success_rate']=item['wins']/item['episodes']
    return groups


async def run(args):
    args.directory.mkdir(parents=True,exist_ok=False)
    policy=NeuralPolicy.load(args.load,seed=args.seed) if args.load else untrained_policy(seed=args.seed,
        candidate_interactions=True,player_readout=True,stable_heads=True,fact_readout=True)
    model=policy.model
    config=dict(algorithm='ppo',seed=args.seed,learning_rate=args.learning_rate,train_batch=args.train_batch,
                ppo_epochs=args.ppo_epochs,target_kl=.01,max_state_kl=.05,advantage_scale_floor=1.,
                tasks=args.tasks,load=str(args.load) if args.load else None,workers=args.workers)
    trainer=make_trainer(model,config);trainer.compile()
    batcher=await InferenceBatcher(model,max_batch=args.workers,wait_ms=2).start()
    executor=ThreadPoolExecutor(max_workers=args.workers)
    state=dict(round=0,completed_games=0,config=config,objective='synthetic terminal task success',rounds=[])
    rng=np.random.default_rng(args.seed)
    try:
        async def evaluate(label):
            results=await collect(args.eval_cases*len(args.tasks),label,model,batcher,executor,args,
                                  np.random.default_rng(args.seed+90000),heldout=True)
            result=dict(objective=state['objective'],summary=summary(results),episodes=results,model_version=batcher.version)
            atomic_json(args.directory/f'evaluation-{label}.json',result)
            print(json.dumps(dict(evaluation=label,summary=result['summary'])),flush=True)
            return result['summary']
        state['initial']=await evaluate('initial')
        for update in range(args.updates):
            phase=f'round-{update+1:04d}'
            trajectories=await collect(args.round_episodes,phase,model,batcher,executor,args,rng)
            metrics=await asyncio.to_thread(trainer.update_files,trajectories)
            state['round']+=1;state['completed_games']+=len(trajectories)
            state['rng_state']=rng.bit_generator.state
            state['rounds'].append(dict(round=state['round'],summary=summary(trajectories),update=metrics))
            batcher.set_version(state['round'])
            if (update+1)%10==0 or update==0:
                atomic_json(args.directory/'progress.json',state)
                print(json.dumps(state['rounds'][-1]),flush=True)
            shutil.rmtree(args.directory/'rollouts'/phase)
        await asyncio.to_thread(save_checkpoint,args.directory/'checkpoints',model,trainer,state)
        state['final']=await evaluate('final')
        state['optimizer_steps']=int(trainer.optimizer.iterations)
        state['inference']=batcher.stats()
        state['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path('src/patchwork_automaton/neural/episodes.py'))}
        atomic_json(args.directory/'report.json',state)
    finally:
        await asyncio.to_thread(executor.shutdown,wait=True)
        await batcher.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--load',type=Path)
    parser.add_argument('--tasks',choices=EPISODE_TASKS,nargs='+',default=list(EPISODE_TASKS))
    parser.add_argument('--seed',type=int,default=301)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--updates',type=int,default=100)
    parser.add_argument('--round-episodes',type=int,default=64)
    parser.add_argument('--eval-cases',type=int,default=100)
    parser.add_argument('--learning-rate',type=float,default=.0001)
    parser.add_argument('--train-batch',type=int,default=48)
    parser.add_argument('--ppo-epochs',type=int,default=2)
    args=parser.parse_args()
    if min(args.workers,args.round_episodes,args.eval_cases,args.train_batch,args.ppo_epochs,args.learning_rate)<=0 or args.updates<0:
        parser.error('Invalid counts or learning rate')
    asyncio.run(run(args))
