"""Train five native-layout Burn skills together on CPU.

Warmup is explicitly supervised imitation of synthetic puzzle solutions. The
optional PPO phase uses only episode-terminal success/failure. Evaluation uses
the model alone, with the normal recurrent history and no teacher fallback.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('KERAS_BACKEND','tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import tensorflow as tf
import keras

from patchwork_automaton.neural.curriculum import TASKS,ALL_TASKS,make_episode
from patchwork_automaton.neural.policy import NeuralPolicy,untrained_policy
from patchwork_automaton.neural.features import collate,Encoder
from patchwork_automaton.neural.batching import BatchedPolicy,InferenceBatcher,example_batch,input_signature
from patchwork_automaton.neural.experiment import atomic_json,TrajectoryWriter,make_trainer,save_checkpoint
from patchwork_automaton.backends.xmage_prompts import prompt_metadata
from patchwork_automaton.core import Response
from patchwork_automaton.neural.calibration import calibrate_context


class PuzzlePolicy(BatchedPolicy):
    def __init__(self,*args,mode='sampled',**kwargs):
        super().__init__(*args,**kwargs)
        self.mode=mode;self.targets=set();self.examples=[];self.options=();self.example_group=None

    def _select(self,arrays,state,*,advance,stage,records):
        index,updated,value=super()._select(arrays,state,advance=advance,stage=stage,records=records)
        if self.mode=='teacher':
            if stage!='option':raise ValueError('Curriculum currently uses finite primitive options only')
            targets=np.asarray([o.id in self.targets for o in self.options],bool)
            if not targets.any():raise ValueError('Teacher has no offered solution')
            index=int(self.rng.choice(np.flatnonzero(targets)))
            self.examples.append(({k:v.copy() for k,v in arrays.items()},state.copy(),targets,self.example_group))
        elif self.mode=='greedy':index=int(np.argmax(self.last_output['logits'][0]))
        if self.mode!='sampled':
            logits=self.last_output['logits'][0].astype(np.float64)
            logp=logits-np.max(logits);logp-=np.log(np.exp(logp).sum())
            records[-1]=replace(records[-1],index=index,log_probability=float(logp[index]))
        return index,updated,value


async def collect(args,model,batcher,executor,*,count,seed,mode,heldout=False,phase='eval',write=False):
    jobs=asyncio.Queue()
    for i in range(count):jobs.put_nowait(i)
    results=[];examples=[]
    async def worker():
        while not jobs.empty():
            i=jobs.get_nowait();task=args.tasks[i%len(args.tasks)]
            case_seed=seed+i*7919
            episode=make_episode(task,np.random.default_rng(case_seed),heldout=heldout)
            writer=TrajectoryWriter(args.directory/'rollouts'/phase/f'{i:06d}.pkl.gz') if write else None
            policy=PuzzlePolicy(model,batcher,version=batcher.version,seed=case_seed+1,mode=mode,sink=writer)
            timed_out=False
            try:
                while not episode.terminal:
                    if episode.steps>=32:timed_out=True;break
                    obs=episode.observation()
                    policy.targets=episode.oracle_options() if mode=='teacher' else set()
                    policy.options=obs.decision.options
                    policy.example_group=(task,prompt_metadata(obs.decision.kind,obs.decision.native)['purpose'])
                    if getattr(args,'balance_actions',False):
                        # External teacher-class balancing, never a model input.
                        kinds=tuple(sorted({o.kind for o in obs.decision.options if o.id in policy.targets}))
                        policy.example_group+= (kinds,)
                    action=await asyncio.get_running_loop().run_in_executor(executor,policy.act,obs)
                    reward=episode.step(action.response)
                    if not episode.terminal and reward is not None:raise RuntimeError('Intermediate reward is forbidden')
                # A bounded puzzle timeout is a defined synthetic failure, not
                # a fabricated server outcome or an intermediate reward.
                row=dict(task=task,branch=episode.branch,seed=case_seed,steps=episode.steps,
                    reward=-1. if timed_out else episode.reward,timeout=timed_out)
                if writer:row.update(path=str(writer.path),samples=writer.count)
                results.append(row);examples.extend(policy.examples)
            finally:
                if writer:writer.close()
    workers=[asyncio.create_task(worker()) for _ in range(min(args.workers,count))]
    try:await asyncio.gather(*workers)
    except BaseException:
        for worker_task in workers:worker_task.cancel()
        await asyncio.gather(*workers,return_exceptions=True)
        raise
    return results,examples


def summarize(rows,*,branches=False):
    groups={}
    for r in rows:
        key=r['task']+':'+r['branch'] if branches else r['task']
        g=groups.setdefault(key,dict(episodes=0,wins=0,timeouts=0,decisions=0))
        g['episodes']+=1;g['wins']+=r['reward']==1.;g['timeouts']+=r['timeout'];g['decisions']+=r['steps']
    for g in groups.values():g['success_rate']=g['wins']/g['episodes']
    return groups


def make_supervised_update(model,optimizer):
    signature=[input_signature(example_batch(model.memory_size)),tf.TensorSpec([None,None],tf.bool),tf.TensorSpec([None],tf.float32)]
    @tf.function(input_signature=signature,autograph=False)
    def update(inputs,targets,weights):
        with tf.GradientTape() as tape:
            logp=tf.nn.log_softmax(model(inputs,training=True)['logits'])
            # Equivalent payments, spells and attackers are all correct.
            loss=-tf.reduce_mean(weights*tf.reduce_logsumexp(tf.where(targets,logp,-1.e9),axis=1))
        gradients=tape.gradient(loss,model.trainable_variables)
        gradients=[tf.zeros_like(v) if g is None else tf.convert_to_tensor(g) for g,v in zip(gradients,model.trainable_variables)]
        for g in gradients:tf.debugging.assert_all_finite(g,'Nonfinite curriculum gradient')
        gradients,norm=tf.clip_by_global_norm(gradients,1.)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        return loss,norm
    return update


def demonstration_weights(examples,balance):
    """Equal task weight, then equal prompts within each task; not an RL reward."""
    from collections import Counter
    if not balance:return np.ones(len(examples),np.float32)
    counts=Counter(r[3] for r in examples)
    family=lambda group:group[0] if isinstance(group,tuple) else group
    prompt=lambda group:group[1] if isinstance(group,tuple) else None
    classes=Counter((family(group),prompt(group)) for group in counts)
    prompts=Counter(task for task,_ in classes)
    return np.asarray([len(examples)/(len(prompts)*prompts[family(r[3])]
        *classes[(family(r[3]),prompt(r[3]))]*counts[r[3]]) for r in examples],np.float32)


def fit_examples(examples,model,update,rng,*,batch,epochs,balance=False):
    losses=[]
    weights=demonstration_weights(examples,balance)
    for _ in range(epochs):
        order=rng.permutation(len(examples))
        for start in range(0,len(order),batch):
            chunk=[examples[i] for i in order[start:start+batch]]
            inputs=collate([r[0] for r in chunk])
            inputs['memory_in']=np.stack([r[1] for r in chunk])
            inputs['advance_memory']=np.ones((len(chunk),1),np.float32)
            targets=np.zeros(inputs['candidate_mask'].shape,bool)
            for i,r in enumerate(chunk):targets[i,:len(r[2])]=r[2]
            loss,_=update(inputs,targets,weights[order[start:start+batch]]);losses.append(float(loss))
    return dict(loss=float(np.mean(losses)),examples=len(examples),steps=len(losses))


async def run(args):
    args.directory.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    policy=NeuralPolicy.load(args.load,seed=args.seed) if args.load else untrained_policy(seed=args.seed,
        candidate_interactions=True,player_readout=True,stable_heads=True,fact_readout=True,isolate_value=True)
    model=policy.model
    if args.calibrate_context:
        contexts=[];calibration_rng=np.random.default_rng(args.seed+60000);encoder=Encoder()
        for i in range(160):
            episode=make_episode(args.tasks[i%len(args.tasks)],calibration_rng)
            while not episode.terminal:
                contexts.append(encoder.encode(episode.observation()).arrays['context'])
                episode.step(Response(sorted(episode.oracle_options())[0]))
        model=calibrate_context(model,np.stack(contexts));policy=NeuralPolicy(model,seed=args.seed)
    batcher=await InferenceBatcher(model,max_batch=args.workers,wait_ms=2).start()
    executor=ThreadPoolExecutor(max_workers=args.workers)
    optimizer=keras.optimizers.Adam(args.learning_rate);optimizer.build(model.trainable_variables)
    update=make_supervised_update(model,optimizer)
    rng=np.random.default_rng(args.seed+60000)
    config=vars(args)|dict(directory=str(args.directory),load=str(args.load) if args.load else None)
    modules=('curriculum','episodes','payment_scenarios','scenarios','native_layout','features','model','policy','history','batching','streaming_ppo','calibration')
    source={name:hashlib.sha256(Path(importlib.import_module('patchwork_automaton.neural.'+name).__file__).read_bytes()).hexdigest() for name in modules}
    for name in ('backends.xmage_choices','backends.xmage_prompts','backends.burn_actions','baselines.burn_mulligan','baselines.modern_burn','burn_semantics'):
        source[name]=hashlib.sha256(Path(importlib.import_module('patchwork_automaton.'+name).__file__).read_bytes()).hexdigest()
    source['native_layout_data']=hashlib.sha256(Path('src/patchwork_automaton/data/burn-opening-layout.json').read_bytes()).hexdigest()
    source['runner']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    state=dict(config=config,source_sha256=source,objective='synthetic terminal puzzle success',
        warmup='supervised teacher imitation; not reinforcement learning',rounds=[],evaluations={})
    atomic_json(args.directory/'progress.json',state)
    try:
        async def evaluate(label,*,heldout=True,cases=None):
            result={}
            for mode in ('greedy','sampled'):
                rows,_=await collect(args,model,batcher,executor,count=(cases or args.eval_cases)*len(args.tasks),
                    seed=args.seed+(90000 if heldout else 80000),mode=mode,heldout=heldout,phase=label)
                result[mode]=dict(summary=summarize(rows),branches=summarize(rows,branches=True),episodes=rows)
            atomic_json(args.directory/f'evaluation-{label}.json',result)
            state['evaluations'][label]={mode:part['summary'] for mode,part in result.items()}
            print(json.dumps(dict(evaluation=label,summary=state['evaluations'][label])),flush=True)
            return result
        await evaluate('initial')
        for round_index in range(args.warmup_rounds):
            rows,examples=await collect(args,model,batcher,executor,count=args.round_episodes,
                seed=args.seed+round_index*1000000,mode='teacher',phase='warmup')
            if any(r['reward']!=1. for r in rows):raise RuntimeError('Teacher failed its puzzle')
            metrics=await asyncio.to_thread(fit_examples,examples,model,update,rng,batch=args.train_batch,epochs=args.supervised_epochs,balance=args.balance_prompts or args.balance_actions)
            batcher.set_version(batcher.version+1)
            state['rounds'].append(dict(round=round_index+1,phase='supervised',**metrics))
            if (round_index+1)%5==0 or round_index==0:
                state['seconds']=time.monotonic()-start
                atomic_json(args.directory/'progress.json',state)
                print(json.dumps(dict(**state['rounds'][-1],seconds=state['seconds'])),flush=True)
            if (round_index+1)%args.evaluate_every==0:
                await evaluate(f'warmup-{round_index+1}',cases=min(32,args.eval_cases))
                policy.save(args.directory/'warmup.keras')
        policy.save(args.directory/'warmup.keras')
        await evaluate('warmup')
        if args.ppo_rounds:
            ppo_config=dict(algorithm='ppo',seed=args.seed,learning_rate=args.ppo_learning_rate,
                train_batch=args.train_batch,ppo_epochs=2,target_kl=.01,max_state_kl=.05,advantage_scale_floor=1.)
            trainer=make_trainer(model,ppo_config);trainer.compile()
            for round_index in range(args.ppo_rounds):
                phase=f'ppo-{round_index+1:04d}'
                rows,_=await collect(args,model,batcher,executor,count=args.round_episodes,
                    seed=args.seed+500000000+round_index*1000000,mode='sampled',phase=phase,write=True)
                metrics=await asyncio.to_thread(trainer.update_files,rows)
                batcher.set_version(batcher.version+1)
                state['rounds'].append(dict(round=round_index+1,phase='ppo',summary=summarize(rows),update=metrics))
                atomic_json(args.directory/'progress.json',state)
                print(json.dumps(state['rounds'][-1]),flush=True)
                shutil.rmtree(args.directory/'rollouts'/phase)
            await asyncio.to_thread(save_checkpoint,args.directory/'ppo-checkpoints',model,trainer,
                dict(round=args.ppo_rounds,config=ppo_config,rng_state=rng.bit_generator.state))
        policy.save(args.directory/'policy.keras')
        await evaluate('final')
        await evaluate('final-iid',heldout=False)
        # Verify the saved artifact reproduces actual numerical predictions.
        restored=NeuralPolicy.load(args.directory/'policy.keras').model
        check=example_batch(model.memory_size)
        np.testing.assert_allclose(np.asarray(model(check)['logits']),np.asarray(restored(check)['logits']),atol=1.e-5)
        state.update(seconds=time.monotonic()-start,parameters=model.count_params(),
            supervised_optimizer_steps=int(optimizer.iterations),inference=batcher.stats(),
            checkpoint_sha256=hashlib.sha256((args.directory/'policy.keras').read_bytes()).hexdigest())
        state['passed']=all(g['success_rate']>=args.pass_rate for label in ('final','final-iid')
            for part in state['evaluations'][label].values() for g in part.values())
        atomic_json(args.directory/'report.json',state)
        print(json.dumps(dict(passed=state['passed'],seconds=state['seconds'])),flush=True)
    finally:
        # Outstanding executor calls must still be able to consume inference.
        await asyncio.to_thread(executor.shutdown,wait=True)
        await batcher.close()
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--load',type=Path)
    p.add_argument('--tasks',nargs='+',choices=ALL_TASKS,default=list(TASKS))
    p.add_argument('--seed',type=int,default=401)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--warmup-rounds',type=int,default=80)
    p.add_argument('--ppo-rounds',type=int,default=10)
    p.add_argument('--round-episodes',type=int,default=80)
    p.add_argument('--train-batch',type=int,default=128)
    p.add_argument('--supervised-epochs',type=int,default=2)
    p.add_argument('--balance-prompts',action='store_true',help='Balance tasks, then prompt groups within each task')
    p.add_argument('--balance-actions',action='store_true',help='Also balance winning primitive kinds within each prompt; supervised loss only')
    p.add_argument('--calibrate-context',action='store_true',help='Normalize factual context while preserving warm-start predictions')
    p.add_argument('--eval-cases',type=int,default=100)
    p.add_argument('--evaluate-every',type=int,default=20)
    p.add_argument('--learning-rate',type=float,default=.0003)
    p.add_argument('--ppo-learning-rate',type=float,default=.00003)
    p.add_argument('--pass-rate',type=float,default=.9)
    p.add_argument('--require-pass',action='store_true')
    args=p.parse_args()
    if min(args.workers,args.round_episodes,args.train_batch,args.supervised_epochs,args.eval_cases,args.evaluate_every,args.learning_rate,args.ppo_learning_rate)<=0:
        p.error('Counts and learning rates must be positive')
    if min(args.warmup_rounds,args.ppo_rounds)<0 or not 0<args.pass_rate<=1:p.error('Invalid round count or pass rate')
    state=asyncio.run(run(args))
    if args.require_pass and not state['passed']:raise SystemExit(1)
