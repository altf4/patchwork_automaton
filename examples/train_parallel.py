"""CPU parallel XMage self-play, shared batched inference and round checkpoints."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('KERAS_BACKEND','tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import math
from pathlib import Path
import resource
import shutil
import time
import numpy as np

from patchwork_automaton.baselines.modern_burn import BurnBaseline, modern_burn_deck
from patchwork_automaton.neural.policy import NeuralPolicy, untrained_policy
from patchwork_automaton.neural.batching import BatchedPolicy, InferenceBatcher
from patchwork_automaton.neural.experiment import (TrajectoryWriter, StreamingTrainer,
    atomic_json, save_checkpoint, load_checkpoint, make_trainer)
from patchwork_automaton.neural.training import terminal_reward
from patchwork_automaton.rl import XMageSelfPlay, EpisodeEnd


def thaw(value):
    from collections.abc import Mapping
    if isinstance(value,Mapping): return {str(k):thaw(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [thaw(v) for v in value]
    return value


def visible_state(obs):
    objects=[]
    for handle,obj in obs.objects.items():
        hidden=not obj.name or any(obj.properties.get(k) for k in ('hideInfo','faceDown','face_down'))
        objects.append(dict(id=handle.value,name=None if hidden else obj.name,zone=obj.zone,
            controller=obj.controller.value if obj.controller else None,
            properties={} if hidden else {k:thaw(obj.properties[k]) for k in
                ('tapped','power','toughness','damage','manaValue','attacking','blocking','summoningSickness',
                 'cardTypes','color','counters','isToken','isAbility','faceDown','canAttack','canBlock')
                if k in obj.properties}))
    return dict(revision=obs.revision,turn=obs.turn,phase=obs.phase,actor=obs.actor.value,
        active=obs.active_player.value if obs.active_player else None,
        players=[dict(id=p.id.value,life=p.life,hand_count=p.hand_count,library_count=p.library_count,
                      mana=thaw(p.native.get('manaPool'))) for p in obs.players],objects=objects)


async def play_game(job,model,batcher,executor,args,*,evaluate=False,collect=True):
    name,seed,index=job
    directory=args.directory/'rollouts'/name
    sinks={seat:TrajectoryWriter(directory/(seat+'.pkl.gz')) for seat in ('player_0','player_1')} if collect else {}
    replay_sinks={seat:TrajectoryWriter(args.directory/'replay_inputs'/name/(seat+'.pkl.gz'))
                  for seat in ('player_0','player_1')} if evaluate and not collect and getattr(args,'record_inputs',False) else {}
    policies={seat:BatchedPolicy(model,batcher,version=batcher.version,seed=seed+i,sink=sinks.get(seat))
              for i,seat in enumerate(('player_0','player_1'))}
    if replay_sinks:
        for seat,policy in policies.items():policy.sink=replay_sinks[seat]
    # History already keys every buffer by (game, actor). Process each callback
    # once, while each policy consumes only its own actor's buffer.
    history=policies['player_0'].history
    for policy in policies.values():policy.history=history
    def observe(obs):
        history.observe(obs)
    factory=XMageSelfPlay(modern_burn_deck(),timeout=args.timeout,max_decisions=args.limit,
                         client_options={'on_observation':observe})
    baseline=BurnBaseline(); baseline_seat='player_'+str(index%2)
    start=time.monotonic(); runner=None; trace=None; last_steps={}
    if evaluate:
        path=args.directory/'traces'/(name+'.jsonl.gz');path.parent.mkdir(parents=True,exist_ok=True)
        trace=gzip.open(path,'wt',compresslevel=1)
    try:
        runner=await factory.reset()
        while True:
            turn=await runner.next()
            if isinstance(turn,EpisodeEnd):break
            obs=turn.observation
            if evaluate and turn.actor==baseline_seat:
                response=baseline.choose(obs);details=[];controller='baseline'
            else:
                policy=policies[turn.actor]
                record=await asyncio.get_running_loop().run_in_executor(executor,policy.act,obs)
                response=record.response;details=policy.selection_details;controller='neural'
            if trace:
                trace.write(json.dumps(dict(step=runner.steps,actor=turn.actor,controller=controller,
                    model_version=batcher.version,token=thaw(vars(obs.decision.token)),
                    previous_actor_step=last_steps.get(turn.actor),state=visible_state(obs),
                    kind=obs.decision.kind,choices=[dict(id=o.id,kind=o.kind,label=o.label,
                    constraints=thaw(o.constraints)) for o in obs.decision.options],
                    selections=details,response=dict(option=response.option,value=response.value)),default=str)+'\n')
                last_steps[turn.actor]=runner.steps
            await runner.submit(response)
        if turn.truncated:raise RuntimeError('Truncated: '+str(turn.reason))
        if set(turn.observations)!=set(policies):raise RuntimeError('Missing terminal actor result')
        rewards={seat:terminal_reward(obs.outcome) for seat,obs in turn.observations.items()}
        if sorted(rewards.values()) not in ([-1.,1.],[0.,0.]):raise RuntimeError('Inconsistent zero-sum terminal results')
        if trace:
            trace.write(json.dumps(dict(terminal=True,rewards=rewards,
                states={s:visible_state(o) for s,o in turn.observations.items()}))+'\n')
        result=dict(name=name,index=index,seed=seed,model_version=batcher.version,
                    decisions=runner.steps,seconds=round(time.monotonic()-start,3),rewards=rewards)
        if evaluate:result['baseline_seat']=baseline_seat
        if replay_sinks:result['replay_inputs']=[dict(path=str(s.path),samples=s.count) for s in replay_sinks.values()]
        result['trajectories']=[dict(path=str(sink.path),samples=sink.count,reward=rewards[seat])
                                for seat,sink in sinks.items()]
        return result
    except Exception as error:
        return dict(name=name,index=index,seed=seed,error=repr(error),
                    decisions=runner.steps if runner else 0,seconds=round(time.monotonic()-start,3),
                    last_state=visible_state(obs) if 'obs' in locals() else None,
                    choices=[dict(id=o.id,label=o.label,kind=o.kind) for o in obs.decision.options]
                        if 'obs' in locals() and obs.decision else None)
    finally:
        await factory.close()
        for sink in sinks.values():sink.close()
        for sink in replay_sinks.values():sink.close()
        if trace:trace.close()


async def collect_games(count,phase,model,batcher,executor,args,rng,*,evaluate=False,collect=True):
    queue=asyncio.Queue()
    for index in range(count):queue.put_nowait((index,int(rng.integers(0,2**31-2))))
    results=[];failures=[];start=time.monotonic()
    async def worker():
        while not queue.empty():
            try:index,seed=queue.get_nowait()
            except asyncio.QueueEmpty:return
            for attempt in range(args.max_failures+1):
                name=f'{phase}-g{index:04d}-a{attempt}'
                result=await play_game((name,seed,index),model,batcher,executor,args,evaluate=evaluate,collect=collect)
                game_directory=args.directory/'games';game_directory.mkdir(parents=True,exist_ok=True)
                atomic_json(game_directory/(name+'.json'),result)
                if 'error' not in result:
                    results.append(result)
                    progress=dict(phase=phase,completed=len(results),target=count,failures=len(failures),
                                  elapsed_seconds=round(time.monotonic()-start,1),model_version=batcher.version)
                    atomic_json(args.directory/'progress.json',progress)
                    print(json.dumps(dict(progress,game=result['name'],decisions=result['decisions'])),flush=True)
                    break
                failures.append(result)
                with (args.directory/'failures.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
                print(json.dumps({'failure':result}),flush=True)
                if len(failures)>=args.max_failures:raise RuntimeError(f'{phase}: repeated game failures; inspect failures.jsonl')
                seed=int(rng.integers(0,2**31-2))
            else:raise RuntimeError('Retry limit reached')
    tasks=[asyncio.create_task(worker()) for _ in range(min(args.workers,count))]
    try:await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
    results.sort(key=lambda r:r['index'])
    return dict(phase=phase,games=results,failures=failures,seconds=round(time.monotonic()-start,3))


def evaluation_summary(result):
    wins=losses=draws=0
    for game in result['games']:
        baseline=game['rewards'][game['baseline_seat']]
        wins+=baseline<0;losses+=baseline>0;draws+=baseline==0
    n=wins+losses+draws;p=wins/n;z=1.95996398454
    center=(p+z*z/(2*n))/(1+z*z/n)
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(wins=wins,losses=losses,draws=draws,games=n,win_rate=p,
                wilson_95=[max(0,center-radius),min(1,center+radius)])


async def evaluate_checkpoint(label,count,model,batcher,executor,args):
    path=args.directory/f'evaluation-{label}.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['model_version']!=batcher.version or saved['summary']['games']!=count:
            raise ValueError('Saved evaluation does not match checkpoint')
        return saved['summary']
    # Evaluation RNG never changes the training seed stream.
    rng=np.random.default_rng(np.random.SeedSequence([args.seed,9001,batcher.version]))
    result=await collect_games(count,'eval-'+label,model,batcher,executor,args,rng,evaluate=True,collect=False)
    result['summary']=evaluation_summary(result);result['model_version']=batcher.version
    atomic_json(path,result)
    print(json.dumps({'evaluation':label,**result['summary']}),flush=True)
    return result['summary']


async def run(args):
    args.directory.mkdir(parents=True,exist_ok=True)
    checkpoints=args.directory/'checkpoints'
    if args.resume:
        model,trainer,state=await asyncio.to_thread(load_checkpoint,checkpoints)
        saved_algorithm=state['config'].get('algorithm','actor-critic')
        if args.algorithm is not None and args.algorithm!=saved_algorithm:raise ValueError('Cannot change optimizer algorithm on resume')
        args.algorithm=saved_algorithm
        expected={k:v for k,v in state['config'].items() if k not in ('workers','games','load')}
        if expected!={k:getattr(args,k) for k in expected}:raise ValueError('Resume configuration mismatch')
        if args.games<state['completed_games']:raise ValueError('Target below completed games')
        rng=np.random.default_rng();rng.bit_generator.state=state['rng_state']
        report_path=args.directory/'report.json'
        if report_path.exists():
            report=json.loads(report_path.read_text())
            if report['round']==state['round'] and report['completed_games']==state['completed_games']:
                for key in ('evaluations','inference'):
                    if key in report:state[key]=report[key]
        state['config'].update(games=args.games,workers=args.workers)
    else:
        if (checkpoints/'latest.json').exists():raise FileExistsError('Use --resume or a new directory')
        args.algorithm=args.algorithm or 'ppo'
        initial=(NeuralPolicy.load(args.load,seed=args.seed) if args.load else untrained_policy(seed=args.seed,
            candidate_interactions=True,player_readout=True,stable_heads=True,fact_readout=True))
        model=initial.model
        state=dict(round=0,completed_games=0,rounds=[],evaluations={},config={k:getattr(args,k) for k in
            ('workers','games','round_games','seed','limit','timeout','learning_rate','train_batch',
             'inference_batch','wait_ms','initial_eval','final_eval','intermediate_eval','max_failures',
             'algorithm','ppo_epochs','ppo_clip','target_kl','max_state_kl','advantage_scale_floor','record_inputs')})
        state['config']['load']=str(args.load) if args.load else None
        trainer=make_trainer(model,state['config'])
        rng=np.random.default_rng(args.seed)
    batcher=await InferenceBatcher(model,max_batch=args.inference_batch,wait_ms=args.wait_ms).start()
    executor=ThreadPoolExecutor(max_workers=max(args.workers,1),thread_name_prefix='game-policy')
    try:
        if not args.resume:await asyncio.to_thread(trainer.compile)
        batcher.set_version(state['round'])
        if args.resume and 'inference' in state:batcher.restore_stats(state['inference'])
        if not args.resume:
            state['rng_state']=rng.bit_generator.state
            await asyncio.to_thread(save_checkpoint,checkpoints,model,trainer,state)
        if not state['completed_games'] and args.initial_eval:
            state['evaluations']['initial']=await evaluate_checkpoint('initial',args.initial_eval,model,batcher,executor,args)
        # A checkpoint is committed before evaluation; resume runs any pending evaluation.
        async def intermediate():
            for milestone in (250,500,750):
                key=str(milestone)
                if state['completed_games']>=milestone and key not in state['evaluations'] and args.intermediate_eval:
                    state['evaluations'][key]=await evaluate_checkpoint(key,args.intermediate_eval,model,batcher,executor,args)
        await intermediate()
        while state['completed_games']<args.games:
            number=min(args.round_games,args.games-state['completed_games'])
            phase=f"train-r{state['round']+1:04d}"
            result=await collect_games(number,phase,model,batcher,executor,args,rng)
            trajectories=[t for g in result['games'] for t in g['trajectories']]
            atomic_json(args.directory/'progress.json',dict(phase='updating',round=state['round']+1,
                        completed_games=state['completed_games'],samples=sum(t['samples'] for t in trajectories)))
            before=time.monotonic()
            update=await asyncio.to_thread(trainer.update_files,trajectories)
            result['update']=update;result['update_seconds']=round(time.monotonic()-before,3)
            state['round']+=1;state['completed_games']+=number
            state['optimizer_steps']=int(trainer.optimizer.iterations)
            if hasattr(trainer,'rng'):state['trainer_rng_state']=trainer.rng.bit_generator.state
            state['rounds'].append(result);state['rng_state']=rng.bit_generator.state
            state['inference']=batcher.stats()
            state['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            await asyncio.to_thread(save_checkpoint,checkpoints,model,trainer,state)
            atomic_json(args.directory/'report.json',state)
            batcher.set_version(state['round'])
            print(json.dumps({'round_complete':state['round'],'completed_games':state['completed_games'],
                              'update':update,'inference':batcher.stats()}),flush=True)
            # Only this committed round's generated temporary rollout directories.
            for game in result['games']:
                path=args.directory/'rollouts'/game['name']
                if path.exists():shutil.rmtree(path)
            await intermediate()
        if args.final_eval:
            state['evaluations']['final']=await evaluate_checkpoint('final',args.final_eval,model,batcher,executor,args)
        state['inference']=batcher.stats()
        atomic_json(args.directory/'report.json',state)
        atomic_json(args.directory/'progress.json',dict(phase='complete',completed_games=state['completed_games'],round=state['round']))
    finally:
        await asyncio.to_thread(executor.shutdown,wait=True)
        await batcher.close()


async def benchmark(args):
    args.directory.mkdir(parents=True,exist_ok=True)
    model=untrained_policy(seed=args.seed).model
    batcher=await InferenceBatcher(model,max_batch=args.inference_batch,wait_ms=args.wait_ms).start()
    executor=ThreadPoolExecutor(max_workers=16,thread_name_prefix='benchmark-policy')
    reports=[]
    try:
        for workers in args.benchmark_workers:
            args.workers=workers
            before=batcher.stats();start=time.monotonic()
            result=await collect_games(workers,f'benchmark-{workers}',model,batcher,executor,args,
                                       np.random.default_rng(args.seed),collect=False)
            seconds=time.monotonic()-start;after=batcher.stats()
            decisions=sum(g['decisions'] for g in result['games'])
            reports.append(dict(workers=workers,games=len(result['games']),seconds=seconds,
                decisions=decisions,decisions_per_second=decisions/seconds,
                games_per_hour=len(result['games'])*3600/seconds,
                mean_batch=(after['requests']-before['requests'])/max(1,after['calls']-before['calls']),
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                failures=result['failures']))
            atomic_json(args.directory/'benchmark.json',reports)
            print(json.dumps({'benchmark':reports[-1]}),flush=True)
    finally:
        await asyncio.to_thread(executor.shutdown,wait=True)
        await batcher.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('.local/training/burn-parallel-1000'))
    parser.add_argument('--games',type=int,default=1000)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--round-games',type=int,default=16)
    parser.add_argument('--seed',type=int,default=41)
    parser.add_argument('--limit',type=int,default=3000)
    parser.add_argument('--timeout',type=float,default=60)
    parser.add_argument('--learning-rate',type=float,default=.0003)
    parser.add_argument('--load',type=Path,help='Warm start from a compatible policy; optimizer starts fresh')
    parser.add_argument('--algorithm',choices=('actor-critic','ppo'),default=None,help='New runs default to PPO; resume preserves its algorithm')
    parser.add_argument('--ppo-epochs',type=int,default=2)
    parser.add_argument('--ppo-clip',type=float,default=.2)
    parser.add_argument('--target-kl',type=float,default=.01)
    parser.add_argument('--max-state-kl',type=float,default=.05)
    parser.add_argument('--advantage-scale-floor',type=float,default=1.)
    parser.add_argument('--record-inputs',action=argparse.BooleanOptionalAction,default=True,
                        help='Retain exact numerical evaluation inputs, recurrent states, and behavior probabilities')
    parser.add_argument('--train-batch',type=int,default=16)
    parser.add_argument('--inference-batch',type=int,default=16)
    parser.add_argument('--wait-ms',type=float,default=5)
    parser.add_argument('--initial-eval',type=int,default=100)
    parser.add_argument('--final-eval',type=int,default=100)
    parser.add_argument('--intermediate-eval',type=int,default=20)
    parser.add_argument('--max-failures',type=int,default=3)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--benchmark-workers',type=int,nargs='+',default=[4,8])
    args=parser.parse_args()
    if args.resume and args.load:parser.error('--load is for a fresh run, not --resume')
    if args.ppo_epochs<1 or min(args.target_kl,args.max_state_kl,args.advantage_scale_floor)<=0 or not 0<args.ppo_clip<1:
        parser.error('Invalid PPO configuration')
    if min(args.benchmark_workers)<1:parser.error('Benchmark worker counts must be positive')
    if min(args.games,args.workers,args.round_games,args.train_batch,args.inference_batch,args.max_failures,args.limit,args.timeout)<1:
        parser.error('Counts and timeout must be positive')
    if min(args.initial_eval,args.final_eval,args.intermediate_eval,args.wait_ms)<0:parser.error('Invalid negative option')
    asyncio.run(benchmark(args) if args.benchmark else run(args))
