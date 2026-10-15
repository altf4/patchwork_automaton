"""Frozen Burn turn-to-win benchmark against an inert 60-Mountain opponent.

Turns are the evaluated player's own turns, not XMage's global turn number.
Average/median cover wins only; nonwins and cutoffs remain in the denominator
of win rate and win-by-turn rates. No optimizer or reward modification is used.
"""
from examples.train_parallel import (
    asyncio,Path,time,np,ThreadPoolExecutor,NeuralPolicy,InferenceBatcher,
    BatchedPolicy,atomic_json,thaw,visible_state)
import argparse
from collections import Counter
import gzip
import hashlib
import json

from patchwork_automaton.baselines.modern_burn import BurnBaseline,modern_burn_deck
from patchwork_automaton.baselines.passive import PassivePolicy,passive_deck
from patchwork_automaton.backends.xmage_prompts import prompt_metadata
from patchwork_automaton.core import Response
from patchwork_automaton.rl import XMageSelfPlay,EpisodeEnd
from patchwork_automaton.rl.goldfish import OwnTurnCounter,RepeatGuard,decision_state_key,summarize


async def play(index,seed,args,model,batcher,executor):
    # Seat and play/draw rotate independently, covering all four combinations.
    bot_seat='player_'+str(index%2);on_play=index%4<2
    policy=BatchedPolicy(model,batcher,version=0,seed=seed) if model is not None else None
    baseline=BurnBaseline();passive=PassivePolicy();tracker=None;runner=None
    callbacks=Counter();start=time.monotonic();trace=None
    repeats=RepeatGuard(args.repeat_limit)
    def observe(obs):
        if policy:policy.observe(obs)
        if tracker:tracker.observe(obs)
    decks=(modern_burn_deck(),passive_deck()) if bot_seat=='player_0' else (passive_deck(),modern_burn_deck())
    factory=XMageSelfPlay(*decks,timeout=args.timeout,max_decisions=args.limit,
        client_options={'on_observation':observe})
    row=dict(index=index,seed=seed,bot_seat=bot_seat,on_play=on_play,controller=args.controller)
    try:
        path=args.directory/'traces'/f'g{index:04d}.jsonl.gz';path.parent.mkdir(parents=True,exist_ok=True)
        trace=gzip.open(path,'wt',compresslevel=1)
        async with asyncio.timeout(args.game_timeout):
            runner=await factory.reset()
            bot_actor=(await runner.games[bot_seat].snapshot()).actor
            tracker=OwnTurnCounter(bot_actor)
            while True:
                turn=await runner.next()
                if isinstance(turn,EpisodeEnd):
                    for obs in turn.observations.values():tracker.observe(obs)
                    if turn.truncated:
                        row.update(status='cutoff',reason=turn.reason);break
                    if set(turn.observations)!=set(runner.games):raise RuntimeError('Missing terminal seat')
                    result=turn.observations[bot_seat]
                    other=turn.observations['player_'+str(1-index%2)]
                    if result.outcome is None or other.outcome is None:raise RuntimeError('Missing terminal outcome')
                    if result.outcome.draw is True and other.outcome.draw is True:status='draw'
                    elif result.outcome.won is True and other.outcome.won is False:status='win'
                    elif result.outcome.won is False and other.outcome.won is True:status='loss'
                    else:raise RuntimeError('Inconsistent terminal outcomes')
                    enemy=next(p for p in result.players if p.id!=bot_actor)
                    row.update(status=status,own_turns_to_win=tracker.turns if status=='win' else None,
                        opponent_final_life=enemy.life,outcomes={seat:dict(won=o.outcome.won,draw=o.outcome.draw)
                            for seat,o in turn.observations.items()})
                    break
                obs=turn.observation;tracker.observe(obs)
                if tracker.first_active_player is not None and (tracker.first_active_player==bot_actor)!=on_play:
                    raise RuntimeError('Actual starting player disagrees with benchmark setup')
                if tracker.turns>args.max_turns:
                    row.update(status='cutoff',reason='own_turn_limit');break
                metadata=prompt_metadata(obs.decision.kind,obs.decision.native);purpose=metadata['purpose']
                if turn.actor==bot_seat:
                    key=decision_state_key(visible_state(obs),obs.decision.kind,
                        [dict(id=o.id,kind=o.kind) for o in obs.decision.options],metadata)
                    if repeats.observe(key):
                        row.update(status='cutoff',reason='repeated_state_limit',repeat_count=repeats.visits[key],
                            cutoff_kind=obs.decision.kind,cutoff_choices=[dict(id=o.id,label=o.label) for o in obs.decision.options])
                        break
                if purpose=='starting_player':
                    target=bot_actor if on_play else next(p.id for p in obs.players if p.id!=bot_actor)
                    response=Response(next(o.id for o in obs.decision.options if o.id=='object:'+target.value))
                    controller='setup'
                elif turn.actor!=bot_seat:response=passive.choose(obs);controller='passive'
                elif policy:
                    action=await asyncio.get_running_loop().run_in_executor(executor,policy.act,obs)
                    response=action.response;controller='neural'
                else:response=baseline.choose(obs);controller='baseline'
                callbacks[controller+':'+obs.decision.kind]+=1
                trace.write(json.dumps(dict(step=runner.steps,actor=turn.actor,controller=controller,
                    own_turns=tracker.turns,kind=obs.decision.kind,state=visible_state(obs),
                    choices=[dict(id=o.id,label=o.label) for o in obs.decision.options],
                    response=thaw(vars(response))))+'\n')
                await runner.submit(response)
    except TimeoutError:row.update(status='cutoff',reason='wall_clock_limit')
    except Exception as error:row.update(status='error',reason=repr(error))
    finally:
        row.update(decisions=runner.steps if runner else 0,own_turns_observed=tracker.turns if tracker else 0,
            actual_on_play=(tracker.first_active_player==tracker.actor) if tracker and tracker.first_active_player else None,
            observed_own_global_turn_ids=sorted(tracker.turn_ids) if tracker else [],callbacks=dict(callbacks),
            seconds=round(time.monotonic()-start,3))
        if trace:trace.write(json.dumps(dict(terminal_summary=row))+'\n');trace.close()
        await factory.close()
    return row


async def run(args):
    args.directory.mkdir(parents=True,exist_ok=False)
    checksum=hashlib.sha256(args.load.read_bytes()).hexdigest() if args.controller=='neural' else None
    model=NeuralPolicy.load(args.load,seed=args.seed).model if checksum else None
    batcher=await InferenceBatcher(model,max_batch=args.workers,wait_ms=3).start() if model else None
    executor=ThreadPoolExecutor(max_workers=args.workers)
    jobs=asyncio.Queue();rng=np.random.default_rng(args.seed)
    for i in range(args.games):jobs.put_nowait((i,int(rng.integers(2**31-1))))
    rows=[];start=time.monotonic()
    def persist():
        data=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            metric='Own turns until verified win, conditional on winning; report nonwins separately',
            opponent='60 Mountains; always keep, never play cards, pass priority; deterministic forced discards',
            setup='Harness alternates play/draw independently of seats; evaluated policy retains its mulligan decisions',
            checkpoint_sha256=checksum,optimizer_steps=0,summary=summarize(rows),
            by_start={name:summarize([r for r in rows if r['on_play']==flag]) for name,flag in (('play',True),('draw',False))},
            games=sorted(rows,key=lambda r:r['index']),seconds=round(time.monotonic()-start,3),
            seed_scope='Policy sampling only; stock XMage shuffles are not seeded')
        atomic_json(args.directory/'results.json',data);return data
    async def worker():
        while not jobs.empty():
            index,seed=jobs.get_nowait();row=await play(index,seed,args,model,batcher,executor)
            rows.append(row);data=persist()
            print(json.dumps(dict(game=index,status=row['status'],turns=row.get('own_turns_to_win'),summary=data['summary'])),flush=True)
    workers=[asyncio.create_task(worker()) for _ in range(min(args.workers,args.games))]
    try:
        persist();await asyncio.gather(*workers)
        if checksum and hashlib.sha256(args.load.read_bytes()).hexdigest()!=checksum:raise RuntimeError('Checkpoint changed')
        return persist()
    finally:
        for task in workers:task.cancel()
        await asyncio.gather(*workers,return_exceptions=True)
        await asyncio.to_thread(executor.shutdown,wait=True)
        if batcher:await batcher.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--controller',choices=('neural','baseline'),default='neural')
    p.add_argument('--load',type=Path)
    p.add_argument('--games',type=int,default=20)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=701)
    p.add_argument('--max-turns',type=int,default=20)
    p.add_argument('--limit',type=int,default=3000)
    p.add_argument('--timeout',type=float,default=60)
    p.add_argument('--game-timeout',type=float,default=240)
    p.add_argument('--repeat-limit',type=int,default=25,help='Repeated identical visible decisions allowed; 0 disables this Burn-only guard')
    args=p.parse_args()
    if args.controller=='neural' and args.load is None:p.error('--load is required for neural evaluation')
    if min(args.games,args.workers,args.max_turns,args.limit,args.timeout,args.game_timeout)<1:p.error('Counts and timeouts must be positive')
    if args.repeat_limit<0:p.error('Repeat limit must be nonnegative')
    asyncio.run(run(args))
