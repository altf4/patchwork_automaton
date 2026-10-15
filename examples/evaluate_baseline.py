"""Frozen-model XMage evaluation against the deterministic Modern Burn baseline.

Exactly the requested number of games are attempted; safety cutoffs and errors
are retained separately and never substituted with replacement games or losses.
"""
from examples.train_parallel import (
    asyncio,os,json,Path,time,np,ThreadPoolExecutor,NeuralPolicy,
    InferenceBatcher,play_game,atomic_json)
import argparse
import hashlib


def summary(games):
    result=dict(attempted=len(games),wins=0,losses=0,draws=0,truncated=0,errors=0)
    for game in games:
        if 'error' in game:
            result['truncated' if game.get('truncated') else 'errors']+=1
        else:
            reward=game['rewards']['player_'+str(1-game['index']%2)]
            result['wins' if reward>0 else 'losses' if reward<0 else 'draws']+=1
    result['completed']=result['wins']+result['losses']+result['draws']
    result['win_rate']=result['wins']/result['completed'] if result['completed'] else None
    return result


async def run(args):
    args.directory.mkdir(parents=True,exist_ok=False)
    checksum=hashlib.sha256(args.load.read_bytes()).hexdigest()
    model=NeuralPolicy.load(args.load,seed=args.seed).model
    batcher=await InferenceBatcher(model,max_batch=args.workers,wait_ms=3).start()
    executor=ThreadPoolExecutor(max_workers=args.workers)
    queue=asyncio.Queue();rng=np.random.default_rng(args.seed)
    for index in range(args.games):queue.put_nowait((index,int(rng.integers(0,2**31-2))))
    games=[];start=time.monotonic()
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    def persist():
        result=dict(config=config,checkpoint_sha256=checksum,optimizer_steps=0,
            summary=summary(games),games=sorted(games,key=lambda g:g['index']),
            seconds=round(time.monotonic()-start,3),inference=batcher.stats())
        atomic_json(args.directory/'results.json',result)
        return result
    async def worker():
        while not queue.empty():
            index,seed=queue.get_nowait();name=f'eval-g{index:04d}'
            try:
                async with asyncio.timeout(args.game_timeout):
                    game=await play_game((name,seed,index),model,batcher,executor,args,evaluate=True,collect=False)
                if 'error' in game:game['truncated']='Truncated:' in game['error']
            except TimeoutError:
                game=dict(name=name,index=index,seed=seed,error='Wall-clock safety cutoff',truncated=True)
            games.append(game)
            folder=args.directory/'games';folder.mkdir(exist_ok=True)
            atomic_json(folder/(name+'.json'),game)
            result=persist()
            print(json.dumps(dict(game=name,summary=result['summary'],decisions=game.get('decisions'),
                                  seconds=game.get('seconds'),error=game.get('error'))),flush=True)
    workers=[asyncio.create_task(worker()) for _ in range(min(args.workers,args.games))]
    try:
        persist()
        await asyncio.gather(*workers)
        if hashlib.sha256(args.load.read_bytes()).hexdigest()!=checksum:raise RuntimeError('Checkpoint changed during evaluation')
        return persist()
    finally:
        for task in workers:task.cancel()
        await asyncio.gather(*workers,return_exceptions=True)
        persist()
        await asyncio.to_thread(executor.shutdown,wait=True)
        await batcher.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--load',type=Path,required=True)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--games',type=int,default=20)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=501)
    p.add_argument('--limit',type=int,default=3000)
    p.add_argument('--timeout',type=float,default=60)
    p.add_argument('--game-timeout',type=float,default=600)
    p.add_argument('--record-inputs',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args()
    if min(args.games,args.workers,args.limit,args.timeout,args.game_timeout)<1:p.error('Counts and timeouts must be positive')
    asyncio.run(run(args))
