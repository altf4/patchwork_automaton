"""Frozen mono-red turns-to-kill evaluation against a completely inert opponent."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

# The training entry point sets the CPU environment before importing TensorFlow.
from . import train_rustmage  # noqa: F401
from patchwork_automaton.neural.batching import InferenceBatcher
from patchwork_automaton.neural.experiment import atomic_json
from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
from patchwork_automaton.rl.rustmage_experiment import run_games, goldfish_summary


async def run(args):
    if min(args.games,args.workers,args.horizon)<1:raise ValueError('Positive counts required')
    checkpoint=args.checkpoint
    if checkpoint and checkpoint.is_dir():
        pointer=json.loads((checkpoint/'latest.json').read_text())['checkpoint']
        if Path(pointer).name!=pointer:raise ValueError('Invalid checkpoint path')
        checkpoint=checkpoint/pointer/'policy.keras'
    before=hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None
    policy=RustMagePolicy.load(checkpoint) if checkpoint else None
    batcher=await InferenceBatcher(policy.model,max_batch=args.workers,wait_ms=2).start() if policy else None
    def progress(row):
        print(json.dumps(dict(index=row['index'],status=row['status'],turns=row['own_turns_to_win'])),flush=True)
    try:
        rows=await run_games(games=args.games,workers=args.workers,seed_base=args.seed,
            mode='goldfish' if policy else 'baseline',model=policy.model if policy else None,
            batcher=batcher,max_own_turns=args.horizon,greedy=args.greedy,progress=progress)
        if checkpoint and hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=before:
            raise RuntimeError('Frozen evaluation checkpoint changed')
        result=dict(checkpoint=str(checkpoint) if checkpoint else None,checkpoint_sha256=before,
            opponent='passive 60 Mountains',policy='greedy' if args.greedy else 'sampled temperature 1',
            summary=goldfish_summary(rows,args.horizon),games=rows)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output,result)
        print(json.dumps(result['summary']),flush=True)
    finally:
        if batcher:await batcher.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,help='policy.keras or a directory containing latest.json; omit for scripted baseline')
    parser.add_argument('--games',type=int,default=100)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--horizon',type=int,default=25)
    parser.add_argument('--seed',type=int,default=20000721)
    parser.add_argument('--greedy',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    asyncio.run(run(parser.parse_args()))
