"""Matched frozen-policy collection throughput, including trajectory writing."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('KERAS_BACKEND','tensorflow')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')

import keras
import numpy as np
from patchwork_automaton.neural.model import SeparateActorCritic
from patchwork_automaton.neural.rustmage_features import RustMageEncoder
from patchwork_automaton.neural.batching import InferenceBatcher
from patchwork_automaton.neural.experiment import atomic_json
from patchwork_automaton.rl.rustmage_experiment import run_games
from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector


async def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    keras.utils.set_random_seed(721)
    encoder=RustMageEncoder()
    model=SeparateActorCritic(encoder.card_count,encoder.feature_spec)
    batcher=await InferenceBatcher(model,max_batch=8,wait_ms=2).start()
    def weights_hash():
        return hashlib.sha256(b''.join(np.asarray(v).tobytes() for v in model.weights)).hexdigest()
    report=dict(games=args.games,encoder=encoder.manifest(),weights_sha256=weights_hash(),runs=[],
                parameters=model.count_params(),timing='Warm-up/startup excluded; trajectory writing included')
    try:
        for kind,workers in [('threads',4),('processes',4),('processes',8)]:
            collector=ProcessGameCollector(model,batcher,workers=workers) if kind=='processes' else None
            label=f'{kind}-{workers}'
            async def collect(**options):
                if collector:return await collector.run_games(**options)
                return await run_games(model=model,batcher=batcher,workers=workers,**options)
            try:
                warm=await collect(games=8,seed_base=70_000_000,mode='selfplay',max_own_turns=60)
                if any(r['status'] not in ('win','loss','draw') for r in warm):raise RuntimeError(warm)
                before=batcher.stats(); started=time.monotonic();cpu=time.process_time()
                rows=await collect(games=args.games,seed_base=71_000_000,mode='selfplay',max_own_turns=60,
                    output=args.output/label,progress=lambda r:print(json.dumps(dict(stage=label,index=r['index'],status=r['status'])),flush=True))
                elapsed=time.monotonic()-started
                cpu=time.process_time()-cpu+sum(r.get('worker_cpu_seconds',0) for r in rows)
                after=batcher.stats()
                completed=sum(r['status'] in ('win','loss','draw') for r in rows)
                result=dict(collector=kind,workers=workers,seconds=elapsed,completed=completed,
                    games_per_minute=60*completed/elapsed,mean_cpu_cores=cpu/elapsed,
                    mean_inference_batch=(after['requests']-before['requests'])/max(1,after['calls']-before['calls']),games=rows)
                report['runs'].append(result)
                if weights_hash()!=report['weights_sha256']:raise AssertionError('Benchmark changed model weights')
                atomic_json(args.output/'report.json',report)
                print(json.dumps({k:v for k,v in result.items() if k!='games'}),flush=True)
            finally:
                if collector:await collector.close()
    finally:await batcher.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--games',type=int,default=16)
    asyncio.run(run(parser.parse_args()))
