"""Independent frozen-policy evaluation against the active baseline."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.update(KERAS_BACKEND='tensorflow', CUDA_VISIBLE_DEVICES='-1',
    TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='4', TF_ENABLE_ONEDNN_OPTS='0')

from patchwork_automaton.neural.devices import configure_runtime
from patchwork_automaton.neural.experiment import atomic_json
from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
from patchwork_automaton.neural.rustmage_evaluation import baseline_evaluation_config, match_summary
from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector


async def run(args):
    configure_runtime('cpu', hide_gpu=True)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.output.exists():
        d = json.loads(args.output.read_text())
        assert d['checkpoint_sha256']==digest and d['config']['cohort']['seed_base']==args.seed
        assert d['summary']['attempted_games']==args.games
        return
    policy = RustMagePolicy.load(args.checkpoint)
    collector = ProcessGameCollector(policy.model, workers=48, inference_processes=4,
        wait_ms=0, inference_device='cpu')
    cohort = baseline_evaluation_config(seed_base=args.seed, games=args.games,
        max_own_turns=60, greedy=False)
    started = time.monotonic()
    count = 0
    def progress(row):
        nonlocal count
        count += 1
        if count%250==0 or count==args.games:
            print(json.dumps(dict(stage='confirmation-evaluation',complete=count,total=args.games)),flush=True)
    try:
        await collector.start()
        rows = await collector.run_games(**cohort['collection'], progress=progress)
        assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==digest
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, dict(status='complete',checkpoint=str(args.checkpoint),
            checkpoint_sha256=digest,config=cohort,summary=match_summary(rows,expected_games=args.games),
            games=rows,seconds=time.monotonic()-started))
    finally:
        await collector.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--games',type=int,default=10000)
    asyncio.run(run(p.parse_args()))
