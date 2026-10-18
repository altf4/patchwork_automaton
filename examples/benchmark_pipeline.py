"""Matched RustMage collection and PPO benchmark, with warm-up excluded.

Use --update-reference to train on an earlier run's identical frozen rollouts.
Weights are changed in memory only; the input checkpoint is never overwritten.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

from . import train_rustmage  # CPU runtime settings before TensorFlow imports
import numpy as np

from patchwork_automaton.neural.batching import InferenceBatcher, example_batch
from patchwork_automaton.neural.experiment import atomic_json, read_trajectory
from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
from patchwork_automaton.neural.streaming_ppo import StreamingPPOTrainer
from patchwork_automaton.rl.rustmage_experiment import run_games


def trajectory_digest(path):
    digest = hashlib.sha256()
    for sample in read_trajectory(path):
        for key, array in sorted(sample.arrays.items()):
            digest.update(key.encode())
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        digest.update(str(sample.action).encode())
    return digest.hexdigest()


async def run(args):
    if min(args.games, args.workers) < 1:
        raise ValueError('Games and workers must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    policy = RustMagePolicy.load(args.checkpoint)
    report = dict(checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        games=args.games, workers=args.workers, seed=args.seed,
        timing_scope='Warm collection and PPO; excludes startup, graph compilation and checkpoints',
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path('src/patchwork_automaton').rglob('*.py'))})
    batcher = await InferenceBatcher(policy.model, max_batch=args.workers, wait_ms=2).start()
    try:
        await run_games(games=4, workers=args.workers, seed_base=args.seed+100000,
            mode='selfplay', model=policy.model, batcher=batcher)
        start = time.perf_counter(); cpu = time.process_time()
        rows = await run_games(games=args.games, workers=args.workers, seed_base=args.seed,
            mode='selfplay', model=policy.model, batcher=batcher, output=args.output/'rollouts')
        report['collection'] = dict(seconds=time.perf_counter()-start,
            cpu_seconds=time.process_time()-cpu, rows=rows)
    finally:
        await batcher.close()
    if any(r['status'] not in ('win', 'loss', 'draw') for r in rows):
        atomic_json(args.output/'report.json', report)
        raise RuntimeError('Benchmark requires natural terminal games')
    report['feature_action_hashes'] = [trajectory_digest(t['path']) for r in rows for t in r['trajectories']]
    atomic_json(args.output/'report.json', report)
    print(json.dumps(dict(stage='collection', seconds=report['collection']['seconds'])), flush=True)
    source = json.loads(args.update_reference.read_text()) if args.update_reference else report
    trajectories = [t for r in source['collection']['rows'] for t in r['trajectories']]
    trainer = StreamingPPOTrainer(policy.model, seed=5721, batch_size=32, epochs=2,
        learning_rate=.0003, target_kl=.02, max_state_kl=.1, advantage_scale_floor=1.)
    trainer.compile()
    saved = trainer.snapshot()
    batch = example_batch(policy.model.memory_size, policy.model.feature_spec)
    zero = np.zeros(1, np.float32)
    trainer.step(batch, np.zeros(1, np.int32), zero, zero, zero)
    trainer.restore(saved)
    start = time.perf_counter(); cpu = time.process_time()
    result = trainer.update_files(trajectories)
    report['update'] = dict(seconds=time.perf_counter()-start,
        cpu_seconds=time.process_time()-cpu, reference=str(args.update_reference), result=result)
    np.savez(args.output/'updated-weights.npz', *policy.model.get_weights())
    report['total_seconds'] = report['collection']['seconds']+report['update']['seconds']
    report['games_per_minute'] = 60*args.games/report['total_seconds']
    atomic_json(args.output/'report.json', report)
    print(json.dumps(dict(stage='complete', update_seconds=report['update']['seconds'],
        games_per_minute=report['games_per_minute'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--update-reference', type=Path)
    parser.add_argument('--games', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=23000000)
    asyncio.run(run(parser.parse_args()))
