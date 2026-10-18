"""CPU mono-red mirror PPO, with independent before/after turns-to-kill evaluation.

uv run --no-sync python -m examples.train_rustmage --output .local/red-ppo
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

# Set before importing Keras/TensorFlow. Worker games share one inference service.
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import asyncio
import keras

from patchwork_automaton.neural.batching import InferenceBatcher, example_batch
from patchwork_automaton.neural.experiment import atomic_json, save_checkpoint, load_checkpoint, make_trainer
from patchwork_automaton.neural.model import PolicyNetwork
from patchwork_automaton.neural.rustmage_features import RustMageEncoder
from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
from patchwork_automaton.rl.rustmage_experiment import run_games, goldfish_summary, paired_improvement


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--games-per-round', type=int, default=32)
    p.add_argument('--eval-games', type=int, default=40)
    p.add_argument('--eval-horizon', type=int, default=20)
    p.add_argument('--train-horizon', type=int, default=60)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--train-batch', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=.0003)
    p.add_argument('--ppo-epochs', type=int, default=2)
    p.add_argument('--seed', type=int, default=721)
    p.add_argument('--resume', action='store_true')
    return p


async def run(args):
    if min(args.rounds,args.games_per_round,args.eval_games,args.eval_horizon,
           args.train_horizon,args.workers,args.train_batch) < 1:
        raise ValueError('Counts and horizons must be positive')
    output = args.output
    config = {k:v for k,v in vars(args).items() if k not in ('output','resume','rounds')}
    config.update(algorithm='ppo', target_kl=.02, max_state_kl=.1,
                  advantage_scale_floor=1., ppo_clip=.2)
    report_path = output/'report.json'
    if args.resume:
        model,trainer,state = load_checkpoint(output/'checkpoints')
        if state['config'] != config: raise ValueError('Resume requires the same experiment configuration')
        first_round = state['round']+1
        report = json.loads(report_path.read_text())
        report['rounds'] = [r for r in report['rounds'] if r['round'] < first_round]
    else:
        output.mkdir(parents=True,exist_ok=False)
        keras.utils.set_random_seed(args.seed)
        encoder = RustMageEncoder()
        model = PolicyNetwork(encoder.card_count, width=32, memory_size=32, blocks=1,
            candidate_interactions=True, player_readout=True, stable_heads=True,
            isolate_value=True, fact_readout=True, feature_spec=encoder.feature_spec)
        model(example_batch(model.memory_size,model.feature_spec),training=False)
        trainer = make_trainer(model,config)
        trainer.compile()
        first_round = 1
        report = dict(config=config, encoder=encoder.manifest(), parameters=model.count_params(),
            training='mono-red versus mono-red PPO; undiscounted terminal +1/-1/0 only; no imitation',
            evaluation='sampled policy, temperature 1; do-nothing 60-Mountain opponent; fixed held-out seeds',
            concession='Voluntary Concede is excluded from both training and evaluation; no other strategy masking',
            cutoffs='Excluded from PPO; retained as failed evaluation attempts', rounds=[])
        library=Path('.local/rustmage/target/release/librustmage.so')
        if library.is_file(): report['library_sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
        report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__), *Path('src/patchwork_automaton/neural').glob('*.py'),
            Path('src/patchwork_automaton/rl/rustmage_experiment.py')]}
        save_checkpoint(output/'checkpoints',model,trainer,dict(round=0,config=config),policy_type=RustMagePolicy)
        atomic_json(report_path,report)
    invocation=dict(first_round=first_round,requested_last_round=args.rounds,
        started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__),Path('src/patchwork_automaton/rl/rustmage_experiment.py'))})
    report.setdefault('invocations',[]).append(invocation)
    atomic_json(report_path,report)
    batcher = await InferenceBatcher(model,max_batch=args.workers,wait_ms=2).start()
    started=time.monotonic()
    def progress(label):
        count=0
        def completed(row):
            nonlocal count
            count+=1
            print(json.dumps(dict(stage=label,complete=count,seed=row['seed'],status=row['status'],
                turns=row['own_turns_to_win'],seconds=round(row['seconds'],2),
                **({'error':row['error']} if row['status']=='error' else {}))),flush=True)
        return completed
    async def evaluate(label,mode):
        rows=await run_games(games=args.eval_games,workers=args.workers,
            seed_base=args.seed+10_000_000,mode=mode,model=model,batcher=batcher,
            version=batcher.version,max_own_turns=args.eval_horizon,progress=progress(label))
        result=dict(summary=goldfish_summary(rows,args.eval_horizon),games=rows)
        print(json.dumps(dict(stage=label,summary=result['summary'])),flush=True)
        return result
    try:
        if 'baseline' not in report:
            report['baseline']=await evaluate('baseline','baseline');atomic_json(report_path,report)
        if 'before' not in report:
            report['before']=await evaluate('before','goldfish');atomic_json(report_path,report)
        for round_id in range(first_round,args.rounds+1):
            batcher.set_version(round_id)
            collection_started=time.monotonic()
            rows=await run_games(games=args.games_per_round,workers=args.workers,
                seed_base=args.seed+round_id*args.games_per_round,
                index_base=(round_id-1)*args.games_per_round,mode='selfplay',model=model,
                batcher=batcher,version=round_id,output=output/'rollouts',
                max_own_turns=args.train_horizon,progress=progress(f'train-{round_id}'))
            trajectories=[t for r in rows for t in r['trajectories']]
            stats=dict(round=round_id,collection_seconds=time.monotonic()-collection_started,
                       outcomes=dict(Counter(r['status'] for r in rows)),games=rows,
                       seat_wins=dict(Counter(str(r.get('winner')) for r in rows if 'winner' in r)))
            if not trajectories: raise RuntimeError('No verified terminal games; refusing a fabricated update')
            # All workers finished; the frozen behavior model can now change.
            update_started=time.monotonic()
            stats['update']=await asyncio.to_thread(trainer.update_files,trajectories)
            stats['update_seconds']=time.monotonic()-update_started
            checkpoint=save_checkpoint(output/'checkpoints',model,trainer,
                dict(round=round_id,config=config),policy_type=RustMagePolicy)
            stats['checkpoint']=str(checkpoint)
            report['rounds'].append(stats);report['batching']=batcher.stats()
            atomic_json(report_path,report)
            print(json.dumps(dict(stage='update',round=round_id,metrics=stats['update'])),flush=True)
        report['after']=await evaluate('after','goldfish')
        report['comparison']=paired_improvement(report['before']['games'],report['after']['games'],args.eval_horizon)
        report['seconds_this_invocation']=time.monotonic()-started
        invocation.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                          seconds=report['seconds_this_invocation'])
        timed=[r for r in report['rounds'] if 'collection_seconds' in r]
        if timed:
            games=sum(len(r['games']) for r in timed)
            collection=sum(r['collection_seconds'] for r in timed)
            updates=sum(r['update_seconds'] for r in timed)
            report['throughput']=dict(timed_selfplay_games=games,collection_seconds=collection,
                update_seconds=updates,collection_games_per_minute=60*games/collection,
                collection_and_updates_games_per_minute=60*games/(collection+updates))
        report['batching']=batcher.stats()
        atomic_json(report_path,report)
        print(json.dumps(dict(stage='result',comparison=report['comparison'],report=str(report_path))),flush=True)
    finally:
        await batcher.close()


if __name__ == '__main__': asyncio.run(run(parser().parse_args()))
