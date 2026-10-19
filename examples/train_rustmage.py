"""CPU mono-red PPO with an 80/20 game split and separate kill-turn evaluation.

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

# Set before importing Keras/TensorFlow, including in spawned inference replicas.
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '4')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import asyncio
import keras

from patchwork_automaton.neural.batching import InferenceBatcher, example_batch
from patchwork_automaton.neural.experiment import atomic_json, save_checkpoint, load_checkpoint, make_trainer
from patchwork_automaton.neural.model import PolicyNetwork, SeparateActorCritic
from patchwork_automaton.neural.rustmage_features import RustMageEncoder
from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
from patchwork_automaton.neural.validation import GameValidator, split_games, training_normalization
from patchwork_automaton.rl.rustmage_experiment import run_games, goldfish_summary, paired_improvement


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--games-per-round', type=int, default=32)
    p.add_argument('--validation-fraction', type=float, default=.2,
        help='Whole games excluded from both optimizers (default: 20%%; count rounded up). Use 0 only for legacy runs.')
    p.add_argument('--eval-games', type=int, default=40)
    p.add_argument('--eval-every-games', type=int, default=0,
        help='Evaluate at this self-play interval; must be a multiple of games-per-round (0: endpoints only)')
    p.add_argument('--eval-horizon', type=int, default=20)
    p.add_argument('--train-horizon', type=int, default=60)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--inference-processes',type=int,
        help='Frozen inference replicas sharing the total engine count (new process runs: 2; old resumes: 1)')
    p.add_argument('--inference-wait-ms', type=float, default=0,
        help='Maximum batching delay; 0 immediately serves currently queued requests')
    p.add_argument('--collector', choices=('threads','processes'), default='processes')
    p.add_argument('--architecture', choices=('shared','separate'), default='separate')
    p.add_argument('--train-batch', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=.0003)
    p.add_argument('--ppo-epochs', type=int, default=2)
    p.add_argument('--concurrent-updates', action='store_true',
        help='Overlap independent actor/critic updates (requires separate architecture)')
    p.add_argument('--prefetch-batches',type=int,default=0,
        help='Prepared minibatches queued ahead per optimizer stream (0: synchronous)')
    p.add_argument('--measurement-workers',type=int,default=1,
        help='Concurrent loss/KL batches while the measured network is frozen')
    p.add_argument('--seed', type=int, default=721)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plot', action='store_true', help='Save a learning-curve PNG/SVG (requires the plots extra)')
    return p


def inference_process_count(args, saved=None):
    count = args.inference_processes
    if count is None:
        count = (saved.get('inference_processes',1) if saved is not None
                 else min(2,args.workers) if args.collector == 'processes' else 1)
    if not 1 <= count <= args.workers: raise ValueError('Inference processes must be between one and workers')
    if args.collector != 'processes' and count != 1:
        raise ValueError('Multiple inference processes require the process collector')
    return count


async def run(args):
    if min(args.rounds,args.games_per_round,args.eval_games,args.eval_horizon,
           args.train_horizon,args.workers,args.train_batch) < 1:
        raise ValueError('Counts and horizons must be positive')
    if args.eval_every_games < 0 or args.eval_every_games % args.games_per_round:
        raise ValueError('Evaluation interval must be a nonnegative multiple of games-per-round')
    if not 0 <= args.inference_wait_ms < float('inf'):
        raise ValueError('Inference wait must be finite and nonnegative')
    if args.concurrent_updates and args.architecture != 'separate':
        raise ValueError('Concurrent updates require separate architecture')
    if args.prefetch_batches < 0 or args.measurement_workers < 1:
        raise ValueError('Invalid prefetch depth or measurement worker count')
    if not 0 <= args.validation_fraction < 1 or (args.validation_fraction and args.games_per_round < 2):
        raise ValueError('Validation needs a fraction in [0,1) and at least two games per round')
    output = args.output
    config = {k:v for k,v in vars(args).items() if k not in ('output','resume','rounds','plot')}
    config['inference_processes'] = inference_process_count(args)
    config.update(algorithm='ppo', target_kl=.02, max_state_kl=.1,
                  advantage_scale_floor=1., ppo_clip=.2)
    report_path = output/'report.json'
    if args.resume:
        model,trainer,state = load_checkpoint(output/'checkpoints')
        saved_config = dict(state['config'])
        saved_config.setdefault('eval_every_games', 0)
        saved_config.setdefault('collector', 'threads')
        saved_config.setdefault('architecture', 'shared')
        saved_config.setdefault('inference_wait_ms', 2)
        saved_config.setdefault('concurrent_updates', False)
        saved_config.setdefault('prefetch_batches',0)
        saved_config.setdefault('measurement_workers',1)
        saved_config.setdefault('inference_processes',1)
        saved_config.setdefault('validation_fraction',0.)
        config['inference_processes'] = inference_process_count(args,saved_config)
        if saved_config != config:
            raise ValueError('Resume requires the same experiment configuration; legacy runs without a holdout require explicit --validation-fraction 0')
        if args.rounds < state['round']: raise ValueError('Target round precedes the saved checkpoint')
        first_round = state['round']+1
        report = json.loads(report_path.read_text())
        report['rounds'] = [r for r in report['rounds'] if r['round'] < first_round]
        # Recover a completed update if interruption occurred after the atomic
        # checkpoint but before publishing its round in report.json.
        if state.get('round_result') and not any(r['round']==state['round'] for r in report['rounds']):
            report['rounds'].append(state['round_result'])
        if [r['round'] for r in report['rounds']] != list(range(1, first_round)):
            raise ValueError('Report is missing completed rounds')
        report['evaluations'] = [e for e in report.get('evaluations', []) if e['round'] < first_round]
    else:
        output.mkdir(parents=True,exist_ok=False)
        keras.utils.set_random_seed(args.seed)
        encoder = RustMageEncoder()
        if args.architecture == 'separate':
            model = SeparateActorCritic(encoder.card_count, encoder.feature_spec)
        else:
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
            cutoffs='Excluded from PPO; retained as failed evaluation attempts', rounds=[], evaluations=[],
            initialization='fresh random weights', initial_training_games=0)
        report['validation_methodology'] = ('Whole-game holdout before outcome filtering; both seats stay together; '
            'training-only advantage normalization; validation never enters either optimizer. '
            'Policy surrogate gain is an outcome-based proxy, not a win rate. '
            'Legacy training_games fields count collected attempts; optimization/validation counts are explicit.')
        report['architecture'] = model.get_config()
        if args.architecture == 'separate':
            report['parameter_counts'] = dict(actor=model.actor.count_params(), critic=model.critic.count_params())
        library=Path('.local/rustmage/target/release/librustmage.so')
        if library.is_file(): report['library_sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
        report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__), *Path('src/patchwork_automaton/neural').glob('*.py'),
            *Path('src/patchwork_automaton/rl').glob('rustmage*.py'),
            Path('src/patchwork_automaton/baselines/foundations.py')]}
        save_checkpoint(output/'checkpoints',model,trainer,dict(round=0,config=config),policy_type=RustMagePolicy)
        atomic_json(report_path,report)
    invocation=dict(first_round=first_round,requested_last_round=args.rounds,
        game_engines=args.workers,inference_processes=config['inference_processes'],
        runtime={key: os.environ.get(key) for key in ('TF_NUM_INTRAOP_THREADS',
            'TF_NUM_INTEROP_THREADS', 'TF_ENABLE_ONEDNN_OPTS', 'CUDA_VISIBLE_DEVICES')},
        started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__),Path('src/patchwork_automaton/rl/rustmage_experiment.py'))})
    report.setdefault('invocations',[]).append(invocation)
    report.update(status='running', training_games_completed=(first_round-1)*args.games_per_round,
                  training_games_target=args.rounds*args.games_per_round)
    report.pop('after', None); report.pop('comparison', None)
    atomic_json(report_path,report)
    batcher = collector = None
    validator = GameValidator(model, clip_ratio=trainer.clip_ratio) if args.validation_fraction else None
    def update_game_counts():
        report['collected_games_completed'] = sum(len(r['games']) for r in report['rounds'])
        report['optimization_games_completed'] = sum(r['split']['training']['terminal_games'] if 'split' in r
            else sum(bool(g['trajectories']) for g in r['games']) for r in report['rounds'])
        report['validation_games_completed'] = sum(r.get('split',{}).get('validation',{}).get('attempted_games',0)
            for r in report['rounds'])
        report['validation_terminal_games_completed'] = sum(r.get('split',{}).get('validation',{}).get('terminal_games',0)
            for r in report['rounds'])
    update_game_counts()
    def collection_version():
        return collector.version if collector else batcher.version
    def collection_stats():
        return collector.stats() if collector else batcher.stats()
    async def sync_weights(version):
        if collector: await collector.sync_weights(model,version)
        else: batcher.set_version(version)
    async def collect(**options):
        if collector: return await collector.run_games(**options)
        return await run_games(workers=args.workers, model=model, batcher=batcher, **options)
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
        evaluation_started=time.monotonic()
        rows=await collect(games=args.eval_games,
            seed_base=args.seed+10_000_000,mode=mode,
            version=collection_version(),max_own_turns=args.eval_horizon,progress=progress(label))
        result=dict(summary=goldfish_summary(rows,args.eval_horizon),games=rows,
                    seconds=time.monotonic()-evaluation_started)
        print(json.dumps(dict(stage=label,summary=result['summary'])),flush=True)
        return result
    def record_evaluation(round_id, result):
        checkpoint=output/'checkpoints'/f'round-{round_id:04d}'/'policy.keras'
        point=dict(round=round_id,training_games=round_id*args.games_per_round,
            checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            **result)
        point['optimization_games'] = sum(r['split']['training']['terminal_games'] if 'split' in r
            else sum(bool(g['trajectories']) for g in r['games']) for r in report['rounds'] if r['round']<=round_id)
        point['validation_games'] = sum(r.get('split',{}).get('validation',{}).get('attempted_games',0)
            for r in report['rounds'] if r['round']<=round_id)
        point['comparison_to_initial']=paired_improvement(report['before']['games'],result['games'],args.eval_horizon)
        report.setdefault('evaluations',[]).append(point)
        atomic_json(report_path,report)
        return point
    async def evaluate_checkpoint(round_id):
        existing=next((e for e in report.get('evaluations',[]) if e['round']==round_id),None)
        if existing is not None:return existing
        result=await evaluate(f'eval-{round_id*args.games_per_round:04d}','goldfish')
        return record_evaluation(round_id,result)
    try:
        if args.collector == 'processes' and config['inference_processes'] > 1:
            from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector
            collector = ProcessGameCollector(model,workers=args.workers,
                inference_processes=config['inference_processes'],wait_ms=args.inference_wait_ms)
            await collector.start(version=first_round-1)
        else:
            batcher = InferenceBatcher(model,max_batch=args.workers,wait_ms=args.inference_wait_ms)
            await batcher.start()
            batcher.set_version(first_round-1)
            if args.collector == 'processes':
                from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector
                collector = ProcessGameCollector(model,batcher,workers=args.workers)
                await collector.start(version=first_round-1)
        started = time.monotonic()
        if 'baseline' not in report:
            report['baseline']=await evaluate('baseline','baseline');atomic_json(report_path,report)
        if 'before' not in report:
            report['before']=await evaluate('before','goldfish');atomic_json(report_path,report)
        if not any(e['round']==0 for e in report.get('evaluations',[])):
            record_evaluation(0,report['before'])
        # A resumed boundary evaluation must finish before collecting more data.
        completed=first_round-1
        if completed and args.eval_every_games and completed*args.games_per_round % args.eval_every_games == 0:
            await evaluate_checkpoint(completed)
        for round_id in range(first_round,args.rounds+1):
            collection_started=time.monotonic()
            rows=await collect(games=args.games_per_round,
                seed_base=args.seed+round_id*args.games_per_round,
                index_base=(round_id-1)*args.games_per_round,mode='selfplay',
                version=collection_version(),output=output/'rollouts',
                max_own_turns=args.train_horizon,progress=progress(f'train-{round_id}'))
            train_games, heldout_games, partition = split_games(rows, fraction=args.validation_fraction,
                                                               seed=args.seed, round_id=round_id)
            validation_ids = set(partition['validation_indices'])
            for row in rows:
                row['partition'] = 'validation' if row['index'] in validation_ids else 'training'
            trajectories=[t for r in train_games for t in r['trajectories']]
            stats=dict(round=round_id,collection_seconds=time.monotonic()-collection_started,
                       outcomes=dict(Counter(r['status'] for r in rows)),games=rows,
                       split=partition,
                       seat_wins=dict(Counter(str(r.get('winner')) for r in rows if 'winner' in r)))
            if collector:
                frozen = collection_stats()
                stats['collection_policy'] = dict(version=frozen['version'],weights_sha256=frozen['weights_sha256'])
            if not trajectories: raise RuntimeError('No verified terminal games; refusing a fabricated update')
            normalization = training_normalization(train_games,trainer.advantage_scale_floor) if validator else None
            # All workers finished; the frozen behavior model can now change.
            update_started=time.monotonic()
            stats['update']=await asyncio.to_thread(trainer.update_files,trajectories)
            stats['update_seconds']=time.monotonic()-update_started
            if validator:
                validation_started = time.monotonic()
                stats['generalization'] = dict(
                    training=await asyncio.to_thread(validator.evaluate,train_games,normalization),
                    validation=await asyncio.to_thread(validator.evaluate,heldout_games,normalization))
                stats['validation_seconds'] = time.monotonic()-validation_started
                print(json.dumps(dict(stage='validation',round=round_id,split=partition,
                    metrics={name:{k:value[k] for k in ('value_before','value_after','policy_surrogate_gain')}
                             for name,value in stats['generalization'].items()})),flush=True)
            stats['checkpoint']=str(output/'checkpoints'/f'round-{round_id:04d}')
            save_checkpoint(output/'checkpoints',model,trainer,
                dict(round=round_id,config=config,round_result=stats),policy_type=RustMagePolicy)
            sync_started=time.monotonic()
            await sync_weights(round_id)
            stats['inference_sync_seconds']=time.monotonic()-sync_started
            report['rounds'].append(stats);report['batching']=collection_stats()
            report['training_games_completed']=round_id*args.games_per_round
            update_game_counts()
            atomic_json(report_path,report)
            print(json.dumps(dict(stage='update',round=round_id,metrics=stats['update'])),flush=True)
            if args.eval_every_games and round_id*args.games_per_round % args.eval_every_games == 0:
                await evaluate_checkpoint(round_id)
        final=await evaluate_checkpoint(args.rounds)
        report['after']={key:final[key] for key in ('summary','games')}
        report['comparison']=paired_improvement(report['before']['games'],report['after']['games'],args.eval_horizon)
        report['seconds_this_invocation']=time.monotonic()-started
        invocation.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                          seconds=report['seconds_this_invocation'])
        timed=[r for r in report['rounds'] if 'collection_seconds' in r]
        if timed:
            games=sum(len(r['games']) for r in timed)
            collection=sum(r['collection_seconds'] for r in timed)
            updates=sum(r['update_seconds'] for r in timed)
            validation=sum(r.get('validation_seconds',0.) for r in timed)
            report['throughput']=dict(timed_selfplay_games=games,collection_seconds=collection,
                update_seconds=updates,validation_seconds=validation,
                collection_games_per_minute=60*games/collection,
                collection_and_updates_games_per_minute=60*games/(collection+updates),
                collection_updates_validation_games_per_minute=60*games/(collection+updates+validation))
        report['batching']=collection_stats()
        report['status']='complete'
        atomic_json(report_path,report)
        if args.plot:
            from .plot_rustmage_training import plot_report
            plot_report(report_path)
        print(json.dumps(dict(stage='result',comparison=report['comparison'],report=str(report_path))),flush=True)
    finally:
        if collector: await collector.close()
        if batcher: await batcher.close()


if __name__ == '__main__': asyncio.run(run(parser().parse_args()))
