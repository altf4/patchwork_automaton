"""Fresh supervised policy imitation of a fixed, game-partitioned expert dataset.

Collect first with examples.collect_rustmage_demonstrations. No PPO/value
training is used. TensorFlow is imported only in the learner, not spawned games.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '4')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--train-batch', type=int, default=32, help='Concurrent seat sequences per minibatch')
    p.add_argument('--unroll', type=int, default=8, help='Decisions per truncated backpropagation chunk')
    p.add_argument('--learning-rate', type=float, default=.0003)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--inference-processes', type=int, default=2)
    p.add_argument('--eval-games', type=int, default=40)
    p.add_argument('--eval-horizon', type=int, default=25)
    p.add_argument('--seed', type=int, default=721)
    p.add_argument('--resume', action='store_true')
    return p


def validate_dataset(dataset):
    if dataset['status'] != 'complete':
        raise ValueError('Finish expert collection before training')
    seen, paths, train_ids, val_ids = set(), set(), set(), set()
    for batch in dataset['batches']:
        train, val = (set(batch['split'][k]) for k in ('training_indices', 'validation_indices'))
        if train & val or train | val != {g['index'] for g in batch['games']}:
            raise ValueError('Invalid whole-game partition')
        train_ids |= train
        val_ids |= val
        for game in batch['games']:
            if game['index'] in seen:
                raise ValueError('Repeated game')
            seen.add(game['index'])
            if (game['partition'] == 'training') != (game['index'] in train):
                raise ValueError('Partition label mismatch')
            if game['status'] not in ('win', 'loss', 'draw') or len(game['trajectories']) != 2:
                raise ValueError('Require complete expert games with both seats; do not replace failures silently')
            for t in game['trajectories']:
                if t['path'] in paths or not Path(t['path']).is_file() or 'reward' in t:
                    raise ValueError('Missing, duplicated, or reward-labeled expert trajectory')
                paths.add(t['path'])
    if train_ids & val_ids or len(seen) != dataset['completed_games']:
        raise ValueError('Game partition leakage')
    return dict(training_games=len(train_ids), validation_games=len(val_ids), trajectories=len(paths))


async def run(args):
    import keras
    import numpy as np
    from patchwork_automaton.neural.batching import example_batch
    from patchwork_automaton.neural.experiment import atomic_json, load_checkpoint, make_trainer, save_checkpoint
    from patchwork_automaton.neural.model import SeparateActorCritic
    from patchwork_automaton.neural.rustmage_features import RustMageEncoder
    from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
    from patchwork_automaton.rl.rustmage_experiment import goldfish_summary, paired_improvement
    from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector

    if min(args.epochs, args.train_batch, args.unroll, args.workers, args.eval_games, args.eval_horizon) < 1:
        raise ValueError('Positive counts required')
    dataset_path = args.dataset / 'dataset.json'
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    dataset = json.loads(dataset_path.read_text())
    audit = validate_dataset(dataset)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ('output', 'resume')}
    config.update(algorithm='imitation', dataset_sha256=dataset_hash)
    report_path = args.output / 'report.json'
    if args.resume:
        model, trainer, state = load_checkpoint(args.output / 'checkpoints')
        if config != state['config']:
            raise ValueError('Resume requires the same dataset and configuration')
        report = json.loads(report_path.read_text())
        first = state['round'] + 1
        report['rounds'] = [r for r in report['rounds'] if r['round'] < first]
        if state.get('round_result') and not any(r['round'] == state['round'] for r in report['rounds']):
            report['rounds'].append(state['round_result'])
        if [r['round'] for r in report['rounds']] != list(range(1, first)):
            raise ValueError('Missing completed imitation updates')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        keras.utils.set_random_seed(args.seed)
        encoder = RustMageEncoder()
        model = SeparateActorCritic(encoder.card_count, encoder.feature_spec)
        model(example_batch(model.memory_size, model.feature_spec), training=False)
        trainer = make_trainer(model, config)
        first = 1
        report = dict(config=config, initialization='fresh random weights', status='running',
            training='Expert legal-action cross entropy only; no RL, reward, value fit, or entropy bonus',
            methodology='Whole games partitioned before minibatching; both seats stay together. '
                'Two epochs on each new training batch. Memory resets per seat and replays in order; '
                'backpropagation is truncated to unroll decisions. Validation never updates weights or sampler.',
            dataset_audit=audit, encoder=encoder.manifest(), rounds=[], evaluations=[],
            baseline_teacher=dataset['teacher'], source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), Path('src/patchwork_automaton/neural/imitation.py'),
                    Path('src/patchwork_automaton/neural/demonstrations.py'),
                    Path('src/patchwork_automaton/rl/rustmage_experiment.py')]})
        save_checkpoint(args.output / 'checkpoints', model, trainer, dict(round=0, config=config), policy_type=RustMagePolicy)
    frozen_value = [w.copy() for w in model.critic.get_weights()]
    report['status'] = 'running'
    report['target_games'] = dataset['completed_games']
    atomic_json(report_path, report)
    collector = ProcessGameCollector(model, workers=args.workers, inference_processes=args.inference_processes, wait_ms=0)
    started = time.monotonic()
    await collector.start(version=first - 1)

    async def evaluation(round_id):
        if any(e['round'] == round_id for e in report['evaluations']):
            return
        point = dict(round=round_id, games_seen=sum(len(b['games']) for b in dataset['batches'][:round_id]))
        for name, greedy in (('sampled', False), ('greedy', True)):
            rows = await collector.run_games(games=args.eval_games, seed_base=args.seed + 10_000_000,
                mode='goldfish', version=collector.version, max_own_turns=args.eval_horizon, greedy=greedy)
            point[name] = dict(summary=goldfish_summary(rows, args.eval_horizon), games=rows)
        report['evaluations'].append(point)
        atomic_json(report_path, report)
        print(json.dumps(dict(stage='imitation_evaluation', round=round_id,
            games_seen=point['games_seen'], summaries={k: point[k]['summary'] for k in ('sampled', 'greedy')})), flush=True)

    try:
        if 'baseline' not in report:
            rows = await collector.run_games(games=args.eval_games, seed_base=args.seed + 10_000_000,
                mode='baseline', version=collector.version, max_own_turns=args.eval_horizon)
            report['baseline'] = dict(summary=goldfish_summary(rows, args.eval_horizon), games=rows)
            atomic_json(report_path, report)
        await evaluation(first - 1)
        for batch in dataset['batches'][first - 1:]:
            round_id = batch['batch']
            train = [g for g in batch['games'] if g['partition'] == 'training']
            validation = [g for g in batch['games'] if g['partition'] == 'validation']
            step_started = time.monotonic()
            fit = await asyncio.to_thread(trainer.fit, train)
            print(json.dumps(dict(stage='imitation_fit', round=round_id, steps=fit['steps_this_update'],
                cross_entropy=fit['epochs'][-1]['cross_entropy'], accuracy=fit['epochs'][-1]['accuracy'])), flush=True)
            training_metrics = await asyncio.to_thread(trainer.evaluate, train)
            validation_metrics = await asyncio.to_thread(trainer.evaluate, validation)
            if not all(np.array_equal(a, b) for a, b in zip(frozen_value, model.critic.get_weights())):
                raise AssertionError('Imitation must not update the value network')
            result = dict(round=round_id, games_seen=sum(len(b['games']) for b in dataset['batches'][:round_id]),
                split=batch['split'], fit=fit, training=training_metrics, validation=validation_metrics,
                seconds=time.monotonic() - step_started)
            save_checkpoint(args.output / 'checkpoints', model, trainer,
                dict(round=round_id, config=config, round_result=result), policy_type=RustMagePolicy)
            report['rounds'].append(result)
            atomic_json(report_path, report)
            print(json.dumps(dict(stage='imitation_update', round=round_id, games_seen=result['games_seen'],
                training={k: training_metrics[k] for k in ('cross_entropy', 'accuracy', 'non_pass_accuracy')},
                validation={k: validation_metrics[k] for k in ('cross_entropy', 'accuracy', 'non_pass_accuracy')})), flush=True)
            await collector.sync_weights(model, round_id)
            await evaluation(round_id)
        heldout = [g for b in dataset['batches'] for g in b['games'] if g['partition'] == 'validation']
        report['full_validation'] = await asyncio.to_thread(trainer.evaluate, heldout)
        if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != dataset_hash:
            raise AssertionError('The expert dataset changed during training')
        report['value_network_unchanged'] = True
        report['audit'] = dict(status='passed', **audit, dataset_unchanged=True,
            value_network_unchanged=True, rl_updates=0, validation_optimizer_updates=0)
        report['comparison_to_initial'] = paired_improvement(report['evaluations'][0]['sampled']['games'],
            report['evaluations'][-1]['sampled']['games'], args.eval_horizon)
        report['seconds_this_invocation'] = time.monotonic() - started
        report['status'] = 'complete'
        atomic_json(report_path, report)
        from examples.plot_rustmage_imitation import plot_report
        plot_report(report_path)
        print(json.dumps(dict(stage='imitation_complete', report=str(report_path),
            comparison=report['comparison_to_initial'], full_validation=report['full_validation'])), flush=True)
    finally:
        await collector.close()


if __name__ == '__main__':
    asyncio.run(run(parser().parse_args()))
