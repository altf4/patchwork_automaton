"""Parallel CPU collection, one GPU imitation learner, and CPU baseline evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from patchwork_automaton.baselines.standard_decks import STANDARD_DECKS


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def source_hashes():
    paths = sorted(p for root in ('src', 'examples') for p in Path(root).rglob('*.py'))
    paths += sorted(Path('src/patchwork_automaton/data').glob('*.json'))
    paths += [Path('pyproject.toml'), Path('.local/rustmage/python/rustmage.py'),
              Path('.local/rustmage/data/observation-schema.txt'),
              Path('.local/rustmage/target/release/librustmage.so')]
    if os.environ.get('PATCHWORK_STANDARD_PLAYBACK', 'direct') == 'direct':
        from patchwork_automaton.neural.direct_standard import library_path
        paths.append(library_path())
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(args):
    if args.games_per_stage % 120 or args.stages < 1:
        raise ValueError('Each stage must divide into six balanced 80/20 pairings')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    report_path = args.output / 'experiment.json'
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'}
    hashes = source_hashes()
    if args.resume and report_path.exists():
        report = json.loads(report_path.read_text())
        if report['config'] != config or report['source_sha256'] != hashes:
            raise ValueError('Resume requires the same frozen source and configuration')
    else:
        report = dict(status='running', policy_scope='shared', training_decks=list(STANDARD_DECKS), config=config, source_sha256=hashes,
            started_at=datetime.now(timezone.utc).isoformat(), stages=[],
            methodology='Six unordered pairings including mirrors; 80/20 whole games stratified by pairing, '
                'deck seat and starting seat. One shared policy and optimizer across all three decks, continued through stages. '
                'Two epochs on each new chunk of training games. Validation never trains. '
                'The same checkpoint pilots every deck against every baseline with independent fixed evaluation seeds, '
                'in sampled and greedy modes. Mainboards only; full gameplay including mulligans. '
                'Historical cross-pair win rates use the prior 3000-game audit; mirror reference is 50%.')
    atomic_json(report_path, report)
    env = dict(os.environ, PYTHONPATH='src:.',
        PATCHWORK_OBSERVATION_PIPELINE='python', OPENBLAS_NUM_THREADS='1',
        TF_NUM_INTRAOP_THREADS=str(args.training_threads), TF_NUM_INTEROP_THREADS='2',
        TF_ENABLE_ONEDNN_OPTS='0', TF_CPP_MIN_LOG_LEVEL='3', KERAS_BACKEND='tensorflow')
    datasets = []
    logs = args.output / 'logs'
    logs.mkdir(exist_ok=True)

    def command(command, log_path, *, cpu_only=False):
        print(json.dumps(dict(event='launch', command=command, log=str(log_path))), flush=True)
        with log_path.open('a') as stream:
            child_env = dict(env)
            if cpu_only: child_env['CUDA_VISIBLE_DEVICES'] = '-1'
            subprocess.run(command, env=child_env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(json.dumps(dict(event='finished', log=str(log_path))), flush=True)

    try:
        for stage in range(1, args.stages + 1):
            dataset = (args.dataset_root or args.output / 'datasets') / f'stage-{stage}'
            datasets.append(dataset)
            if any(r['stage'] == stage and r['status'] == 'complete' for r in report['stages']):
                continue
            point = next((r for r in report['stages'] if r['stage'] == stage), None)
            if point is None:
                point = dict(stage=stage, cumulative_games=stage * args.games_per_stage, status='collecting')
                report['stages'].append(point)
            report.update(status='running', active_stage=stage)
            atomic_json(report_path, report)
            collect_command = [sys.executable, '-u', '-m', 'examples.collect_standard_demonstrations',
                '--output', str(dataset), '--games-per-pair', str(args.games_per_stage // 6),
                '--workers', str(args.workers), '--runners', str(args.runners), '--seed', str(args.seed + (stage - 1) * args.games_per_stage),
                '--index-base', str((stage - 1) * args.games_per_stage),
                '--game-timeout', str(args.game_timeout), '--max-turns', str(args.max_turns)]
            saved_dataset = dataset / 'dataset.json'
            existing = json.loads(saved_dataset.read_text()) if saved_dataset.exists() else None
            if existing is not None and existing['status'] == 'complete':
                from examples.collect_standard_demonstrations import schedule
                from patchwork_automaton.neural.standard_features import StandardEncoder
                expected = list(schedule(args.games_per_stage // 6,
                    args.seed + (stage - 1) * args.games_per_stage,
                    index_base=(stage - 1) * args.games_per_stage))
                if (existing['encoder'] != json.loads(json.dumps(StandardEncoder().manifest()))
                        or len(existing['games']) != len(expected)
                        or any(any(actual[k] != planned[k] for k in planned)
                               for actual, planned in zip(existing['games'], expected))):
                    raise ValueError('Existing dataset does not match this experiment')
                print(json.dumps(dict(event='reuse_complete_dataset', path=str(saved_dataset))), flush=True)
            else:
                if dataset.exists(): collect_command.append('--resume')
                command(collect_command, logs / f'collect-stage-{stage}.log', cpu_only=True)
            point['status'] = 'training_and_evaluating'
            atomic_json(report_path, report)
            training_command = [sys.executable, '-u', '-m', 'examples.train_standard_imitation',
                '--datasets', *(str(d) for d in datasets), '--output', str(args.output / 'agent'),
                '--stage', str(stage), '--epochs', str(args.epochs),
                '--train-batch', str(args.train_batch), '--unroll', str(args.unroll),
                '--eval-games', str(args.eval_games), '--eval-workers', str(args.eval_workers),
                '--inference-processes', str(args.inference_processes),
                '--learner-device', args.learner_device, '--learner-gru-backend', args.learner_gru_backend,
                '--game-timeout', str(args.game_timeout), '--max-turns', str(args.max_turns)]
            command(training_command, logs / f'train-stage-{stage}-shared.log')
            point['status'] = 'complete'
            point['agent'] = str(args.output / 'agent' / f'stage-{stage}.json')
            if source_hashes() != hashes:
                raise AssertionError('Source or engine changed during the experiment')
            atomic_json(report_path, report)
        report.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(),
            total_games=args.games_per_stage * args.stages, sources_unchanged=True)
        atomic_json(report_path, report)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        atomic_json(report_path, report)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--games-per-stage', type=int, default=12000)
    p.add_argument('--stages', type=int, default=3)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--runners', type=int, default=2)
    p.add_argument('--learner-device', choices=('cpu','gpu'), default='gpu')
    p.add_argument('--learner-gru-backend', choices=('default','unfused','cudnn-memory'), default='default')
    p.add_argument('--inference-processes', type=int, default=2)
    p.add_argument('--training-threads', type=int, default=4)
    p.add_argument('--dataset-root', type=Path, help='Reuse previously collected expert games without copying or relabeling')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--train-batch', type=int, default=32)
    p.add_argument('--unroll', type=int, default=8)
    p.add_argument('--eval-games', type=int, default=400)
    p.add_argument('--eval-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=302610270000)
    p.add_argument('--game-timeout', type=float, default=60)
    p.add_argument('--max-turns', type=int, default=50)
    p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
