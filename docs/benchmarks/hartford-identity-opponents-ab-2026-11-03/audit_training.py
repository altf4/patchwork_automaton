"""Read-only audit of independent continuation arms and their preserved parent."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_static(launch):
    files = dict(launch['preserved_files'])
    files.update({str(Path(launch['project']) / name): digest
                  for name, digest in launch['source_hashes'].items()})
    files.update(launch['engine_files'])
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError('Preserved artifact changed: ' + path)
    return len(files)


def finite_checkpoint(folder):
    import h5py
    import numpy as np
    for name in ('optimizer.npz', 'critic-optimizer.npz'):
        with np.load(folder / name, allow_pickle=False) as arrays:
            if not arrays.files or any(not np.isfinite(arrays[k]).all() for k in arrays.files):
                raise ValueError('Invalid optimizer arrays: ' + str(folder / name))
    count = 0
    with zipfile.ZipFile(folder / 'policy.keras') as archive:
        with h5py.File(io.BytesIO(archive.read('model.weights.h5')), 'r') as model:
            def check(name, item):
                nonlocal count
                if isinstance(item, h5py.Dataset) and np.issubdtype(item.dtype, np.number):
                    count += 1
                    if not np.isfinite(item[()]).all():
                        raise ValueError('Nonfinite model array: ' + name)
            model.visititems(check)
    state = read(folder / 'state.json')
    if not count or 'trainer_rng_state' not in state:
        raise ValueError('Missing model or trainer RNG')
    return count


def audit_arm(launch, name):
    from examples.train_hartford_ppo import planned_training, checkpoint_hashes, source_manifest
    from patchwork_automaton.rl.standard_ppo import validate_collection
    root = Path(launch['arms'][name]['output'])
    report = read(root / 'report.json')
    config = report['config']
    if (report['status'] != 'complete' or report['completed_games'] != 50000
            or len(report['rounds']) != 50 or report['evaluations'] or report.get('evaluation_games') != 0
            or not config['skip_evaluations'] or config['rounds'] != 50
            or config['opponent_recipe'] != ('current' if name == 'control' else 'diverse')):
        raise ValueError('Incomplete arm or changed protocol: ' + name)
    manifest = read(root / 'manifest.json')
    if config != launch['arms'][name]['config'] or manifest['config'] != config:
        raise ValueError('Arm manifest configuration differs')
    if (manifest['pins']['source'] != source_manifest(Path(launch['project']))
            or manifest['pins']['readiness']['engine'] != launch['engine']):
        raise ValueError('Arm source or engine admission changed')
    if ([r['round'] for r in report['rounds']] != list(range(1,51))
            or read(root/'checkpoints/latest.json') != {'checkpoint':'round-0050'}):
        raise ValueError('Round order or latest checkpoint differs')
    if checkpoint_hashes(root/'checkpoints/round-0000') != launch['arms'][name]['initial_checkpoint_hashes']:
        raise ValueError('Initial branch checkpoint changed')
    if set(report['checkpoints']) != {str(n) for n in range(51)}:
        raise ValueError('Missing committed checkpoint')
    parent_state = read(config['fork']['parent_checkpoint'] + '/state.json')
    initial = read(root / 'checkpoints/round-0000/state.json')
    for key in ('optimizer_steps', 'critic_optimizer_steps', 'trainer_rng_state'):
        if initial[key] != parent_state[key]:
            raise ValueError('Fork reset training state: ' + key)
    for filename, digest in config['fork']['parent_hashes'].items():
        if filename != 'state.json' and sha(root / 'checkpoints/round-0000' / filename) != digest:
            raise ValueError('Fork changed initial policy or optimizer: ' + filename)
    seeds, partitions, outcomes = set(), Counter(), Counter()
    for number in range(51):
        folder = root / 'checkpoints' / f'round-{number:04d}'
        if checkpoint_hashes(folder) != report['checkpoints'][str(number)]:
            raise ValueError('Changed checkpoint: ' + str(folder))
        state = read(folder / 'state.json')
        if state['round'] != number or state['config'] != config:
            raise ValueError('Checkpoint metadata differs')
        finite_checkpoint(folder)
        if not number:
            continue
        result = report['rounds'][number - 1]
        if result != state['round_result'] or sha(result['games_file']) != result['games_sha256']:
            raise ValueError('Round summary changed')
        rows = read(result['games_file'])
        planned = planned_training(config, number)
        for game in planned:
            if game['opponent_kind'] == 'frozen_neural':
                version = game['opponent_version']
                external = config['fork']['external_opponents'].get(str(version))
                game['opponent_checkpoint_sha256'] = (external['hashes']['policy.keras'] if external
                    else report['checkpoints'][str(version)]['policy.keras'])
        validate_collection(rows, planned, number - 1)
        if len(rows) != 1000 or Counter(r['partition'] for r in rows) != {'training': 800, 'validation': 200}:
            raise ValueError('Wrong round size or partition')
        for row in rows:
            if row['seed'] in seeds:
                raise ValueError('Replayed collection seed')
            seeds.add(row['seed']); partitions[row['partition']] += 1
            outcomes['draw' if row['draw'] else 'win' if row['learner_won'] else 'loss'] += 1
    if len(seeds) != 50000 or list((root / 'evaluation').rglob('game-*.json')):
        raise ValueError('Wrong collection/evaluation count')
    failures = list((root / 'rollouts').glob('round-*/failures/*.json'))
    if failures:
        raise ValueError('Unresolved failure artifacts: ' + str(failures))
    return dict(passed=True, output=str(root), games=len(seeds), partitions=dict(partitions), outcomes=dict(outcomes),
                checkpoints=51, final_policy_sha256=report['checkpoints']['50']['policy.keras'],
                seed_min=min(seeds), seed_max=max(seeds), evaluation_games=0,
                initial_optimizer_steps=initial['optimizer_steps'],
                initial_critic_optimizer_steps=initial['critic_optimizer_steps'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--arm', choices=('control', 'diverse')); args = parser.parse_args()
    launch = read(args.launch)
    checked = verify_static(launch)
    arms = {name: audit_arm(launch, name) for name in ([args.arm] if args.arm else ['control', 'diverse'])}
    if len(arms) == 2:
        for key in ('seed_min', 'seed_max', 'initial_optimizer_steps', 'initial_critic_optimizer_steps'):
            if arms['control'][key] != arms['diverse'][key]:
                raise ValueError('Unmatched arm initialization or schedule: ' + key)
    verify_static(launch)
    result = dict(passed=True, observed_at=datetime.now(timezone.utc).isoformat(),
                  preserved_files_checked=checked, arms=arms, games=sum(a['games'] for a in arms.values()))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
