"""Perspective-safe, paired comparison of two completed Standard arena runs."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from ..baselines.standard_decks import STANDARD_DECKS


LEARNER_FIELDS = frozenset(('learner_label', 'learner_checkpoint', 'learner_files_sha256'))
OUTCOMES = frozenset(('win', 'loss', 'draw'))
FAILURES = frozenset(('cutoff', 'error'))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _cell_parts(cell):
    parts = cell.split('--')
    if len(parts) != 3 or parts[0] not in STANDARD_DECKS or parts[1] not in STANDARD_DECKS \
            or parts[2] not in ('aggro', 'tactical', 'checkpoint'):
        raise ValueError(f'Invalid arena cell ID: {cell!r}')
    return parts


def _expected_cells(manifest):
    learners = manifest['learner_decks']
    opponents = manifest['opponent_decks']
    styles = manifest['opponents']
    panel = manifest['panel']
    if panel not in ('mirrors', 'matrix'):
        raise ValueError('Unknown arena panel')
    if any(not isinstance(group, list) or not group or len(group) != len(set(group))
           for group in (learners, opponents, styles)):
        raise ValueError('Invalid or duplicated arena deck/opponent configuration')
    expected = [f'{learner}--{opponent}--{style}' for style in styles
                for learner in learners for opponent in opponents
                if panel == 'matrix' or learner == opponent]
    if not expected or expected != manifest.get('cells'):
        raise ValueError('Arena cells differ from their declared panel')
    for cell in expected:
        _cell_parts(cell)
    return expected


def _validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get('schema') != 1:
        raise ValueError('Unsupported arena manifest')
    cells = _expected_cells(manifest)
    count = manifest.get('games_per_cell')
    base = manifest.get('seed_base')
    if type(count) is not int or count < 4 or count % 4:
        raise ValueError('Games per cell must balance seats and starts')
    if type(base) is not int or base < 0 or base + count * len(cells) > 2**64:
        raise ValueError('Invalid arena seed range')
    for name in ('learner_label', 'learner_checkpoint'):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError(f'Missing {name}')
    files = manifest.get('learner_files_sha256')
    if not isinstance(files, dict) or manifest['learner_checkpoint'] not in files:
        raise ValueError('Learner checkpoint hash missing')
    opponent_files = manifest.get('opponent_files_sha256')
    if not isinstance(opponent_files, dict):
        raise ValueError('Opponent checkpoint hash manifest missing')
    if 'checkpoint' in manifest['opponents']:
        if manifest.get('opponent_checkpoint') not in opponent_files:
            raise ValueError('Frozen opponent checkpoint hash missing')
    elif opponent_files or manifest.get('opponent_checkpoint') is not None:
        raise ValueError('Script-only arena has an opponent checkpoint')
    return cells


def _validate_row(row, *, manifest, cell, cell_index, offset):
    learner_deck, opponent_deck, style = _cell_parts(cell)
    count = manifest['games_per_cell']
    index = cell_index * count + offset
    seat = offset % 2
    decks = [opponent_deck, opponent_deck]
    decks[seat] = learner_deck
    order = tuple(STANDARD_DECKS)
    expected = dict(index=index, seed=manifest['seed_base'] + index,
                    cell_id=cell, learner_deck=learner_deck,
                    opponent_deck=opponent_deck, opponent_style=style,
                    learner_seat=seat, starting_player=f'P{(offset // 2) % 2}',
                    decks=decks, pairing=sorted((learner_deck, opponent_deck), key=order.index),
                    partition='evaluation',
                    opponent_kind='frozen_neural' if style == 'checkpoint' else 'scripted',
                    behavior_version=0, learner_label=manifest['learner_label'],
                    opponent_label=manifest['opponent_label'] if style == 'checkpoint'
                        else 'StandardTactical' if style == 'tactical' else 'StandardAggro')
    if style == 'checkpoint':
        expected.update(opponent_checkpoint_sha256=manifest['opponent_files_sha256'][
            manifest['opponent_checkpoint']], opponent_version=0)
    if not isinstance(row, dict) or any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Arena row differs from its schedule: {cell} game {index}')
    if row.get('trajectories') != []:
        raise ValueError('Arena row contains training trajectories')
    status = row.get('status')
    if status not in OUTCOMES | FAILURES:
        raise ValueError(f'Invalid arena outcome: {cell} game {index}')
    if status in ('win', 'loss'):
        winner = 0 if status == 'win' else 1
        if row.get('winner_seat') != winner or row.get('draw') is not False \
                or row.get('learner_won') is not (winner == seat):
            raise ValueError(f'Inconsistent arena winner: {cell} game {index}')
    elif status == 'draw':
        if row.get('winner_seat') is not None or row.get('draw') is not True \
                or row.get('learner_won') is not False:
            raise ValueError(f'Inconsistent arena draw: {cell} game {index}')
    elif row.get('winner_seat') is not None or row.get('learner_won') is True:
        raise ValueError(f'Failed arena game claims a winner: {cell} game {index}')
    return row


def read_run(root):
    """Read a complete arena and independently validate every scheduled game."""
    root = Path(root).resolve()
    manifest = _read_json(root / 'manifest.json')
    cells = _validate_manifest(manifest)
    summary = _read_json(root / 'summary.json')
    if summary.get('complete') is not True:
        raise ValueError(f'Arena is not complete: {root}')
    rows = {}
    count = manifest['games_per_cell']
    for cell_index, cell in enumerate(cells):
        path = root / 'cells' / cell / 'games.jsonl'
        found = {}
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                index = row.get('index') if isinstance(row, dict) else None
                offset = index - cell_index * count if type(index) is int else -1
                if not 0 <= offset < count or index in found:
                    raise ValueError(f'Unexpected or duplicate game in {path}')
                found[index] = _validate_row(row, manifest=manifest, cell=cell,
                                             cell_index=cell_index, offset=offset)
        expected = set(range(cell_index * count, (cell_index + 1) * count))
        if set(found) != expected:
            raise ValueError(f'Incomplete arena cell: {cell}')
        rows[cell] = [found[index] for index in sorted(found)]
    return dict(root=str(root), manifest=manifest, rows=rows,
                manifest_sha256=_sha256(root / 'manifest.json'))


def _score(row):
    if row['status'] == 'draw':
        return .5
    return float(row['learner_won'])


def _paired_summary(pairs, *, bootstrap_samples, rng):
    left = Counter(a['status'] for a, _ in pairs)
    right = Counter(b['status'] for _, b in pairs)
    usable = [(a, b) for a, b in pairs if a['status'] in OUTCOMES and b['status'] in OUTCOMES]
    result = dict(scheduled=len(pairs), paired_completed=len(usable),
                  excluded=len(pairs) - len(usable),
                  baseline_status=dict(sorted(left.items())),
                  candidate_status=dict(sorted(right.items())),
                  baseline_failed=sum(left[x] for x in FAILURES),
                  candidate_failed=sum(right[x] for x in FAILURES))
    if not usable:
        result.update(baseline_score=None, candidate_score=None, score_delta=None,
                      paired_bootstrap_95=None)
        return result
    baseline = np.asarray([_score(a) for a, _ in usable], np.float64)
    candidate = np.asarray([_score(b) for _, b in usable], np.float64)
    differences = candidate - baseline
    result.update(baseline_score=float(baseline.mean()),
                  candidate_score=float(candidate.mean()),
                  score_delta=float(differences.mean()))
    if bootstrap_samples:
        means = []
        for start in range(0, bootstrap_samples, 256):
            size = min(256, bootstrap_samples - start)
            indices = rng.integers(0, len(differences), size=(size, len(differences)))
            means.extend(differences[indices].mean(axis=1).tolist())
        result['paired_bootstrap_95'] = [float(np.quantile(means, .025)),
                                         float(np.quantile(means, .975))]
    else:
        result['paired_bootstrap_95'] = None
    return result


def compare_runs(baseline_root, candidate_root, *, bootstrap_samples=2000, bootstrap_seed=0):
    """Compare outcomes on exactly matched seeds, roles, decks, and opponent."""
    if type(bootstrap_samples) is not int or bootstrap_samples < 0:
        raise ValueError('Bootstrap sample count must be nonnegative')
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError('Bootstrap seed must be nonnegative')
    baseline = read_run(baseline_root)
    candidate = read_run(candidate_root)
    for name in (baseline, candidate):
        if not name['manifest'].get('mode') in ('greedy', 'sampled'):
            raise ValueError('Unknown arena policy mode')
    left = {key: value for key, value in baseline['manifest'].items() if key not in LEARNER_FIELDS}
    right = {key: value for key, value in candidate['manifest'].items() if key not in LEARNER_FIELDS}
    if left != right:
        differing = sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key))
        raise ValueError('Arena schedules or opponents differ: ' + ', '.join(differing))
    if baseline['root'] == candidate['root']:
        raise ValueError('Choose two distinct arena runs')
    rng = np.random.default_rng(bootstrap_seed)
    cells = {}
    all_pairs = []
    play_pairs, draw_pairs = [], []
    for cell in baseline['manifest']['cells']:
        pairs = list(zip(baseline['rows'][cell], candidate['rows'][cell], strict=True))
        for a, b in pairs:
            identity = ('index', 'seed', 'decks', 'learner_seat', 'starting_player',
                        'opponent_style', 'opponent_checkpoint_sha256')
            if any(a.get(key) != b.get(key) for key in identity):
                raise ValueError(f'Paired game identity differs in {cell}')
        cells[cell] = _paired_summary(pairs, bootstrap_samples=bootstrap_samples, rng=rng)
        all_pairs.extend(pairs)
        for a, b in pairs:
            (play_pairs if a['starting_player'] == f"P{a['learner_seat']}" else draw_pairs).append((a, b))
    aggregate = _paired_summary(all_pairs, bootstrap_samples=bootstrap_samples, rng=rng)
    aggregate['on_play'] = _paired_summary(play_pairs, bootstrap_samples=bootstrap_samples, rng=rng)
    aggregate['on_draw'] = _paired_summary(draw_pairs, bootstrap_samples=bootstrap_samples, rng=rng)
    return dict(schema=1, baseline=dict(root=baseline['root'], label=baseline['manifest']['learner_label'],
                                        checkpoint=baseline['manifest']['learner_checkpoint'],
                                        files_sha256=baseline['manifest']['learner_files_sha256'],
                                        manifest_sha256=baseline['manifest_sha256']),
                candidate=dict(root=candidate['root'], label=candidate['manifest']['learner_label'],
                               checkpoint=candidate['manifest']['learner_checkpoint'],
                               files_sha256=candidate['manifest']['learner_files_sha256'],
                               manifest_sha256=candidate['manifest_sha256']),
                schedule=dict(cells=baseline['manifest']['cells'],
                              games_per_cell=baseline['manifest']['games_per_cell'],
                              seed_base=baseline['manifest']['seed_base'],
                              opponent_checkpoint=baseline['manifest']['opponent_checkpoint'],
                              opponent_files_sha256=baseline['manifest']['opponent_files_sha256']),
                bootstrap=dict(samples=bootstrap_samples, seed=bootstrap_seed),
                clean=aggregate['baseline_failed'] == aggregate['candidate_failed'] == 0,
                cells=cells, aggregate=aggregate)
