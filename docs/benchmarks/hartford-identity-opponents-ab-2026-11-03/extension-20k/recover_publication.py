"""Recover already adjudicated outcomes after post-game transport cleanup failed.

No engine, inference process or game is started by this repair.
"""
import base64
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def module(launch):
    specification = importlib.util.spec_from_file_location('original_extension', launch['evaluation']['script'])
    loaded = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loaded)
    return loaded


def validated_rows(launch, *, include_adjudications=False):
    from patchwork_automaton.rl.hartford_evaluation import validate_row
    evaluation = module(launch)
    root = Path(launch['evaluation']['output'])
    original_manifest = evaluation.read(root / 'manifest.json')
    evaluation.verify_files(original_manifest['files'])
    evaluation.verify_files(launch['recovery']['preserved_files'])
    planned = evaluation.read(root / 'schedule.json')
    if evaluation.value_sha(planned) != original_manifest['schedule_sha256']:
        raise ValueError('Original extension schedule changed')
    expected = {row['index']: row for row in planned}
    rows = {}
    for path in sorted((root / 'games').glob('*.json')):
        envelope = evaluation.read(path)
        row = envelope['row']
        index = row['index']
        if (set(envelope) != {'row', 'sha256'} or evaluation.value_sha(row) != envelope['sha256']
                or index not in expected or index in rows or path.name != f'game-{index}.json'):
            raise ValueError('Invalid or duplicate durable outcome')
        validate_row(row, expected[index], checkpoint_round=150)
        if row.get('adjudicated') and row['diagnostic_hashes'] != evaluation.diagnostic_hashes(root, row):
            raise ValueError('Changed adjudication diagnostics')
        rows[index] = row
    recovery = launch['recovery']
    failed_attempt = evaluation.read(recovery['failed_attempt'])
    if (failed_attempt['error'] != "TimeoutError('Inference reset timed out')"
            or failed_attempt['status'] != 'failed'
            or set(path.name for path in (root / 'failures').glob('*.json')) != {Path(recovery['failed_attempt']).name}):
        raise ValueError('Unexpected or unpreserved failure')
    missing = set(expected) - rows.keys()
    if include_adjudications:
        if missing != set(recovery['recovered_ids']):
            raise ValueError('Missing outcomes differ from approved publication repair')
        for index in sorted(missing):
            path = root / 'diagnostics' / f'game-{index:06d}' / 'adjudication.json'
            row = evaluation.read(path)
            validate_row(row, expected[index], checkpoint_round=150)
            if (row['status'] != 'draw' or not row['adjudicated']
                    or row['reason'] != 'wall_clock_timeout' or row['timeout_seconds'] != 20):
                raise ValueError('Missing game has no valid original timeout adjudication')
            row = dict(row, actions=None, diagnostic_hashes=evaluation.diagnostic_hashes(root, row))
            rows[index] = row
    if set(rows) != set(expected):
        raise ValueError('Incomplete extension outcomes')
    counts = Counter((r['learner_deck'], r['opponent_deck'], r['learner_seat'], r['starting_player'])
                     for r in rows.values())
    if len(rows) != 20000 or len(counts) != 400 or set(counts.values()) != {50}:
        raise ValueError('Unbalanced extension outcomes')
    return [rows[index] for index in sorted(rows)]


def publish(launch):
    from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS
    from patchwork_automaton.rl.hartford_evaluation import atomic_json, summarize
    evaluation = module(launch)
    root = Path(launch['evaluation']['output'])
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = validated_rows(launch, include_adjudications=True)
        for row in rows:
            if row['index'] in launch['recovery']['recovered_ids']:
                path = root / 'games' / f"game-{row['index']}.json"
                if path.exists():
                    raise FileExistsError('Publication recovery already applied')
                atomic_json(path, dict(row=row, sha256=evaluation.value_sha(row)))
        validated_rows(launch)
        result = dict(status='complete', training=False, games=len(rows),
            completed_at=datetime.now(timezone.utc).isoformat(), focal='diverse', opponent='control',
            summary=summarize(rows, HARTFORD_DECKS), rows_sha256=evaluation.value_sha(rows),
            manifest_sha256=evaluation.sha(root / 'manifest.json'),
            recovery=launch['recovery'], no_games_replayed=True)
        atomic_json(root / 'summary.json', result)
        atomic_json(root / 'progress.json', dict(status='complete', phase='complete',
            completed=20000, total=20000, finished_at=result['completed_at'],
            recovered_original_adjudications=8, new_games_played=0))
        atomic_json(Path(launch['recovery']['directory']) / 'receipt.json', dict(passed=True,
            preserved_existing_outcomes=19992, recovered_original_adjudications=8,
            recovered_ids=launch['recovery']['recovered_ids'], games_replayed=0,
            failure_evidence_retained=True, result_sha256=evaluation.sha(root / 'summary.json')))
    print(json.dumps(dict(passed=True, outcomes=20000, recovered=8, new_games=0)))


def audit(launch):
    from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS
    from patchwork_automaton.rl.hartford_evaluation import summarize
    evaluation = module(launch)
    root = Path(launch['evaluation']['output'])
    guard = evaluation.read(Path(launch['guard']) / 'guard-status.json')
    if guard['guard_exit_code'] != 0 or guard['child_exit_code'] != 0 or guard['status'] != 'exited':
        raise ValueError('Publication repair has no healthy final guard exit')
    if (guard['limits']['min_available_mib'] != 1536 or guard['limits']['memory_grace_seconds'] != 5
            or not guard['windows_awake']['release']['graceful_release']):
        raise ValueError('Publication repair guard differs')
    original_guard = evaluation.read(launch['recovery']['original_guard'])
    if original_guard['child_exit_code'] != 1:
        raise ValueError('Original failed guard evidence differs')
    receipt = evaluation.read(Path(launch['recovery']['directory']) / 'receipt.json')
    if receipt.get('passed') is not True or receipt['games_replayed'] != 0:
        raise ValueError('Publication recovery receipt is missing')
    with (root / 'run.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        rows = validated_rows(launch)
        result = evaluation.read(root / 'summary.json')
        if (result['rows_sha256'] != evaluation.value_sha(rows)
                or result['summary'] != summarize(rows, HARTFORD_DECKS)
                or receipt['result_sha256'] != evaluation.sha(root / 'summary.json')):
            raise ValueError('Recovered summary differs from original outcomes')
    samples = [json.loads(line) for line in
        (Path(launch['recovery']['original_guard']).parent / 'host-memory-samples.jsonl').read_text().splitlines()]
    if not samples or max(row['low_memory_seconds'] for row in samples) >= 5:
        raise ValueError('Missing resource history or memory breach')
    payload = gzip.compress(evaluation.canonical(rows), mtime=0)
    evidence = dict(passed=True, games=20000, audited_at=datetime.now(timezone.utc).isoformat(),
        scheduled_identities_verified=True, deck_seat_start_balance_verified=True,
        model_source_parent_hashes_verified=True, original_diagnostics_verified=True,
        publication_recovered_outcomes=8, games_replayed=0, failure_logs_preserved=True,
        original_guard_exit_code=1, publication_guard_exit_code=0,
        minimum_available_gib=min(row['available_memory_bytes'] for row in samples) / 1024**3,
        games_gzip_sha256=hashlib.sha256(payload).hexdigest())
    return dict(audit=evidence, result=result, manifest=evaluation.read(root / 'manifest.json'),
                games_gzip_b64=base64.b64encode(payload).decode())


if __name__ == '__main__':
    publish(json.loads(Path(sys.argv[1]).read_text()))
