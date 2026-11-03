"""Pinned, resumable 10,000-game diverse-versus-control endpoint comparison.

Run on Raytracer under the production resource guard with --launch launch.json.
No training, baseline panels, automatic retries, or replacement of valid games.
"""
import argparse
import asyncio
import base64
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import random
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def value_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def protocol(launch):
    raw = launch['evaluation']
    fixed = dict(seed=432611030000, games=10000, workers=48, runners=2,
                 inference_processes=2, opponent_inference_processes=2,
                 game_timeout=20, max_turns=50, max_decisions=6000)
    for name, expected in fixed.items():
        if raw.get(name) != expected:
            raise ValueError(f'Unexpected evaluation {name}: {raw.get(name)!r}')
    return dict(fixed, focal='diverse', opponent='control', local_checkpoint=50,
                lineage_checkpoint=150, policy='sampled temperature 1', training=False,
                deck_catalog='hartford', games_per_ordered_matchup=100,
                input_format='sparse-identity-standard-v2')


def schedule(config, decks, opponent_sha256):
    decks = tuple(decks)
    if len(decks) != 10 or len(set(decks)) != 10:
        raise ValueError('The endpoint requires ten distinct Hartford decks')
    rows = []
    for cell, (deck, other) in enumerate(product(decks, repeat=2)):
        for repeat in range(100):
            seat = repeat % 2
            seating = [other, other]
            seating[seat] = deck
            index = cell * 100 + repeat
            rows.append(dict(index=index, seed=config['seed'] + index,
                pairing=sorted((deck, other), key=decks.index), decks=seating,
                learner_seat=seat, learner_deck=deck, opponent_deck=other,
                starting_player=f'P{(repeat // 2) % 2}', partition='evaluation',
                deck_catalog='hartford', behavior_version=150, opponent_version=150,
                opponent_kind='frozen_neural', opponent_checkpoint_sha256=opponent_sha256))
    random.Random('hartford-identity-opponents-ab-v1').shuffle(rows)
    counts = Counter((r['learner_deck'], r['opponent_deck'], r['learner_seat'],
                      r['starting_player']) for r in rows)
    if len(rows) != 10000 or len(counts) != 400 or set(counts.values()) != {25}:
        raise AssertionError('Unbalanced endpoint schedule')
    return rows


def verify_files(files):
    for name, digest in files.items():
        if not Path(name).is_file() or sha(name) != digest:
            raise ValueError('Pinned artifact changed: ' + name)


def diagnostic_hashes(root, row):
    if not row.get('adjudicated'):
        return {}
    directory = Path(root).resolve() / 'diagnostics' / f"game-{row['index']:06d}"
    if (not row.get('diagnostics') or Path(row['diagnostics']).resolve() != directory
            or not (directory / 'adjudication.json').is_file()):
        raise ValueError('Adjudicated game is missing its original diagnostics')
    return {str(path.relative_to(directory)): sha(path) for path in sorted(directory.rglob('*'))
            if path.is_file()}


def validate_training_audit(value, launch, models):
    if (value.get('passed') is not True or value.get('games') != 100000
            or set(value.get('arms', {})) != {'control', 'diverse'}):
        raise ValueError('Both completed training arms require one complete passing audit')
    for name in ('control', 'diverse'):
        arm = value['arms'][name]
        if (arm.get('passed') is not True or arm.get('games') != 50000
                or arm.get('checkpoints') != 51
                or arm.get('partitions') != {'training': 40000, 'validation': 10000}
                or arm.get('evaluation_games') != 0
                or arm.get('output') != launch['arms'][name]['output']
                or arm.get('final_policy_sha256') != models[name]['sha256']):
            raise ValueError('Training audit does not match this completed branch: ' + name)


def input_manifest(launch):
    config = protocol(launch)
    project = Path(launch['project'])
    files = {str(project / name): digest for name, digest in launch['source_hashes'].items()}
    preserved = launch.get('preserved_files')
    if not isinstance(preserved, dict) or not preserved:
        raise ValueError('Original parent artifact hashes must be supplied in preserved_files')
    files.update(preserved)
    engine = launch['engine']
    engine_files = launch.get('engine_files', {})
    required_engine_files = {engine['library'], engine['build_manifest'],
                             launch['environment']['RUSTMAGE_LIBRARY']}
    if (not required_engine_files <= engine_files.keys()
            or engine_files[engine['library']] != engine['library_sha256']):
        raise ValueError('Explicit validated native library and build-manifest hashes are required')
    files.update(engine_files)
    files.setdefault(str(Path(__file__).resolve()), sha(__file__))
    models = {}
    for arm in ('control', 'diverse'):
        root = Path(launch['arms'][arm]['output'])
        report = read(root / 'report.json')
        if (report.get('status') != 'complete' or report.get('completed_games') != 50000
                or len(report['rounds']) != 50 or report.get('evaluation_games', 0) != 0
                or report.get('evaluations') or not report['config'].get('skip_evaluations')):
            raise ValueError('Training arm is not complete and training-only: ' + arm)
        if set(report['checkpoints']) != set(map(str, range(51))):
            raise ValueError('Training checkpoint history is incomplete: ' + arm)
        for number, pins in report['checkpoints'].items():
            for name, digest in pins.items():
                files[str(root / 'checkpoints' / f'round-{int(number):04d}' / name)] = digest
        for name in ('report.json', 'manifest.json', 'progress.json', 'checkpoints/latest.json'):
            files[str(root / name)] = sha(root / name)
        final = root / 'checkpoints/round-0050'
        models[arm] = dict(path=str(final / 'policy.keras'),
                          sha256=report['checkpoints']['50']['policy.keras'],
                          manifest_sha256=report['checkpoints']['50']['policy.manifest.json'])
    training_audit = Path(launch.get('training_audit',
        str(Path(launch['arms']['control']['output']).parent / 'training-audit.json')))
    validate_training_audit(read(training_audit), launch, models)
    files[str(training_audit)] = sha(training_audit)
    verify_files(files)
    return dict(schema=1, config=config, models=models, files=files,
                engine=engine, training_audit=str(training_audit),
                training=False, comparison='diverse versus control after 50K additional games each')


def load_results(root, planned):
    """Reject changed, duplicate, foreign or failed outcomes before resuming."""
    from patchwork_automaton.rl.hartford_evaluation import validate_row
    root = Path(root)
    failures = sorted((root / 'failures').glob('*.json'))
    if failures:
        raise ValueError('Unresolved evaluation failures retained: ' + ', '.join(map(str, failures)))
    expected = {r['index']: r for r in planned}
    if len(expected) != len(planned):
        raise ValueError('Duplicate scheduled identity')
    rows = {}
    for path in sorted((root / 'games').glob('*.json')):
        envelope = read(path)
        if set(envelope) != {'row', 'sha256'} or value_sha(envelope['row']) != envelope['sha256']:
            raise ValueError('Durable outcome checksum mismatch: ' + str(path))
        row = envelope['row']
        index = row.get('index')
        if (type(index) is not int or index not in expected or index in rows
                or path.name != f'game-{index}.json'):
            raise ValueError('Foreign or duplicate durable evaluation outcome: ' + str(path))
        validate_row(row, expected[index], checkpoint_round=150)
        if row.get('adjudicated') and row.get('diagnostic_hashes') != diagnostic_hashes(root, row):
            raise ValueError('Adjudication diagnostics changed: ' + str(path))
        rows[index] = row
    return rows


async def settle(awaitables):
    results = await asyncio.gather(*awaitables, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


async def run(launch):
    from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS
    from patchwork_automaton.neural.devices import configure_runtime, device_scope
    from patchwork_automaton.neural.standard_policy import StandardPolicy
    from patchwork_automaton.neural.inference_processes import weights_digest
    from patchwork_automaton.rl.hartford_evaluation import atomic_json, validate_row, summarize
    from patchwork_automaton.rl.standard_selfplay_processes import StandardSelfplayCollector

    manifest = input_manifest(launch)
    config = manifest['config']
    planned = schedule(config, HARTFORD_DECKS, manifest['models']['control']['sha256'])
    manifest['schedule_sha256'] = value_sha(planned)
    root = Path(launch['evaluation']['output'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'manifest.json').exists():
            if read(root / 'manifest.json') != manifest or read(root / 'schedule.json') != planned:
                raise ValueError('Evaluation configuration, source, or schedule changed on resume')
        else:
            if any(p.name != 'run.lock' for p in root.iterdir()):
                raise ValueError('Nonempty evaluation output has no pinned manifest')
            atomic_json(root / 'schedule.json', planned)
            atomic_json(root / 'manifest.json', manifest)
        rows = load_results(root, planned)
        expected = {r['index']: r for r in planned}
        remaining = [r for r in planned if r['index'] not in rows]
        if not remaining:
            verify_files(manifest['files'])
            summary = summarize([rows[i] for i in sorted(rows)], HARTFORD_DECKS)
            if (root / 'summary.json').exists():
                if read(root / 'summary.json')['summary'] != summary:
                    raise ValueError('Completed evaluation summary changed')
                print(json.dumps(dict(status='already_complete', games=len(rows))), flush=True)
                return
        started = time.monotonic()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        attempt = dict(started_at=datetime.now(timezone.utc).isoformat(), reused_games=len(rows),
                       missing_games=len(remaining), status='running')
        atomic_json(root / 'attempts' / (stamp + '.json'), attempt)
        collectors = []
        original = None
        focal = opponent = None
        stats = []

        def persist(row):
            row = dict(row, actions=None)
            index = row.get('index')
            target = root / 'games' / f'game-{index}.json'
            if index in rows or index not in expected or target.exists():
                atomic_json(root / 'failures' / f'duplicate-{stamp}-{index}.json', row)
                raise ValueError('Duplicate or unexpected evaluation completion')
            try:
                validate_row(row, expected[index], checkpoint_round=150)
                if row.get('adjudicated'):
                    row['diagnostic_hashes'] = diagnostic_hashes(root, row)
            except BaseException:
                atomic_json(root / 'failures' / f'game-{index}.json', row)
                raise
            atomic_json(target, dict(row=row, sha256=value_sha(row)))
            rows[index] = row
            if len(rows) % 100 == 0 or len(rows) == config['games']:
                progress = dict(status='running', phase='head-to-head', completed=len(rows),
                                total=config['games'], elapsed_seconds=time.monotonic() - started)
                atomic_json(root / 'progress.json', progress)
                print(json.dumps(progress), flush=True)

        try:
            if remaining:
                configure_runtime('cpu', hide_gpu=True)
                loop = asyncio.get_running_loop()
                def tick():
                    if not loop.is_closed():
                        loop.call_later(.002, tick)
                loop.call_later(.002, tick)
                with device_scope('cpu'):
                    focal = StandardPolicy.load(manifest['models']['diverse']['path']).model
                    opponent = StandardPolicy.load(manifest['models']['control']['path']).model
                if any(m.input_format != config['input_format'] for m in (focal, opponent)):
                    raise ValueError('Both endpoint models must retain the identity architecture')
                original = [weights_digest(m.get_weights()) for m in (focal, opponent)]
                collectors = [StandardSelfplayCollector(focal, opponent, workers=24,
                              inference_processes=1, opponent_inference_processes=1) for _ in range(2)]
                await settle(c.start(version=150, opponent_version=150) for c in collectors)
                if any([c.policy_digest, c.opponent_digest] != original for c in collectors):
                    raise ValueError('Inference replicas differ from the endpoint checkpoints')
                jobs = [dict(index=r['index'], seed=r['seed'], game=r) for r in remaining]
                await settle(c.run_schedule(jobs[i::2], progress=persist, game_runner='standard_ppo',
                    version=150, opponent_version=150,
                    opponent_sha256=manifest['models']['control']['sha256'], output=None,
                    diagnostics=root / 'diagnostics', greedy=False, game_timeout=20,
                    max_turns=50, max_decisions=6000) for i, c in enumerate(collectors))
                stats = [c.stats() for c in collectors]
                if [weights_digest(m.get_weights()) for m in (focal, opponent)] != original:
                    raise ValueError('Evaluation mutated the endpoint model weights')
            if set(rows) != set(expected):
                raise ValueError('Missing evaluation outcomes')
            verify_files(manifest['files'])
            ordered = [rows[i] for i in sorted(rows)]
            result = dict(status='complete', training=False, games=len(rows),
                completed_at=datetime.now(timezone.utc).isoformat(), focal='diverse', opponent='control',
                summary=summarize(ordered, HARTFORD_DECKS), rows_sha256=value_sha(ordered),
                manifest_sha256=sha(root / 'manifest.json'), stats=stats,
                all_source_models_and_training_artifacts_unchanged=True)
            atomic_json(root / 'summary.json', result)
            atomic_json(root / 'progress.json', dict(status='complete', phase='complete',
                        completed=len(rows), total=config['games'], finished_at=result['completed_at']))
            attempt.update(status='complete', new_games=len(rows) - attempt['reused_games'])
        except BaseException as error:
            attempt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            atomic_json(root / 'failures' / ('attempt-' + stamp + '.json'), attempt)
            raise
        finally:
            try:
                await settle(c.close() for c in collectors)
            finally:
                attempt.update(finished_at=datetime.now(timezone.utc).isoformat(),
                               elapsed_seconds=time.monotonic() - started)
                atomic_json(root / 'attempts' / (stamp + '.json'), attempt)


def audit(launch):
    """Read-only final audit, after both training audits and a healthy guard exit."""
    from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS
    from patchwork_automaton.rl.hartford_evaluation import summarize
    manifest = input_manifest(launch)
    root = Path(launch['evaluation']['output'])
    planned = schedule(manifest['config'], HARTFORD_DECKS, manifest['models']['control']['sha256'])
    manifest['schedule_sha256'] = value_sha(planned)
    if read(root / 'manifest.json') != manifest or read(root / 'schedule.json') != planned:
        raise ValueError('Final evaluation manifest or schedule differs from its pins')
    guard = read(Path(launch['evaluation']['guard']) / 'guard-status.json')
    if (guard.get('status') != 'exited' or guard.get('guard_exit_code') != 0
            or guard.get('child_exit_code') != 0 or guard.get('exit_code') != 0):
        raise ValueError('Final evaluation guard has not exited successfully')
    if (guard['limits']['min_available_mib'] != 1536
            or guard['limits']['memory_grace_seconds'] != 5):
        raise ValueError('Final evaluation resource guard configuration changed')
    awake = guard.get('windows_awake', {})
    if awake.get('requested') and not awake.get('release', {}).get('graceful_release'):
        raise ValueError('Windows wakefulness helper did not release cleanly')
    pid = guard.get('child_pid')
    if pid and Path('/proc', str(pid), 'cmdline').exists():
        args = Path('/proc', str(pid), 'cmdline').read_bytes()
        if str(Path(__file__).resolve()).encode() in args:
            raise ValueError('Evaluation child is still active')
    with (root / 'run.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        rows_by_id = load_results(root, planned)
        if set(rows_by_id) != {r['index'] for r in planned}:
            raise ValueError('Endpoint does not contain exactly all 10,000 scheduled outcomes')
        rows = [rows_by_id[i] for i in sorted(rows_by_id)]
        result = read(root / 'summary.json')
        recomputed = summarize(rows, HARTFORD_DECKS)
        if (result.get('status') != 'complete' or result.get('training') is not False
                or result.get('games') != 10000 or result['summary'] != recomputed
                or result['rows_sha256'] != value_sha(rows)
                or result['manifest_sha256'] != sha(root / 'manifest.json')):
            raise ValueError('Endpoint summary or outcome digest differs from the complete results')
        verify_files(manifest['files'])
    samples = [json.loads(line) for line in
               (Path(launch['evaluation']['guard']) / 'host-memory-samples.jsonl').read_text().splitlines()]
    if not samples or max(r['low_memory_seconds'] for r in samples) >= 5:
        raise ValueError('Missing resource history or memory guard breach')
    payload = gzip.compress(canonical(rows), mtime=0)
    evidence = dict(passed=True, audited_at=datetime.now(timezone.utc).isoformat(), games=10000,
        scheduled_identities_verified=True, deck_seat_start_balance_verified=True,
        failed_games=0, training=False, guard_exit_code=0,
        adjudication_diagnostic_hashes_verified=True,
        model_source_parent_hashes_verified=True, provenance_files_verified=len(manifest['files']),
        both_training_completion_audits_passed=True, summary_recomputed=True,
        games_gzip_sha256=hashlib.sha256(payload).hexdigest(),
        minimum_available_gib=min(r['available_memory_bytes'] for r in samples) / 1024 ** 3,
        limitation='One training run per recipe; endpoint head-to-head does not measure absolute strength or performance against unseen opponents.')
    return dict(audit=evidence, result=result, manifest=manifest,
                games_gzip_b64=base64.b64encode(payload).decode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    launch = read(args.launch)
    if args.audit_only:
        print(json.dumps(audit(launch), allow_nan=False))
    else:
        asyncio.run(run(launch))


if __name__ == '__main__':
    main()
