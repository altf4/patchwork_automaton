"""Stream an incremental, recoverable study backup; stdout is tar.gz only.

Published checkpoints are immutable. Mutable JSON is captured and parsed before
any archive header is emitted. The receiver must verify backup-manifest.json,
JSON and ZIP CRCs before atomically accepting the archive. Raw rollout data is
intentionally excluded; its original remote copy is not modified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

CHECKPOINT_FILES = ('policy.keras', 'policy.manifest.json', 'optimizer.npz',
                    'critic-optimizer.npz', 'state.json')
MAX_JSON_BYTES = 128 * 1024**2
MAX_FILE_BYTES = 512 * 1024**2
MAX_TOTAL_BYTES = 1024**3


def _json(payload):
    return json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError('Nonfinite JSON in study backup')))


def _safe_name(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or path.as_posix() != name or '\\' in name
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError('Unsafe backup member path')
    return name


def _read(path, maximum):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Backup source must be a regular file: '+str(path))
    with path.open('rb') as stream:
        payload = stream.read(maximum+1)
    if len(payload) > maximum:
        raise ValueError('Backup member exceeds byte limit: '+str(path))
    return payload


def _round(name):
    match = re.fullmatch(r'round-(\d{4,})', name)
    if match is None:
        raise ValueError('Invalid published checkpoint name')
    return int(match.group(1))


def _capture_once(root, after_round, include_logs, max_log_bytes):
    root = Path(root).resolve()
    if type(after_round) is not int or after_round < -1:
        raise ValueError('after-round must be -1 or a completed round number')
    latest_path = root/'training/checkpoints/latest.json'
    latest_bytes = _read(latest_path, MAX_JSON_BYTES)
    latest = _json(latest_bytes)
    name = latest['checkpoint']
    latest_round = _round(name)
    if after_round > latest_round:
        raise ValueError('after-round is ahead of the captured latest checkpoint')
    report_bytes = _read(root/'training/report.json', MAX_JSON_BYTES)
    report = _json(report_bytes)
    rounds = report.get('rounds', [])
    ids = [entry['round'] for entry in rounds]
    if ids != list(range(1, len(ids)+1)):
        raise ValueError('Training report has missing or duplicate rounds')
    report_round = ids[-1] if ids else 0
    if report_round not in (latest_round, latest_round-1):
        raise ValueError('Training report is inconsistent with the captured checkpoint')
    members = {'training/checkpoints/latest.json':latest_bytes, 'training/report.json':report_bytes}
    optional = ('study.json', 'initialization.json', 'cleanup.json', 'manifest.json')
    missing = []
    for filename in optional:
        path = root/filename
        if path.exists():
            payload = _read(path, MAX_JSON_BYTES)
            _json(payload)
            members[filename] = payload
        else:
            missing.append(filename)
    # Only named top-level JSON outputs; never recurse into exchange/rollouts.
    for pattern in ('monitor*.json', 'confirmation*.json', 'stage*.process.json',
                    'stage*.cleanup.json', 'train-through-*.process.json',
                    'train-through-*.cleanup.json'):
        for path in sorted(root.glob(pattern)):
            payload = _read(path, MAX_JSON_BYTES)
            _json(payload)
            members[path.name] = payload
    # Capture latest state even on a metadata-only incremental backup, to prove
    # the report can be recovered by the existing resume implementation.
    latest_state_bytes = _read(root/'training/checkpoints'/name/'state.json', MAX_JSON_BYTES)
    state = _json(latest_state_bytes)
    if state.get('round') != latest_round or state.get('config') != report.get('config'):
        raise ValueError('Latest checkpoint state/config differs from report')
    recoverable = report_round != latest_round
    if recoverable and state.get('round_result', {}).get('round') != latest_round:
        raise ValueError('Report lag has no checkpoint round_result recovery record')
    included = list(range(after_round+1, latest_round+1))
    for round_id in included:
        directory = root/'training/checkpoints'/f'round-{round_id:04d}'
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('Missing published checkpoint: '+str(directory))
        if {path.name for path in directory.iterdir()} != set(CHECKPOINT_FILES):
            raise ValueError('Unexpected/incomplete checkpoint artifact set')
        for filename in CHECKPOINT_FILES:
            relative = f'training/checkpoints/round-{round_id:04d}/{filename}'
            payload = (latest_state_bytes if round_id == latest_round and filename == 'state.json'
                       else _read(directory/filename, MAX_JSON_BYTES if filename.endswith('.json') else MAX_FILE_BYTES))
            if filename.endswith('.json'):
                data = _json(payload)
                if filename == 'state.json' and data.get('round') != round_id:
                    raise ValueError('Published checkpoint state has the wrong round')
            members[relative] = payload
    if include_logs:
        if not 0 < max_log_bytes <= 16*1024**2:
            raise ValueError('Log tail limit must be between 1 byte and 16 MiB')
        for path in sorted(root.glob('*.log')):
            if path.is_symlink() or not path.is_file():
                raise ValueError('Log source must be a regular file')
            with path.open('rb') as stream:
                size = stream.seek(0, 2)
                start = max(0, size-max_log_bytes)
                stream.seek(start)
                members['log-tails/'+path.name] = stream.read(max_log_bytes)
    # If the trainer crossed a checkpoint while we copied metadata, retry before
    # writing any output. An immutable captured checkpoint never changes later.
    if _read(latest_path, MAX_JSON_BYTES) != latest_bytes:
        raise RuntimeError('Latest checkpoint changed during backup capture')
    if sum(map(len, members.values())) > MAX_TOTAL_BYTES:
        raise ValueError('Backup exceeds total byte limit')
    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        study_root=str(root), after_round=after_round, latest_round=latest_round,
        latest_checkpoint=name, included_checkpoint_rounds=included,
        report_last_round=report_round, report_recoverable_from_checkpoint=recoverable,
        missing_optional_metadata=missing,
        raw_rollouts_included=False, raw_rollouts_preserved_at_source=True,
        log_tail_bytes=max_log_bytes if include_logs else 0,
        latest_state_sha256=hashlib.sha256(latest_state_bytes).hexdigest(),
        members=[dict(path=_safe_name(member), size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                 for member, payload in sorted(members.items())])
    members['backup-manifest.json'] = (json.dumps(manifest, sort_keys=True, indent=2,
                                                allow_nan=False)+'\n').encode()
    return members, manifest


def capture_study(study_root, *, after_round=-1, include_logs=False,
                  max_log_bytes=1024*1024, attempts=3):
    """Capture before emitting bytes, retrying only a concurrent round advance."""
    if type(attempts) is not int or attempts < 1:
        raise ValueError('Backup capture needs at least one attempt')
    for attempt in range(attempts):
        try:
            return _capture_once(study_root, after_round, include_logs, max_log_bytes)
        except RuntimeError:
            if attempt+1 == attempts:
                raise


def write_backup(members, output):
    """Write only bounded regular relative members; JSON was already parsed."""
    with tarfile.open(fileobj=output, mode='w|gz', format=tarfile.USTAR_FORMAT) as archive:
        for name, payload in sorted(members.items()):
            info = tarfile.TarInfo(_safe_name(name))
            info.size, info.mode, info.mtime = len(payload), 0o600, 0
            archive.addfile(info, io.BytesIO(payload))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--after-round', type=int, default=-1)
    parser.add_argument('--include-logs', action='store_true')
    parser.add_argument('--max-log-bytes', type=int, default=1024*1024)
    args = parser.parse_args(argv)
    members, _ = capture_study(args.study_root, after_round=args.after_round,
        include_logs=args.include_logs, max_log_bytes=args.max_log_bytes)
    write_backup(members, sys.stdout.buffer)


if __name__ == '__main__':
    main()
