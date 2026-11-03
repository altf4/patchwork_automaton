"""Run two isolated 50K continuations, audit them, then their 10K head-to-head."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from audit_training import read, verify_static


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--launch', type=Path, required=True)
    args = parser.parse_args(); launch = read(args.launch); base = args.launch.parent
    verify_static(launch)
    for name in ('control', 'diverse'):
        write(base / 'progress.json', dict(phase=name, status='running', at=datetime.now(timezone.utc).isoformat()))
        arm = launch['arms'][name]; report = read(Path(arm['output']) / 'report.json')
        if report['status'] != 'prepared':
            raise ValueError('Inspect existing arm before any controlled resume: ' + name)
        subprocess.run(arm['command'], cwd=launch['project'], check=True)
        verify_static(launch)
        audit = subprocess.run([sys.executable, str(Path(__file__).with_name('audit_training.py')),
            '--launch', str(args.launch), '--arm', name], capture_output=True, text=True, check=True)
        write(base / (name + '-audit.json'), json.loads(audit.stdout))
    audit = subprocess.run([sys.executable, str(Path(__file__).with_name('audit_training.py')),
        '--launch', str(args.launch)], capture_output=True, text=True, check=True)
    write(Path(launch['training_audit']), json.loads(audit.stdout))
    write(base / 'progress.json', dict(phase='head_to_head', status='running', at=datetime.now(timezone.utc).isoformat()))
    subprocess.run([sys.executable, launch['evaluation']['script'], '--launch', str(args.launch)], check=True)
    verify_static(launch)
    write(base / 'progress.json', dict(phase='complete', status='complete',
        collection_games=100000, training_games=80000, holdout_games=20000,
        evaluation_games=10000, at=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
