"""Run the three separate evaluation panels sequentially under one guard."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from evaluate import verify
from patchwork_automaton.rl.hartford_evaluation import atomic_json

ROOT = Path(__file__).resolve().parent
config = json.loads((ROOT / 'config.json').read_text())
verify()
for panel in config['panels']:
    atomic_json(ROOT / 'progress.json', dict(phase=panel['name'], status='running', started_at=datetime.now(timezone.utc).isoformat()))
    subprocess.run([sys.executable, '-u', str(ROOT / 'evaluate.py'), panel['name']], check=True)
    verify()
atomic_json(ROOT / 'summary.json', dict(status='complete', games=11200, training=False,
    panels={p['name']: json.loads((ROOT / p['name'] / 'summary.json').read_text()) for p in config['panels']}))
atomic_json(ROOT / 'progress.json', dict(phase='complete', status='complete', finished_at=datetime.now(timezone.utc).isoformat(), games=11200))
