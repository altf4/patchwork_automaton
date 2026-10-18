"""Fetch and build the reviewed RustMage revision. Run with uv run python.

Builds always use RustMage's mandatory shared memory-limited resource guard.
Existing checkouts must match the pin and be clean; never reset another's work.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from patchwork_automaton.backends.rustmage import RUSTMAGE_REVISION, OBSERVATION_SCHEMA

URL = 'https://github.com/altf4/rustmage.git'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, default=Path('.local/rustmage'))
    args = p.parse_args(); root = args.directory.resolve()
    if not root.exists():
        subprocess.run(['git', 'clone', URL, str(root)], check=True)
        subprocess.run(['git', '-C', str(root), 'checkout', '--detach', RUSTMAGE_REVISION], check=True)
    def git(*parts): return subprocess.check_output(['git', '-C', str(root), *parts], text=True).strip()
    if git('rev-parse', 'HEAD') != RUSTMAGE_REVISION or git('status', '--porcelain'):
        raise RuntimeError('Use a clean checkout at ' + RUSTMAGE_REVISION + '; existing work was preserved')
    subprocess.run([sys.executable, 'scripts/resource_guard.py', '--', 'cargo', 'build', '--release', '--lib'], cwd=root, check=True)
    lib = root / 'target/release' / {'linux': 'librustmage.so', 'darwin': 'librustmage.dylib', 'win32': 'rustmage.dll'}[sys.platform]
    record = dict(repository=URL, revision=RUSTMAGE_REVISION, observation_schema=OBSERVATION_SCHEMA,
                  library=str(lib), sha256=hashlib.sha256(lib.read_bytes()).hexdigest())
    (root / 'target/patchwork-build.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__': main()
