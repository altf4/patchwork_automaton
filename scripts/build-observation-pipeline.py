"""Build the CPU observation encoder using the project's bounded Rust runner."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
engine = Path(os.environ.get('RUSTMAGE_ROOT', root / '.local/rustmage')).resolve()
target = root / '.local/observation-target'
subprocess.run([sys.executable, str(engine / 'scripts/resource_guard.py'), '--',
                'cargo', 'build', '--release', '--locked', '--manifest-path',
                str(root / 'native/observation-pipeline/Cargo.toml')], cwd=root,
               env=dict(os.environ, CARGO_TARGET_DIR=str(target)), check=True)
filename = {'linux': 'libpatchwork_observation_pipeline.so',
            'darwin': 'libpatchwork_observation_pipeline.dylib',
            'win32': 'patchwork_observation_pipeline.dll'}[sys.platform]
library = target / 'release' / filename
print(json.dumps(dict(library=str(library), sha256=hashlib.sha256(library.read_bytes()).hexdigest())))
