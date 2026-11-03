
import hashlib,json
from pathlib import Path
base=Path(__file__).resolve().parent.parent
launch=json.loads((base/'run-launch-v1/launch.json').read_text())
project=Path(launch['project'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in launch['source_hashes'].items():
 if sha(project/name)!=digest:raise ValueError('Frozen source changed: '+name)
for name,digest in launch['parent_checkpoint_hashes'].items():
 if sha(base/'inputs/source-200k'/name)!=digest:raise ValueError('Transferred input changed: '+name)
 if sha(Path(launch['original_output'])/'checkpoints/round-0200'/name)!=digest:raise ValueError('Original checkpoint changed: '+name)
engine=launch['engine']
if sha(Path(engine['library']))!=engine['library_sha256']:raise ValueError('Engine binary changed')
from patchwork_automaton.rl.hartford_preflight import hartford_preflight
if not hartford_preflight(playback='direct')['ready']:raise ValueError('Runtime preflight failed')
print('Frozen source, parent checkpoint and runtime verified; no experiment started.')
