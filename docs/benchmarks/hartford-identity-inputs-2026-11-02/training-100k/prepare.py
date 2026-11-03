"""Stage only: no training, games, model initialization, tmux or timers."""
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
BASE='/home/altf4/benchmarks/hartford-identity-ppo-20261102'
PARENT='/home/altf4/benchmarks/hartford-compact-ppo-20261101'
EXPECTED='eb56f40c7e3519a9cbb69dbbc9d85129bec91a6bf600d13ce3d599c5ee64fef6'

REMOTE=r'''
import hashlib,io,json,os,shlex,shutil,subprocess,sys,tarfile
from pathlib import Path
base=Path(BASE);parent=Path(PARENT);base.mkdir(exist_ok=False)
project=base/'project-v1';project.mkdir()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as archive:
 archive.extractall(project,filter='data')
old=parent/'project-noeval-v1'
# Reuse the exact source/binary pair of the validated bridge.
shutil.rmtree(project/'native/direct-standard')
shutil.copytree(old/'native/direct-standard',project/'native/direct-standard')
shutil.copy2(old/'src/patchwork_automaton/data/runtime-engine.json',project/'src/patchwork_automaton/data/runtime-engine.json')
(project/'.local').mkdir()
(project/'.local/direct-standard').symlink_to(parent/'project-loop-v2/.local/direct-standard',target_is_directory=True)
inputs=base/'inputs/source-200k';inputs.mkdir(parents=True)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
preserved={}
for name in ('policy.keras','policy.manifest.json','state.json'):
 original=parent/'run-v3/checkpoints/round-0200'/name
 before=sha(original);shutil.copy2(original,inputs/name)
 assert sha(inputs/name)==sha(original)==before
 preserved[name]=before
assert preserved['policy.keras']==EXPECTED
state=json.loads((inputs/'state.json').read_text())
assert state['round']==200 and state['config']['games_per_round']==1000
control=base/'run-launch-v1';control.mkdir()
output=base/'run-v1';guard=base/'run-guard-v1'
env=dict(ENV)
env.update(PYTHONPATH=str(project/'src')+':'+str(project/'tests')+':'+str(project),
 PATCHWORK_ARENA_POLL_INFERENCE='1',PATCHWORK_STANDARD_PLAYBACK='direct')
python=PYTHON
command=[python,'-u','-m','examples.train_hartford_ppo','--architecture','compact-identity',
 '--initialize-from',str(inputs/'policy.keras'),'--initialize-from-sha256',EXPECTED,
 '--output',str(output),'--rounds','100','--seed','412610510000','--eval-seed','912610510000',
 '--eval-every-games','10000','--skip-evaluations','--workers','48','--runners','2',
 '--inference-processes','2','--opponent-inference-processes','2','--compact-input-mib','512',
 '--game-timeout','20','--update-timeout','1800']
write(control/'command.json',command)
supervised=[python,'-u',str(project/'scripts/guard-hartford-run.py'),'--root',str(guard),
 '--command',str(control/'command.json'),'--cwd',str(project),'--min-available-mib','1536',
 '--memory-grace-seconds','5','--keep-windows-awake','--powershell',
 '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe']
launch=dict(status='prepared_not_started',project=str(project),output=str(output),guard=str(guard),
 control=str(control),launcher=str(control/'launch.sh'),environment=env,command_argv=command,
 supervisor_argv=supervised,parent_checkpoint_hashes=preserved,parent_collection_games=200000,
 additional_collection_games=100000,rounds=100,training_games=80000,holdout_games=20000,
 milestone_evaluations=False,lookahead_enabled=False,rule_descriptors_enabled=False,
 optimizer_initialization='fresh Adam actor and critic; fresh trainer RNG',
 startup_opponents='Transferred checkpoint zero first; build new-architecture predecessor window normally',
 original_output=str(parent/'run-v3'))
runenv=dict(os.environ,**env)
metadata=r"""
import json
from collections import Counter
from pathlib import Path
from examples.train_hartford_ppo import parser,configuration
from patchwork_automaton.rl.hartford_schedule import training_schedule,training_seed_bounds
from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS
from patchwork_automaton.rl.hartford_preflight import hartford_preflight
command=json.loads(Path(CONTROL+'/command.json').read_text())
config=configuration(parser().parse_args(command[4:]))
readiness=hartford_preflight(playback='direct')
if not readiness['ready']:raise ValueError(readiness['errors'])
assert config['skip_evaluations'] and config['use_identity'] and not config['use_rules']
totals=Counter();seeds=set()
for number in range(1,101):
 rows=training_schedule(tuple(HARTFORD_DECKS),round_id=number,seed=config['seed'])
 assert len(rows)==1000
 assert Counter(r['partition'] for r in rows)=={'training':800,'validation':200}
 if number>=4:assert sorted(Counter(r['opponent_version'] for r in rows).values())==[100,100,100,700]
 for row in rows:
  if row['seed'] in seeds:raise ValueError('Duplicate seed')
  seeds.add(row['seed']);totals[row['partition']]+=1
assert len(seeds)==100000 and min(seeds)>412610509999
Path(CONTROL+'/configuration.json').write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
Path(CONTROL+'/preflight.json').write_text(json.dumps(readiness,indent=2,sort_keys=True)+'\n')
Path(CONTROL+'/schedule-audit.json').write_text(json.dumps(dict(passed=True,games=len(seeds),
 partitions=dict(totals),seed_bounds=training_seed_bounds(rounds=100,seed=config['seed']),
 startup_rounds=3,steady_opponent_quotas=[700,100,100,100]),indent=2)+'\n')
"""
metadata='CONTROL='+repr(str(control))+'\n'+metadata
# Metadata-only: no TensorFlow, models, games or optimizer updates.
subprocess.run([python,'-c',metadata],cwd=project,env=runenv,check=True)
launch['source_hashes']={str(p.relative_to(project)):sha(p) for p in sorted(project.rglob('*'))
 if p.is_file() and not p.is_symlink() and '.local' not in p.relative_to(project).parts}
launch['engine']=json.loads((control/'preflight.json').read_text())['engine']
verify=r"""
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
"""
(control/'verify_prepared.py').write_text(verify)
write(control/'launch.json',launch)
script=['#!/bin/bash','set -euo pipefail','source '+shlex.quote(ENV_SOURCE)]
script+=['export '+key+'='+shlex.quote(value) for key,value in env.items()]
script+=['cd '+shlex.quote(str(project)),shlex.join([python,str(control/'verify_prepared.py')]),
 'exec '+shlex.join(supervised)]
(control/'launch.sh').write_text('\n'.join(script)+'\n')
(control/'launch.sh').chmod(0o755)
subprocess.run(['/bin/bash','-n',str(control/'launch.sh')],check=True)
subprocess.run([python,str(control/'verify_prepared.py')],cwd=project,env=runenv,check=True)
assert not output.exists() and not guard.exists()
for folder in (project,base/'inputs'):
 for current,dirs,files in os.walk(folder,followlinks=False):
  for name in files:
   path=Path(current)/name
   if not path.is_symlink():path.chmod(0o444)
  Path(current).chmod(0o555)
assert all(sha(parent/'run-v3/checkpoints/round-0200'/name)==digest for name,digest in preserved.items())
print(json.dumps(dict(status='prepared_not_started',root=str(base),output_exists=output.exists(),
 guard_exists=guard.exists(),source_files=len(launch['source_hashes']),source_200k_unchanged=True)))
'''


def main():
    old=json.loads((ROOT/'docs/benchmarks/hartford-compact-ppo-2026-11-01/launch.json').read_text())
    remote=old['remote']
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for folder in ('src','examples','scripts','tests','native/direct-standard'):
            for path in sorted((ROOT/folder).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'):
                    archive.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
        archive.add(ROOT/'pyproject.toml',arcname='pyproject.toml')
    header='\n'.join(name+'='+repr(value) for name,value in dict(BASE=BASE,PARENT=PARENT,EXPECTED=EXPECTED,
        ENV=remote['environment_overrides'],PYTHON=remote['venv_python'],ENV_SOURCE=remote['environment_source']).items())+'\n'
    result=subprocess.run(remote['ssh_argv_prefix']+['python3 -c '+shlex.quote(header+REMOTE)],
        input=buffer.getvalue(),capture_output=True,check=False,timeout=180)
    (HERE/'prepare-stdout.log').write_bytes(result.stdout);(HERE/'prepare-stderr.log').write_bytes(result.stderr)
    if result.returncode:raise RuntimeError(result.stderr.decode())
    print(result.stdout.decode())
    for name in ('launch.json','configuration.json','preflight.json','schedule-audit.json','command.json','launch.sh','verify_prepared.py'):
        data=subprocess.run(remote['ssh_argv_prefix']+['cat '+shlex.quote(BASE+'/run-launch-v1/'+name)],
            capture_output=True,check=True,timeout=40).stdout
        (HERE/name).write_bytes(data)
    launch=json.loads((HERE/'launch.json').read_text())
    launch['remote']=dict(remote,working_directory=launch['project'],environment_overrides=launch['environment'])
    (HERE/'launch.json').write_text(json.dumps(launch,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
