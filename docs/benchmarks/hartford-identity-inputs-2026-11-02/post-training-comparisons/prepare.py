"""Stage a new, immutable evaluation suite and launch its resource supervisor."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

BASE = Path(__file__).resolve().parent
REMOTE = r'''
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
s=json.loads(sys.stdin.read());l=s['launch'];root=Path(l['output']).parent/'post-identity-comparisons-v1'
control=root.with_name(root.name+'-launch');guard=root.with_name(root.name+'-guard')
assert not any(p.exists() for p in (root,control,guard)), 'Existing suite: inspect, never replay'
assert shutil.disk_usage(root.parent).free>32*1024**3
assert int(next(z.split()[1] for z in Path('/proc/meminfo').read_text().splitlines() if z.startswith('MemAvailable:')))>1536*1024
def read(p):return json.loads(p.read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
identityroot=Path(l['output']);parentroot=Path(l['original_output'])
assert read(Path(l['guard'])/'guard-status.json')['guard_exit_code']==0
assert read(identityroot/'report.json')['status']=='complete'
assert read(parentroot/'report.json')['status']=='complete'
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=p.read_bytes().split(b'\0')
  assert b'examples.train_hartford_ppo' not in args, 'Training still active'
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
files={str(Path(l['project'])/name):digest for name,digest in l['source_hashes'].items()}
files[l['engine']['library']]=l['engine']['library_sha256']
files[l['engine']['build_manifest']]=sha(l['engine']['build_manifest'])
files[l['environment']['RUSTMAGE_LIBRARY']]=sha(l['environment']['RUSTMAGE_LIBRARY'])
for folder in (identityroot,parentroot):
 report=read(folder/'report.json')
 for number,pins in report['checkpoints'].items():
  for name,digest in pins.items():files[str(folder/'checkpoints'/('round-'+number.zfill(4))/name)]=digest
 for name in ('report.json','manifest.json','progress.json','checkpoints/latest.json'):
  files[str(folder/name)]=sha(folder/name)
for name,digest in files.items():assert sha(name)==digest,name
root.mkdir();control.mkdir()
models={}
for name,folder,want in [('identity',identityroot/'checkpoints/round-0100',s['identity_sha256']),('parent',parentroot/'checkpoints/round-0200',s['parent_sha256'])]:
 dest=root/'models'/name;dest.mkdir(parents=True)
 for f in ('policy.keras','policy.manifest.json'):
  shutil.copy2(folder/f,dest/f);files[str(dest/f)]=sha(dest/f)
 assert sha(dest/'policy.keras')==want
 models[name]=dict(path=str(dest/'policy.keras'),sha256=want,
  local_checkpoint=100 if name=='identity' else 200,total_exposure_games=300000 if name=='identity' else 200000)
panels=[dict(name='identity-vs-parent',kind='neural',model='identity',version=300,games=4000,offset=0,game_timeout=20),
 dict(name='identity-vs-scripted',kind='scripted',model='identity',version=300,games=3600,offset=4000,game_timeout=60),
 dict(name='parent-vs-scripted',kind='scripted',model='parent',version=200,games=3600,offset=7600,game_timeout=60)]
config=dict(models=models,panels=panels,neural_seed=3126110300000,baseline_seed=802610300000,
 workers=48,inference_processes=2,opponent_inference_processes=2,
 max_turns=50,max_decisions=6000,policy='sampled temperature 1',engine=l['engine'],
 training=False,notes='Independent evaluation only; historical three-deck StandardAggro decklists. Baseline arms share 400 seed offsets repeated over nine cells; use paired clustered intervals.')
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
write(root/'config.json',config);files[str(root/'config.json')]=sha(root/'config.json')
for name,body in s['scripts'].items():
 (root/name).write_text(body);files[str(root/name)]=sha(root/name)
provenance=dict(files=files,identity_training_output=str(identityroot),parent_training_output=str(parentroot),
 frozen_source=l['project'],engine=l['engine'],original_artifacts_preserved=True)
write(root/'provenance.json',provenance)
env=dict(l['environment']);env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1')
command=[l['remote']['venv_python'],'-u',str(root/'run_suite.py')]
write(control/'command.json',command)
supervised=[l['remote']['venv_python'],'-u',str(Path(l['project'])/'scripts/guard-hartford-run.py'),
 '--root',str(guard),'--command',str(control/'command.json'),'--cwd',l['project'],
 '--min-available-mib','1536','--memory-grace-seconds','5','--keep-windows-awake',
 '--powershell','/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe']
import shlex
script=['#!/bin/bash','set -euo pipefail','source '+shlex.quote(l['remote']['environment_source'])]
script+=['export '+k+'='+shlex.quote(v) for k,v in env.items()]
script+=['cd '+shlex.quote(l['project']),'exec '+shlex.join(supervised)]
(control/'launch.sh').write_text('\n'.join(script)+'\n')
subprocess.run(['/bin/bash','-n',str(control/'launch.sh')],check=True)
for path in [root/'config.json',root/'provenance.json',*[root/n for n in s['scripts']],*list((root/'models').rglob('*'))]:
 if path.is_file():path.chmod(0o444)
record=dict(status='prepared',remote=l['remote'],project=l['project'],output=str(root),control=str(control),guard=str(guard),
 environment=env,engine=l['engine'],config=config,provenance_sha256=sha(root/'provenance.json'),
 script_hashes={name:sha(root/name) for name in s['scripts']},session='hartford-identity-final-comparisons',
 command=command,supervised=supervised,launcher=str(control/'launch.sh'),games=11200)
write(control/'launch.json',record)
subprocess.run(['tmux','new-session','-d','-s',record['session'],'/bin/bash '+shlex.quote(record['launcher'])],check=True)
record['status']='started';print(json.dumps(record))
'''


def main():
    launch = json.loads((BASE.parent / 'training-100k/launch.json').read_text())
    complete = json.loads((BASE.parent / 'training-100k/completion.json').read_text())
    assert complete['status'] == 'complete'
    payload = dict(launch=launch, identity_sha256=complete['final_policy_sha256'],
        parent_sha256='eb56f40c7e3519a9cbb69dbbc9d85129bec91a6bf600d13ce3d599c5ee64fef6',
        scripts={name: (BASE / name).read_text() for name in ('evaluate.py', 'run_suite.py')})
    for name, body in payload['scripts'].items():
        compile(body, name, 'exec')
    command = launch['remote']['ssh_argv_prefix'] + [shlex.join([launch['remote']['venv_python'], '-c', REMOTE])]
    result = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True, timeout=90)
    (BASE / 'prepare-stdout.log').write_text(result.stdout)
    (BASE / 'prepare-stderr.log').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr)
    record = json.loads(result.stdout)
    (BASE / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('status', 'output', 'guard', 'games')}))


if __name__ == '__main__':
    main()
