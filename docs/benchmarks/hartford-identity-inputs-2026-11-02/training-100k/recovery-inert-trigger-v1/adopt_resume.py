"""Apply the recorded source-only runtime recovery under run.lock, then resume."""
import json,shlex,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent

REMOTE=r'''
import fcntl,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
s=json.loads(sys.argv[1]);b=json.loads(sys.argv[2]);root=Path(s['output']);project=Path(b['project'])
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  argv=p.read_bytes().split(b'\0')
  if b'examples.train_hartford_ppo' in argv and str(root).encode() in argv:raise RuntimeError('Owned learner active')
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
archive=root/'decision-limit-investigation-v1'
assert read(Path(s['guard'])/'guard-status.json')['child_exit_code']==1
assert (Path(b['control'])/'exit-code').read_text().strip()=='0'
retry=archive/'retry-fixed-engine-v1';assert read(retry/'audit.json')['passed']
old=read(root/'manifest.json');report=read(root/'report.json')
assert report['completed_games']==42000 and len(report['rounds'])==42 and report['status']=='failed'
assert old==read(archive/'manifest.json') and report==read(archive/'report.json')
assert not report['evaluations'] and report['config']['skip_evaluations']
from examples.train_hartford_ppo import source_manifest,atomic_json,checkpoint_hashes
from patchwork_automaton.rl.hartford_preflight import hartford_preflight
from patchwork_automaton.rl.standard_ppo import validate_collection
from patchwork_automaton.neural.compact_identity import IdentityEncoder
ready=hartford_preflight(playback='direct')
assert ready['ready'],ready['errors']
assert IdentityEncoder().manifest()['fingerprint']==read(root/'checkpoints/round-0042/policy.manifest.json')['fingerprint']
for number,files in report['checkpoints'].items():
 assert checkpoint_hashes(root/'checkpoints'/('round-'+number.zfill(4)))==files
preserved={}
folder=root/'rollouts/round-0043'
schedule=read(folder/'schedule.json')
game=next(g for g in schedule if g['index']==42776)
for f in (folder/'results').glob('game-*.json'):
 row=read(f)
 assert sha(f)==sha(archive/'round-0043/results'/f.name)
 if row['index']!=42776:
  validate_collection([row],[next(g for g in schedule if g['index']==row['index'])],42)
  for trajectory in row['trajectories']:assert sha(Path(trajectory['path']))==trajectory['sha256']
  preserved[str(row['index'])]=sha(f)
assert len(preserved)==999
newrow=read(retry/'result.json');validate_collection([newrow],[game],42)
for t in newrow['trajectories']:assert sha(Path(t['path']))==t['sha256']
pins=dict(source=source_manifest(),readiness=ready)
amendment=dict(kind='inert-entry-trigger-loop-recovery-v1',through_checkpoint=42,
 previous_pins=old['pins'],new_pins=pins,engine_revision=b['revision'],
 failed_game=42776,original_failed_result_sha256=sha(folder/'results/game-042776.json'),
 retry_audit=read(retry/'audit.json'),preserved_successful_results=preserved,
 previous_checkpoint_hashes=report['checkpoints'],config_unchanged=True,source_archive_sha256=b['archive_sha256'])
assert not (archive/'runtime-amendment.json').exists()
atomic_json(archive/'runtime-amendment.json',amendment)
newrow['actions']=None
atomic_json(folder/'results/game-042776.json',newrow)
report.setdefault('runtime_amendments',[]).append(amendment)
report.update(status='recovered_ready_to_resume')
report.pop('error',None)
atomic_json(root/'manifest.json',dict(config=old['config'],pins=pins))
atomic_json(root/'report.json',report)
assert all(sha(folder/'results'/('game-'+n.zfill(6)+'.json'))==digest for n,digest in preserved.items())
control=root.parent/'run-trigger-recovery-v1-launch';control.mkdir(exist_ok=False)
guard=root.parent/'run-trigger-recovery-v1-guard'
command=list(s['command_argv'])+['--resume']
atomic_json(control/'command.json',command)
env=dict(s['environment']);env.update(RUSTMAGE_ROOT=b['engine'],RUSTMAGE_LIBRARY=b['rustmage_library'],
 DIRECT_STANDARD_LIBRARY=b['library'],PYTHONPATH=str(project/'src')+':'+str(project/'tests')+':'+str(project))
python=s['remote']['venv_python']
supervised=[python,'-u',str(project/'scripts/guard-hartford-run.py'),'--root',str(guard),
 '--command',str(control/'command.json'),'--cwd',str(project),'--min-available-mib','1536',
 '--memory-grace-seconds','5','--keep-windows-awake','--powershell','/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe']
import shlex
script=['#!/bin/bash','set -euo pipefail','source '+shlex.quote(s['remote']['environment_source'])]
script+=['export '+k+'='+shlex.quote(v) for k,v in env.items()]
script+=['cd '+shlex.quote(str(project)),'exec '+shlex.join(supervised)]
(control/'launch.sh').write_text('\n'.join(script)+'\n')
subprocess.run(['/bin/bash','-n',str(control/'launch.sh')],check=True)
out=dict(s,status='recovery_ready',project=str(project),control=str(control),guard=str(guard),
 launcher=str(control/'launch.sh'),command_argv=command,supervisor_argv=supervised,
 environment=env,engine=ready['engine'],tmux_session='hartford-identity-trigger-recovery',
 runtime_recovery=amendment,source_hashes={name:sha(project/name) for name in s['source_hashes']})
out['remote']=dict(s['remote'],working_directory=str(project),environment_overrides=env)
out.pop('active_recovery',None)
atomic_json(control/'launch.json',out)
# Freeze new inputs only; keep build outputs and all shared old libraries intact.
for folder in [*[project/name for name in ('src','examples','scripts','tests','native')],Path(b['engine'])]:
 for current,dirs,files in os.walk(folder,followlinks=False):
  for f in files:
   path=Path(current)/f
   if not path.is_symlink():path.chmod(path.stat().st_mode & ~0o222)
  Path(current).chmod(0o555)
(project/'pyproject.toml').chmod(0o444)
project.chmod(0o555)
for library in (b['library'],b['rustmage_library']):
 path=Path(library);path.chmod(path.stat().st_mode & ~0o222)
assert checkpoint_hashes(root/'checkpoints/round-0042')==report['checkpoints']['42']
lock.close()
assert not guard.exists()
subprocess.run(['tmux','new-session','-d','-s',out['tmux_session'],'/bin/bash '+shlex.quote(out['launcher'])],check=True)
print(json.dumps(dict(control=str(control),guard=str(guard),session=out['tmux_session'],
 preserved_successful_games=len(preserved),only_failed_game_retried=42776,status='resuming')))
'''

def main():
    launch=HERE.parent/'launch.json';s=json.loads(launch.read_text());b=json.loads((HERE/'stage.json').read_text())
    env=dict(s['environment']);env.update(PYTHONPATH=b['project']+'/src:'+b['project']+'/tests:'+b['project'],
        RUSTMAGE_ROOT=b['engine'],RUSTMAGE_LIBRARY=b['rustmage_library'],DIRECT_STANDARD_LIBRARY=b['library'])
    command=['env',*[k+'='+v for k,v in env.items()],s['remote']['venv_python'],'-c',REMOTE,json.dumps(s),json.dumps(b)]
    r=subprocess.run(s['remote']['ssh_argv_prefix']+[shlex.join(command)],capture_output=True,timeout=90)
    (HERE/'resume-stdout.log').write_bytes(r.stdout);(HERE/'resume-stderr.log').write_bytes(r.stderr)
    if r.returncode:raise RuntimeError(r.stderr.decode())
    print(r.stdout.decode())
    result=json.loads(r.stdout)
    copied=subprocess.run(s['remote']['ssh_argv_prefix']+['cat '+shlex.quote(result['control']+'/launch.json')],
        capture_output=True,check=True,timeout=45).stdout
    (HERE/'launch-before-recovery.json').write_bytes(launch.read_bytes())
    launch.write_bytes(copied)

if __name__=='__main__':main()
