"""Read-only A/B health observation, with archived local observations."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess

BASE = Path(__file__).resolve().parent
REMOTE = r'''
import hashlib,json,math,os,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
s=json.loads(sys.stdin.read());sys.path.insert(0,str(Path(s['evaluation']['script']).parent))
from audit_training import verify_static,finite_checkpoint
def read(p):return json.loads(p.read_text()) if p.exists() else None
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
base=Path(s['remote_launch']).parent;guard=Path(s['guard']);errors=[]
try:checked=verify_static(s)
except Exception as e:checked=0;errors.append(str(e))
x=dict(observed_at=datetime.now(timezone.utc).isoformat(),guard=read(guard/'guard-status.json'),
 progress=read(base/'progress.json'),arms={},preserved_files_checked=checked,
 memavailable_bytes=int(next(z.split()[1] for z in Path('/proc/meminfo').read_text().splitlines() if z.startswith('MemAvailable:')))*1024,
 disk_free_bytes=shutil.disk_usage(base).free)
def finite(v,path):
 if isinstance(v,float) and not math.isfinite(v):errors.append('nonfinite:'+path)
 elif isinstance(v,dict):
  for k,z in v.items():finite(z,path+'.'+str(k))
 elif isinstance(v,list):
  for i,z in enumerate(v):finite(z,path+'.'+str(i))
for name,arm in s['arms'].items():
 root=Path(arm['output']);report=read(root/'report.json') or {};finite(report,name)
 info=dict(status=report.get('status'),completed_games=report.get('completed_games',0),
  error=report.get('error'),progress=read(root/'progress.json'),rounds=len(report.get('rounds',[])),
  latest_rounds=[{k:r.get(k) for k in ('round','collection_seconds','update_seconds','validation_seconds','opponent_games','update','validation')} for r in report.get('rounds',[])[-3:]])
 for number,files in report.get('checkpoints',{}).items():
  for fn,digest in files.items():
   p=root/'checkpoints'/('round-'+number.zfill(4))/fn
   if not p.exists() or sha(p)!=digest:errors.append('checkpoint:'+str(p))
 latest=read(root/'checkpoints/latest.json')
 if latest:
  cp=root/'checkpoints'/latest['checkpoint']
  try:info['finite_model_datasets']=finite_checkpoint(cp)
  except Exception as e:errors.append(name+':'+str(e))
  state=read(cp/'state.json');info['checkpoint']={k:state.get(k) for k in ('round','optimizer_steps','critic_optimizer_steps','lineage_checkpoint')}
 for row in report.get('rounds',[]):
  p=Path(row['games_file'])
  if not p.exists() or sha(p)!=row['games_sha256']:errors.append('game summaries:'+str(p))
  if (row['training_games'],row['validation_games'])!=(800,200):errors.append(name+':partition')
 if report.get('evaluations') or not report.get('config',{}).get('skip_evaluations'):errors.append(name+':unexpected training evaluations')
 results=list((root/'rollouts').glob('round-*/results/game-*.json'))
 info['durable_unpruned_results']=len(results)
 info['failures']=[str(p) for p in (root/'rollouts').glob('round-*/failures/*.json')]
 if info['failures']:errors.append(name+':failed games')
 if report.get('status')=='failed':errors.append(name+':'+str(report.get('error')))
 x['arms'][name]=info
ev=Path(s['evaluation']['output']);x['evaluation_progress']=read(ev/'progress.json')
x['evaluation_summary']=read(ev/'summary.json')
x['evaluation_failures']=[str(p) for p in (ev/'failures').glob('*.json')]
if x['evaluation_failures']:errors.append('head_to_head:failed attempts')
if (ev/'manifest.json').exists():
 try:
  from evaluate import load_results,verify_files
  em=read(ev/'manifest.json');verify_files(em['files'])
  x['evaluation_durable_games']=len(load_results(ev,read(ev/'schedule.json')))
 except Exception as e:errors.append('head_to_head:'+str(e))
log=guard/'runner.log';x['log_bytes']=log.stat().st_size if log.exists() else 0
if log.exists():
 with log.open('rb') as f:f.seek(max(0,x['log_bytes']-16000));x['log_tail']=f.read().decode(errors='replace').splitlines()[-25:]
samples=[]
if (guard/'host-memory-samples.jsonl').exists():
 for line in (guard/'host-memory-samples.jsonl').read_text().splitlines():
  try:samples.append(json.loads(line))
  except json.JSONDecodeError:pass
x['resource_history']=dict(samples=len(samples),minimum_available_bytes=min((v['available_memory_bytes'] for v in samples),default=None),maximum_low_memory_seconds=max((v['low_memory_seconds'] for v in samples),default=0))
g=x['guard'] or {};pid=g.get('child_pid');x['owned_child_alive']=False
try:x['owned_child_alive']=str(Path(s['evaluation']['script']).with_name('run_suite.py')).encode() in Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')
except (FileNotFoundError,ProcessLookupError):pass
if g.get('guard_exit_code') not in (None,0):errors.append('guard:'+str(g))
x['integrity_errors']=errors;print(json.dumps(x))
'''


def main():
    launch = json.loads((BASE / 'launch.json').read_text())
    prior = json.loads((BASE / 'monitor-state.json').read_text()) if (BASE / 'monitor-state.json').exists() else {}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    history = BASE / 'monitor-history'; history.mkdir(exist_ok=True)
    for name in ('monitor-state.json', 'latest-monitor-snapshot.json'):
        if (BASE / name).exists():shutil.copy2(BASE / name, history / (stamp + '-' + name))
    env = launch['environment']
    command = shlex.join(['env', *[f'{k}={v}' for k,v in env.items()], launch['remote']['venv_python'], '-c', REMOTE])
    result = subprocess.run(launch['remote']['ssh_argv_prefix'] + [command], input=json.dumps(launch),
                            text=True, capture_output=True, timeout=180, check=True)
    snapshot = json.loads(result.stdout)
    (BASE / 'latest-monitor-snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    signature = dict(arms={k:[v['completed_games'],v['durable_unpruned_results']] for k,v in snapshot['arms'].items()},
                     log_bytes=snapshot['log_bytes'],evaluation=snapshot['evaluation_progress'])
    changed = signature != prior.get('progress_signature')
    last_change = snapshot['observed_at'] if changed else prior.get('last_progress_at',snapshot['observed_at'])
    state = dict(observed_at=snapshot['observed_at'],progress_signature=signature,last_progress_at=last_change,
                 progress_changed=changed,integrity_errors=snapshot['integrity_errors'],
                 guard=snapshot['guard'],previous_observed_at=prior.get('observed_at'),
                 last_alert=prior.get('last_alert'))
    (BASE / 'monitor-state.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state,indent=2))


if __name__ == '__main__':main()
