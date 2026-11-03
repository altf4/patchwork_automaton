"""Read-only remote health check; saves observations locally, never restarts."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess

BASE=Path(__file__).resolve().parent
REMOTE=r'''
import hashlib,json,math,os,shutil,sys,io,zipfile,re
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,h5py
s=json.loads(sys.argv[1]);root=Path(s['output']);guard=Path(s['guard']);project=Path(s['project'])
def read(p):return json.loads(p.read_text()) if p.exists() else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=read(root/'report.json') or {}
x=dict(observed_at=datetime.now(timezone.utc).isoformat(),guard=read(guard/'guard-status.json'),
 progress=read(root/'progress.json'),report=report,checkpoint=None,
 available_memory_bytes=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024,
 disk_free_bytes=shutil.disk_usage(root if root.exists() else root.parent).free)
errors=[]
for name,digest in s['source_hashes'].items():
 if not (project/name).exists() or sha(project/name)!=digest:errors.append('source:'+name)
if sha(Path(s['engine']['library']))!=s['engine']['library_sha256']:errors.append('engine')
for name,digest in s['parent_checkpoint_hashes'].items():
 for folder in (Path(s['control']).parent/'inputs/source-200k',Path(s['original_output'])/'checkpoints/round-0200'):
  if sha(folder/name)!=digest:errors.append('parent:'+str(folder/name))
for number,files in report.get('checkpoints',{}).items():
 for name,digest in files.items():
  p=root/'checkpoints'/('round-'+number.zfill(4))/name
  if not p.exists() or sha(p)!=digest:errors.append('checkpoint:'+str(p))
latest=read(root/'checkpoints/latest.json')
model_datasets=[]
if latest:
 cp=root/'checkpoints'/latest['checkpoint'];x['checkpoint']=read(cp/'state.json')
 for name in ('optimizer.npz','critic-optimizer.npz'):
  with np.load(cp/name,allow_pickle=False) as data:
   if any(not np.isfinite(data[key]).all() for key in data.files):errors.append('nonfinite:'+name)
 with zipfile.ZipFile(cp/'policy.keras') as archive:
  with h5py.File(io.BytesIO(archive.read('model.weights.h5')),'r') as h:
   def check_model(name,item):
    if isinstance(item,h5py.Dataset) and np.issubdtype(item.dtype,np.number):
     model_datasets.append(name)
     if not np.isfinite(item[()]).all():errors.append('nonfinite:model:'+name)
   h.visititems(check_model)
 if 'trainer_rng_state' not in x['checkpoint']:errors.append('checkpoint RNG missing')
x['finite_model_datasets']=len(model_datasets)
for row in report.get('rounds',[]):
 f=Path(row['games_file'])
 if not f.exists() or sha(f)!=row['games_sha256']:errors.append('game summaries changed:'+str(row['round']))
 if row['training_games']!=800 or row['validation_games']!=200:errors.append('partition:'+str(row['round']))
 if row['round']>=4 and sorted(row['opponent_games'].values())!=[100,100,100,700]:
  errors.append('opponent quotas:'+str(row['round']))
if report and (report.get('evaluations') or not report['config'].get('skip_evaluations')):
 errors.append('evaluation policy')
if any((root/'evaluation').rglob('game-*.json')):errors.append('unexpected evaluation artifacts')
samples=[json.loads(line) for line in (guard/'host-memory-samples.jsonl').read_text().splitlines()]
x['resource_history']=dict(samples=len(samples),
 minimum_available_bytes=min((v['available_memory_bytes'] for v in samples),default=None),
 maximum_low_memory_seconds=max((v['low_memory_seconds'] for v in samples),default=0))
def finite(v,path):
 if isinstance(v,float) and not math.isfinite(v):errors.append('nonfinite:'+path)
 elif isinstance(v,dict):
  for k,z in v.items():finite(z,path+'.'+str(k))
 elif isinstance(v,list):
  for i,z in enumerate(v):finite(z,path+'.'+str(i))
finite(report,'report');finite(x['checkpoint'],'checkpoint')
failures=list((root/'rollouts').glob('round-*/failures/*.json'))
x['failure_files']=[str(p) for p in failures]
for p in failures:errors.append('failed_game:'+str(p))
results=list((root/'rollouts').glob('round-*/results/game-*.json'))
x['durable_unpruned_results']=len(results)
for p in results:
 row=read(p)
 if row.get('status') not in ('win','loss','draw'):errors.append('bad_outcome:'+str(p))
log=guard/'runner.log'
if log.exists():
 with log.open('rb') as f:
  f.seek(max(0,log.stat().st_size-16000));x['log_tail']=f.read().decode(errors='replace').splitlines()[-20:]
else:x['log_tail']=[]
x['log_alerts']=[]
if log.exists():
 x['log_alerts']=[line[-2000:] for line in log.read_text().splitlines()
  if re.search(r'Traceback|panic|fatal|engine error|non.?finite',line,re.I)]
 if x['log_alerts']:errors.append('suspicious runner logs')
g=x['guard'] or {};pid=g.get('child_pid')
try:
 argv=Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')
 x['owned_child_alive']=b'examples.train_hartford_ppo' in argv and str(root).encode() in argv
except (OSError,TypeError):x['owned_child_alive']=False
x['integrity_errors']=errors
print(json.dumps(x))
'''

def main():
    launch=BASE/'launch.json';s=json.loads(launch.read_text())
    # Recovery archives contain full historical manifests. Send only fields
    # used by the read-only checker, avoiding OS per-argument size limits.
    remote_config={key:s[key] for key in ('output','guard','project','source_hashes',
        'engine','control','original_output','parent_checkpoint_hashes')}
    result=subprocess.run(s['remote']['ssh_argv_prefix']+[shlex.join([
        s['remote']['venv_python'],'-c',REMOTE,json.dumps(remote_config)])],
        capture_output=True,text=True,check=True,timeout=55)
    x=json.loads(result.stdout)
    history=BASE/'monitor-history';history.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for name in ('latest-monitor-snapshot.json','monitor-state.json'):
        p=BASE/name
        if p.exists():(history/(stamp+'-'+name)).write_bytes(p.read_bytes())
    (BASE/'latest-monitor-snapshot.json').write_text(json.dumps(x,indent=2)+'\n')
    report=x['report'];g=x['guard'] or {};cp=x['checkpoint'] or {}
    state=dict(observed_at=x['observed_at'],guard_status=g.get('status'),owned_child_alive=x['owned_child_alive'],
        progress=x['progress'],checkpoint=cp.get('round'),committed_games=report.get('completed_games',0),
        durable_unpruned_results=x['durable_unpruned_results'],available_memory_bytes=x['available_memory_bytes'],
        disk_free_bytes=x['disk_free_bytes'],integrity_errors=x['integrity_errors'],failure_files=x['failure_files'],
        resource_history=x['resource_history'],finite_model_datasets=x['finite_model_datasets'])
    state['last_check_result']='healthy_progress' if not state['integrity_errors'] and state['owned_child_alive'] and g.get('status')=='running' else 'needs_review'
    (BASE/'monitor-state.json').write_text(json.dumps(state,indent=2)+'\n')
    s.update(status=g.get('status','starting'),last_observed_at=x['observed_at'],
        started_at=g.get('started_at'),tmux_session='hartford-identity-100k',last_observation=state)
    launch.write_text(json.dumps(s,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(state,log_tail=x['log_tail'][-5:]),indent=2))

if __name__=='__main__':main()
