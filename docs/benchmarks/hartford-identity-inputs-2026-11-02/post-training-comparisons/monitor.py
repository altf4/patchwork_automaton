"""Read-only suite health check, retaining prior local observations."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

BASE = Path(__file__).resolve().parent
REMOTE = r'''
import hashlib,json,os,re,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
s=json.loads(sys.argv[1]);root=Path(s['output']);guard=Path(s['guard'])
def read(p):return json.loads(p.read_text()) if p.exists() else None
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
g=read(guard/'guard-status.json');p=read(root/'progress.json');errors=[]
assert sha(root/'provenance.json')==s['provenance_sha256']
prov=read(root/'provenance.json')
for name,digest in prov['files'].items():
 if not Path(name).exists() or sha(name)!=digest:errors.append('changed:'+name)
panels={}
for panel in s['config']['panels']:
 folder=root/panel['name'];fail=list((folder/'failures').glob('*.json'))
 panels[panel['name']]=dict(completed=len(list((folder/'games').glob('*.json'))),
  failures=[str(f) for f in fail],progress=read(folder/'progress.json'),summary=read(folder/'summary.json'))
 errors.extend('failed_game:'+str(f) for f in fail)
samples=[json.loads(z) for z in (guard/'host-memory-samples.jsonl').read_text().splitlines()] if (guard/'host-memory-samples.jsonl').exists() else []
log=(guard/'runner.log').read_text() if (guard/'runner.log').exists() else ''
alerts=[z[-2000:] for z in log.splitlines() if re.search(r'Traceback|panic|fatal|engine error|non.?finite',z,re.I)]
if alerts:errors.append('runner log errors')
print(json.dumps(dict(observed_at=datetime.now(timezone.utc).isoformat(),guard=g,progress=p,panels=panels,
 errors=errors,log_alerts=alerts,log_tail=log.splitlines()[-8:],
 available_memory_bytes=int(next(z.split()[1] for z in Path('/proc/meminfo').read_text().splitlines() if z.startswith('MemAvailable:')))*1024,
 disk_free_bytes=shutil.disk_usage(root).free,
 minimum_available_bytes=min((z['available_memory_bytes'] for z in samples),default=None),
 maximum_low_memory_seconds=max((z['low_memory_seconds'] for z in samples),default=0))))
'''


def main():
    launch = json.loads((BASE / 'launch.json').read_text())
    config = {key: launch[key] for key in ('output', 'guard', 'provenance_sha256', 'config')}
    result = subprocess.run(launch['remote']['ssh_argv_prefix'] + [shlex.join([
        launch['remote']['venv_python'], '-c', REMOTE, json.dumps(config)])],
        capture_output=True, text=True, check=True, timeout=55)
    snapshot = json.loads(result.stdout)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    history = BASE / 'monitor-history'
    history.mkdir(exist_ok=True)
    path = BASE / 'latest-snapshot.json'
    if path.exists():
        (history / (stamp + '-snapshot.json')).write_bytes(path.read_bytes())
    path.write_text(json.dumps(snapshot, indent=2) + '\n')
    print(json.dumps(dict(observed_at=snapshot['observed_at'],
        guard_status=(snapshot['guard'] or {}).get('status'),
        guard_exit=(snapshot['guard'] or {}).get('guard_exit_code'),
        progress=snapshot['progress'], errors=snapshot['errors'],
        panels={k: dict(completed=v['completed'], summary_present=bool(v['summary']), failures=v['failures']) for k, v in snapshot['panels'].items()},
        available_gib=snapshot['available_memory_bytes'] / 1024**3,
        minimum_available_gib=(snapshot['minimum_available_bytes'] or 0) / 1024**3,
        log_tail=snapshot['log_tail'][-3:]), indent=2))


if __name__ == '__main__':
    main()
