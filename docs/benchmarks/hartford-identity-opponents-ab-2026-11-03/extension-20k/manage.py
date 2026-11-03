"""Prepare, launch and observe the fixed 20K evaluation extension over SSH."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent

PREPARE = r'''
import fcntl,hashlib,importlib.util,json,os,shlex,shutil,subprocess,sys
from pathlib import Path
payload=json.loads(sys.stdin.read()); old=payload['previous']
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value): Path(path).write_text(json.dumps(value,indent=2)+'\n')
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
root=Path(old['output'])/'head-to-head-extension-20k-v1'
if root.exists(): raise FileExistsError('Inspect existing extension before resuming')
if shutil.disk_usage(root.parent).free < 32*1024**3: raise RuntimeError('Less than 32 GiB free disk')
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        argv=path.read_bytes().split(b'\0')
        if b'examples.train_hartford_ppo' in argv: raise RuntimeError('Learner is active')
    except (FileNotFoundError,PermissionError,ProcessLookupError): pass
original=module(old['evaluation']['script'],'original_evaluation')
audited=original.audit(old)
if not audited['audit']['passed']: raise ValueError('Original evaluation audit failed')
if audited['manifest'] != payload['original_manifest']: raise ValueError('Original provenance differs from fetched audit')
root.mkdir()
with (root/'run.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    guard=root/'guard-v1'; control=root/'launch-v1';control.mkdir()
    launch=dict(old)
    for key in ('completion','completed_at','started_at','automation'): launch.pop(key,None)
    launch.update(status='prepared',output=str(root),guard=str(guard),control=str(control),
        remote_launch=str(root/'launch.json'),session='hartford-ab-extra-20k')
    launch['evaluation']=dict(old['evaluation'],output=str(root/'games-20k'),script=str(root/'evaluate.py'),
        guard=str(guard),seed=432611040000,games=20000)
    launch['extension']=dict(original_launch=old['remote_launch'],original_output=old['evaluation']['output'],
        original_games=10000,additional_games=20000,combined_games=30000,
        seed_min=432611040000,seed_max=432611059999,original_audit=audited['audit'],
        statistical_note='Fixed additional sample requested after inspecting 10K results. Combined 95% intervals are nominal fixed-sample intervals.')
    preserved=dict(old['preserved_files'])
    for path in Path(old['evaluation']['output']).rglob('*'):
        if path.is_file(): preserved[str(path)]=sha(path)
    for path in (old['remote_launch'], str(Path(old['output'])/'progress.json'),str(Path(old['guard'])/'guard-status.json')):
        preserved[path]=sha(path)
    launch['preserved_files']=preserved
    launch['final_evaluation_games']=30000
    for name,body in payload['scripts'].items():
        path=root/name;path.write_text(body);path.chmod(0o444)
        launch['preserved_files'][str(path)]=sha(path)
    write(root/'original-audit.json',audited['audit'])
    write(root/'original-launch.json',old)
    launch['preserved_files'][str(root/'original-launch.json')]=sha(root/'original-launch.json')
    new=module(root/'evaluate.py','extension_evaluation')
    admitted=new.input_manifest(launch)
    from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS
    planned=new.schedule(admitted['config'],HARTFORD_DECKS,admitted['models']['control']['sha256'])
    original_rows=json.loads(__import__('gzip').decompress(__import__('base64').b64decode(audited['games_gzip_b64'])))
    if {row['index'] for row in planned} & {row['index'] for row in original_rows}: raise ValueError('Reused game ID')
    if {row['seed'] for row in planned} & {row['seed'] for row in original_rows}: raise ValueError('Reused seed')
    write(root/'schedule-audit.json',dict(passed=True,new_games=len(planned),original_games=len(original_rows),
        combined_games=30000,original_schedule_unchanged=True,seed_and_id_disjoint=True,
        new_schedule_sha256=new.value_sha(planned)))
    command=[old['remote']['venv_python'],'-u',str(root/'evaluate.py'),'--launch',str(root/'launch.json')]
    write(control/'command.json',command)
    supervisor=[old['remote']['venv_python'],'-u',str(Path(old['project'])/'scripts/guard-hartford-run.py'),
        '--root',str(guard),'--command',str(control/'command.json'),'--cwd',old['project'],
        '--min-available-mib','1536','--memory-grace-seconds','5','--keep-windows-awake',
        '--powershell','/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe']
    shell=['#!/bin/bash','set -euo pipefail','source '+shlex.quote(old['remote']['environment_source'])]
    shell+=['export '+key+'='+shlex.quote(value) for key,value in old['environment'].items()]
    shell+=['cd '+shlex.quote(old['project']),'exec '+shlex.join(supervisor)]
    (control/'launch.sh').write_text('\n'.join(shell)+'\n')
    subprocess.run(['/bin/bash','-n',str(control/'launch.sh')],check=True)
    launch.update(command_argv=command,supervisor_argv=supervisor,launcher=str(control/'launch.sh'))
    write(root/'launch.json',launch)
print(json.dumps(launch))
'''

START = r'''
import fcntl,hashlib,json,shlex,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
launch=json.loads(sys.stdin.read());root=Path(launch['output'])
if Path(launch['guard']).exists(): raise FileExistsError('Guard already exists')
with (root/'run.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,digest in launch['preserved_files'].items():
        with Path(name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=digest: raise ValueError(name)
    subprocess.run(['tmux','new-session','-d','-s',launch['session'],
        '/bin/bash '+shlex.quote(launch['launcher'])],check=True)
    receipt=dict(status='started',started_at=datetime.now(timezone.utc).isoformat(),
        additional_games=20000,combined_games=30000)
    (root/'start-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''

STATUS = r'''
import hashlib,importlib.util,json,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
launch=json.loads(sys.stdin.read());root=Path(launch['evaluation']['output'])
def read(path): return json.loads(path.read_text()) if path.exists() else None
spec=importlib.util.spec_from_file_location('extension_evaluation',launch['evaluation']['script'])
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
errors=[];rows={}
try:
    evaluation.verify_files({**launch['preserved_files'],**launch['engine_files'],
        **{str(Path(launch['project'])/name):digest for name,digest in launch['source_hashes'].items()}})
    manifest=read(root/'manifest.json')
    if manifest:
        evaluation.verify_files(manifest['files'])
        if 'recovery' in launch:
            spec=importlib.util.spec_from_file_location('publication_recovery',launch['recovery']['script'])
            recovered=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovered)
            rows=recovered.validated_rows(launch)
        else:
            rows=evaluation.load_results(root,read(root/'schedule.json'))
except Exception as error: errors.append(str(error))
guard=read(Path(launch['guard'])/'guard-status.json')
if guard and guard.get('guard_exit_code') not in (None,0): errors.append('Guard failed')
log=Path(launch['guard'])/'runner.log'
tail=[]
if log.exists():
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-8000));tail=stream.read().decode(errors='replace').splitlines()[-6:]
print(json.dumps(dict(observed_at=datetime.now(timezone.utc).isoformat(),guard=guard,
    progress=read(root/'progress.json'),durable_games=len(rows),integrity_errors=errors,
    summary=read(root/'summary.json'),log_tail=tail,
    memavailable_bytes=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024,
    disk_free_bytes=shutil.disk_usage(Path(launch['output'])).free)))
'''


def ssh(launch, code, payload):
    command = ['env', *[key + '=' + value for key, value in launch['environment'].items()],
               launch['remote']['venv_python'], '-c', code]
    result = subprocess.run(launch['remote']['ssh_argv_prefix'] + [shlex.join(command)],
        input=json.dumps(payload),capture_output=True,text=True,timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','start','status'))
    args = parser.parse_args()
    if args.action == 'prepare':
        previous = json.loads((HERE.parent/'launch.json').read_text())
        payload = dict(previous=previous,
            original_manifest=json.loads((HERE.parent/'head-to-head/manifest.json').read_text()),
            scripts={'evaluate.py':(HERE.parent/'evaluate.py').read_text()})
        launch = ssh(previous,PREPARE,payload)
        (HERE/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        print(json.dumps({key:launch[key] for key in ('status','output','evaluation')},indent=2))
    else:
        launch = json.loads((HERE/'launch.json').read_text())
        value = ssh(launch,START if args.action == 'start' else STATUS,launch)
        if args.action == 'start':
            launch.update(value)
            (HERE/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
            filename = 'start-receipt.json'
        else:
            filename = 'latest-snapshot.json'
            if (HERE/filename).exists():
                history=HERE/'monitor-history';history.mkdir(exist_ok=True)
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                (history/(stamp+'.json')).write_bytes((HERE/filename).read_bytes())
        (HERE/filename).write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps({key:value[key] for key in value if key not in ('summary','guard')},indent=2))


if __name__ == '__main__':
    main()
