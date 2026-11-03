"""Stage and monitor the four user-authorized evaluation panels."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

PREPARE = r'''
import fcntl,hashlib,json,os,shlex,shutil,subprocess,sys
from pathlib import Path
payload=json.loads(sys.stdin.read());previous=payload['previous']
root=Path(previous['output'])/'post-ab-comparisons-v1';project=root/'project-v1'
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
if root.exists():raise FileExistsError('Existing comparison suite requires inspection')
if shutil.disk_usage(root.parent).free<32*1024**3:raise RuntimeError('Insufficient disk')
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        argv=path.read_bytes().split(b'\0')
        if b'examples.train_hartford_ppo' in argv:raise RuntimeError('Training remains active')
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
preserved=payload['original_files']
for name,digest in preserved.items():
    if sha(name)!=digest:raise ValueError('Original artifact changed: '+name)
root.mkdir()
with (root/'run.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    shutil.copytree(previous['project'],project,ignore=shutil.ignore_patterns('.local','__pycache__'))
    (project/'.local').mkdir();(project/'.local/direct-standard').symlink_to(Path(previous['project'])/'.local/direct-standard',target_is_directory=True)
    target=project/'src/patchwork_automaton/rl/standard_ppo.py';target.chmod(0o644);target.write_text(payload['handoff_source'])
    (root/'evaluate.py').write_text(payload['evaluator'])
    source_hashes={str(path.relative_to(project)):sha(path) for path in project.rglob('*')
        if path.is_file() and '.local' not in path.relative_to(project).parts}
    preserved.update({str(project/name):digest for name,digest in source_hashes.items()})
    preserved[str(root/'evaluate.py')]=sha(root/'evaluate.py')
    for name,body in payload['audit_tools'].items():
        path=root/name;path.write_text(body);preserved[str(path)]=sha(path);path.chmod(0o444)
    models=payload['models']
    reference=Path(previous['arms']['control']['config']['fork']['parent_checkpoint'])/'policy.keras'
    if sha(reference)!=previous['parent_policy_sha256']:raise ValueError('Identity reference changed')
    models['reference']=dict(path=str(reference),sha256=sha(reference),manifest_sha256=sha(reference.with_suffix('.manifest.json')))
    panels=[]
    for kind in ('scripted','neural'):
        for model in ('control','diverse'):
            panels.append(dict(name=model+('-vs-baselines' if kind=='scripted' else '-vs-identity-100k'),
                model=model,kind=kind,games=10000,offset=len(panels)*10000,game_timeout=60 if kind=='scripted' else 20))
    env=dict(previous['environment'],CUDA_VISIBLE_DEVICES='-1',
        PYTHONPATH=f'{project}/src:{project}/tests:{project}')
    config=dict(output=str(root),project=str(project),models=models,panels=panels,
        preserved_files=preserved,source_hashes=source_hashes,
        helper_script=previous['evaluation']['script'],
        seeds=dict(scripted=452611030000,neural=462611030000),
        training=False,workers=48,runners=2,inference_processes=2,opponent_inference_processes=2,
        max_turns=50,max_decisions=6000,temperature=1,
        baseline_panel='Original StandardAggro 3x3 decklists; seven cells1112games, two1108, exact seat/start balance',
        runtime_amendment='Durable completed outcome handoff precedes inference reset. Reset failure may resolve only from all validated original outcomes after collectors close, with explicit transport-error evidence.')
    write(root/'config.json',config)
    sys.path.insert(0,str(root));sys.path.insert(0,str(project/'src'))
    import evaluate
    from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS,STANDARD_DECKS
    for panel in panels:
        rows=evaluate.schedule(panel,config,tuple(HARTFORD_DECKS if panel['kind']=='neural' else STANDARD_DECKS))
        if len(rows)!=10000:raise ValueError('Wrong schedule count')
    guard=root/'guard-v1';control=root/'launch-v1';control.mkdir()
    command=[previous['remote']['venv_python'],'-u',str(root/'evaluate.py'),'--config',str(root/'config.json')]
    write(control/'command.json',command)
    supervised=[previous['remote']['venv_python'],'-u',str(project/'scripts/guard-hartford-run.py'),
        '--root',str(guard),'--command',str(control/'command.json'),'--cwd',str(project),
        '--min-available-mib','1536','--memory-grace-seconds','5','--keep-windows-awake',
        '--powershell','/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe']
    shell=['#!/bin/bash','set -euo pipefail','source '+shlex.quote(previous['remote']['environment_source'])]
    shell+=['export '+key+'='+shlex.quote(value) for key,value in env.items()]
    shell+=['cd '+shlex.quote(str(project)),'exec '+shlex.join(supervised)]
    (control/'launch.sh').write_text('\n'.join(shell)+'\n');subprocess.run(['/bin/bash','-n',str(control/'launch.sh')],check=True)
    launch=dict(status='prepared',remote=previous['remote'],output=str(root),project=str(project),
        config=str(root/'config.json'),config_sha256=sha(root/'config.json'),environment=env,
        guard=str(guard),launcher=str(control/'launch.sh'),session='hartford-post-ab-comparisons',
        command_argv=command,supervisor_argv=supervised,panels=panels,games=40000,
        runtime_amendment=config['runtime_amendment'])
    write(root/'launch.json',launch)
    for path in [root/'config.json',root/'evaluate.py',*[project/name for name in source_hashes]]:path.chmod(0o444)
print(json.dumps(launch))
'''

START = r'''
import fcntl,hashlib,json,shlex,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
s=json.loads(sys.stdin.read());root=Path(s['output'])
with (root/'run.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if Path(s['guard']).exists():raise FileExistsError('Guard already exists')
    if hashlib.sha256(Path(s['config']).read_bytes()).hexdigest()!=s['config_sha256']:raise ValueError('Configuration changed')
    subprocess.run(['tmux','new-session','-d','-s',s['session'],'/bin/bash '+shlex.quote(s['launcher'])],check=True)
    receipt=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),games=40000)
    (root/'start-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''

STATUS = r'''
import hashlib,json,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
s=json.loads(sys.stdin.read());root=Path(s['output']);sys.path.insert(0,str(root));import evaluate
def read(path):return json.loads(path.read_text()) if path.exists() else None
errors=[]
if evaluate.sha(s['config'])!=s['config_sha256']:errors.append('Configuration changed')
config=read(Path(s['config']))
try:evaluate.verify(config)
except Exception as error:errors.append(str(error))
panels={}
for panel in config['panels']:
    directory=root/panel['name'];result=read(directory/'summary.json')
    failures=[str(p) for p in (directory/'failures').glob('*.json')]
    if failures:errors.append(panel['name']+':failed games')
    panels[panel['name']]=dict(games=len(list((directory/'games').glob('*.json'))),
        handoffs=len(list((directory/'handoffs').glob('*.json'))),complete=bool(result),failures=failures,
        score=result['summary']['score'] if result else None,
        transport_error=bool(result and result.get('transport_cleanup_error')))
guard=read(Path(s['guard'])/'guard-status.json')
if guard and guard.get('guard_exit_code') not in (0,None):errors.append('Guard failed')
log=Path(s['guard'])/'runner.log';tail=[]
if log.exists():
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-8000));tail=stream.read().decode(errors='replace').splitlines()[-8:]
print(json.dumps(dict(observed_at=datetime.now(timezone.utc).isoformat(),guard=guard,panels=panels,
    progress=read(root/'progress.json'),integrity_errors=errors,log_tail=tail,
    memavailable_bytes=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024,
    disk_free_bytes=shutil.disk_usage(root).free)))
'''


def ssh(launch, code, payload):
    command = ['env', *[k+'='+v for k,v in launch['environment'].items()],
        launch['remote']['venv_python'], '-c', code]
    result = subprocess.run(launch['remote']['ssh_argv_prefix']+[shlex.join(command)],
        input=json.dumps(payload),text=True,capture_output=True,timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','start','status'))
    args=parser.parse_args()
    if args.action=='prepare':
        previous=json.loads((HERE.parent/'launch.json').read_text())
        manifest=json.loads((HERE.parent/'head-to-head/manifest.json').read_text())
        payload=dict(previous=previous,models=manifest['models'],original_files=manifest['files'],
            evaluator=(HERE/'evaluate.py').read_text(),
            audit_tools={'audit.py':(HERE/'audit.py').read_text()},
            handoff_source=(REPO/'src/patchwork_automaton/rl/standard_ppo.py').read_text())
        launch=ssh(previous,PREPARE,payload)
        (HERE/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        print(json.dumps({k:launch[k] for k in ('status','output','games','panels')},indent=2))
        return
    launch=json.loads((HERE/'launch.json').read_text())
    value=ssh(launch,START if args.action=='start' else STATUS,launch)
    if args.action=='start':
        launch.update(value);(HERE/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        filename='start-receipt.json'
    else:
        filename='latest-snapshot.json'
        if (HERE/filename).exists():
            history=HERE/'monitor-history';history.mkdir(exist_ok=True)
            from datetime import datetime,timezone
            (history/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')).write_bytes((HERE/filename).read_bytes())
    (HERE/filename).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='guard'},indent=2))


if __name__=='__main__':main()
