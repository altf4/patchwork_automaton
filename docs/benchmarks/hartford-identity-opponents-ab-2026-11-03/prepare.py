"""Prepare independent full-state forks and a frozen runner; never launch training."""
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent / 'hartford-identity-inputs-2026-11-02/training-100k/launch.json'
REMOTE_BASE = '/home/altf4/benchmarks/hartford-identity-ab-20261103'
REMOTE = r'''
import base64,hashlib,io,json,os,shlex,shutil,subprocess,sys,tarfile
from pathlib import Path
s=json.loads(sys.stdin.read());old=s['previous'];base=Path(s['base']);project=base/'project-v1'
assert not base.exists(),'Existing experiment directory; inspect it rather than overwrite'
assert shutil.disk_usage(base.parent).free>32*1024**3
assert int(next(z.split()[1] for z in Path('/proc/meminfo').read_text().splitlines() if z.startswith('MemAvailable:')))>1536*1024
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:assert b'examples.train_hartford_ppo' not in p.read_bytes().split(b'\0'),'Owned learner remains active'
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
assert read(Path(old['guard'])/'guard-status.json')['guard_exit_code']==0
preserved={str(Path(old['project'])/name):digest for name,digest in old['source_hashes'].items()}
for root in (Path(old['output']),Path(old['original_output'])):
 report=read(root/'report.json');assert report['status']=='complete'
 for number,files in report['checkpoints'].items():
  for name,digest in files.items():preserved[str(root/'checkpoints'/('round-'+number.zfill(4))/name)]=digest
 for name in ('report.json','manifest.json','progress.json','checkpoints/latest.json'):
  preserved[str(root/name)]=sha(root/name)
for p,h in preserved.items():assert sha(p)==h,p
base.mkdir();project.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(s['archive'])),mode='r:gz') as t:t.extractall(project,filter='data')
(project/'.local').mkdir();(project/'.local/direct-standard').symlink_to(Path(old['project'])/'.local/direct-standard',target_is_directory=True)
inputs=base/'inputs/identity-100k';(inputs/'checkpoints').mkdir(parents=True)
shutil.copy2(Path(old['output'])/'report.json',inputs/'report.json')
for number in (25,50,75,97,98,99,100):
 source=Path(old['output'])/'checkpoints'/f'round-{number:04d}';dest=inputs/'checkpoints'/source.name
 shutil.copytree(source,dest)
 for p in dest.iterdir():
  assert sha(p)==sha(source/p.name);preserved[str(p)]=sha(p);p.chmod(0o444)
preserved[str(inputs/'report.json')]=sha(inputs/'report.json');(inputs/'report.json').chmod(0o444)
prior_eval=s['previous_comparison'];prior_provenance=Path(prior_eval['output'])/'provenance.json'
assert sha(prior_provenance)==prior_eval['provenance_sha256'],'Prior comparison provenance changed'
old_files=read(prior_provenance)['files'];preserved[str(prior_provenance)]=sha(prior_provenance)
engine_files={p:old_files[p] for p in (old['engine']['library'],old['engine']['build_manifest'],old['environment']['RUSTMAGE_LIBRARY'])}
for p,h in engine_files.items():assert sha(p)==h,p
assert engine_files[old['engine']['library']]==old['engine']['library_sha256']
env=dict(old['environment']);env.update(PYTHONPATH=f'{project}/src:{project}/tests:{project}',
 PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/compact-ppo-mpl',PATCHWORK_ARENA_POLL_INFERENCE='1')
python=old['remote']['venv_python'];runenv=dict(os.environ,**env);ops=project/'scripts/hartford-ab'
source_hashes={str(p.relative_to(project)):sha(p) for p in project.rglob('*') if p.is_file() and '.local' not in p.relative_to(project).parts}
parent=inputs/'checkpoints/round-0100';bundle={p.name:sha(p) for p in parent.iterdir()}
digest=hashlib.sha256(json.dumps(bundle,sort_keys=True,separators=(',',':')).encode()).hexdigest()
arms={}
for name,recipe in [('control','current'),('diverse','diverse')]:
 output=base/name
 command=[python,'-u','-m','examples.train_hartford_ppo','--architecture','compact-identity',
  '--fork-from',str(parent),'--fork-from-sha256',digest,'--opponent-recipe',recipe,
  '--output',str(output),'--rounds','50','--seed','422611030000','--eval-seed','932611030000',
  '--skip-evaluations','--workers','48','--runners','2','--inference-processes','2',
  '--opponent-inference-processes','2','--compact-input-mib','512','--game-timeout','20','--update-timeout','1800']
 prep=subprocess.run(command+['--prepare-fork-only'],env=runenv,cwd=project,capture_output=True,text=True)
 (base/(name+'-prepare-stdout.log')).write_text(prep.stdout);(base/(name+'-prepare-stderr.log')).write_text(prep.stderr)
 if prep.returncode:raise RuntimeError(prep.stderr)
 arms[name]=dict(output=str(output),recipe=recipe,command=command+['--resume'],
  config=read(output/'manifest.json')['config'],initial_checkpoint_hashes={p.name:sha(p) for p in (output/'checkpoints/round-0000').iterdir()})
for name in ('policy.keras','policy.manifest.json','optimizer.npz','critic-optimizer.npz'):
 assert arms['control']['initial_checkpoint_hashes'][name]==arms['diverse']['initial_checkpoint_hashes'][name]==bundle[name]
for key in ('optimizer_steps','critic_optimizer_steps','trainer_rng_state'):
 assert read(base/'control/checkpoints/round-0000/state.json')[key]==read(base/'diverse/checkpoints/round-0000/state.json')[key]==read(parent/'state.json')[key]
guard=base/'run-guard-v1';control=base/'run-launch-v1';control.mkdir()
record=dict(status='prepared',remote=old['remote'],project=str(project),output=str(base),remote_launch=str(base/'launch.json'),
 source_hashes=source_hashes,engine=old['engine'],engine_files=engine_files,preserved_files=preserved,
 environment=env,arms=arms,guard=str(guard),control=str(control),training_audit=str(base/'training-audit.json'),
 evaluation=dict(output=str(base/'head-to-head'),script=str(ops/'evaluate.py'),guard=str(guard),
  seed=432611030000,games=10000,workers=48,runners=2,inference_processes=2,opponent_inference_processes=2,
  game_timeout=20,max_turns=50,max_decisions=6000),
 parent_policy_sha256=bundle['policy.keras'],parent_bundle_sha256=digest,
 collection_games=100000,training_games=80000,holdout_games=20000,final_evaluation_games=10000,
 execution='Sequential arms on identical GPU/CPU allocation; paired experimental schedules',
 git_commit=s['git_commit'],session='hartford-identity-opponents-ab')
command=[python,'-u',str(ops/'run_suite.py'),'--launch',str(base/'launch.json')];write(control/'command.json',command)
supervised=[python,'-u',str(project/'scripts/guard-hartford-run.py'),'--root',str(guard),
 '--command',str(control/'command.json'),'--cwd',str(project),'--min-available-mib','1536',
 '--memory-grace-seconds','5','--keep-windows-awake','--powershell','/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe']
shell=['#!/bin/bash','set -euo pipefail','source '+shlex.quote(old['remote']['environment_source'])]
shell+=['export '+k+'='+shlex.quote(v) for k,v in env.items()]
shell+=['cd '+shlex.quote(str(project)),'exec '+shlex.join(supervised)]
(control/'launch.sh').write_text('\n'.join(shell)+'\n');subprocess.run(['/bin/bash','-n',str(control/'launch.sh')],check=True)
record.update(launcher=str(control/'launch.sh'),supervisor_argv=supervised,command_argv=command)
record['remote']=dict(record['remote'],working_directory=str(project),environment_overrides=env)
write(base/'launch.json',record)
for name,digest in source_hashes.items():
 p=project/name;assert sha(p)==digest,name;p.chmod(0o444)
for p,h in preserved.items():assert sha(p)==h,p
print(json.dumps(record))
'''


def main():
    previous = json.loads(PREVIOUS.read_text())
    originals = HERE.parent / 'hartford-identity-inputs-2026-11-02/preservation/frozen-source.tar.gz'
    files = {}
    with tarfile.open(originals, 'r:gz') as archive:
        for item in archive.getmembers():files[item.name] = archive.extractfile(item).read()
    # Keep the exact recovered native/runtime inputs; overlay current Python
    # support and tests, whose hashes are recorded before anything can run.
    for folder in ('src/patchwork_automaton', 'examples'):
        for path in (ROOT / folder).rglob('*.py'):
            files[str(path.relative_to(ROOT))] = path.read_bytes()
    for path in (ROOT / 'tests').glob('test_*.py'):
        files[str(path.relative_to(ROOT))] = path.read_bytes()
    for name in ('audit_training.py','run_suite.py','evaluate.py','acceptance.py'):
        files['scripts/hartford-ab/' + name] = (HERE / name).read_bytes()
    for name in ('evaluate.py','finalize.py'):
        files[str((HERE/name).relative_to(ROOT))]=(HERE/name).read_bytes()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name, data in sorted(files.items()):
            item = tarfile.TarInfo(name);item.size=len(data);item.mode=0o644
            archive.addfile(item,io.BytesIO(data))
    comparison=json.loads((PREVIOUS.parent.parent/'post-training-comparisons/launch.json').read_text())
    payload = dict(previous=previous,previous_comparison=comparison,base=REMOTE_BASE,archive=base64.b64encode(stream.getvalue()).decode(),
                   git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    result = subprocess.run(previous['remote']['ssh_argv_prefix'] + [shlex.join(['python3','-c',REMOTE])],
                            input=json.dumps(payload),text=True,capture_output=True,timeout=180)
    (HERE/'prepare-stdout.log').write_text(result.stdout);(HERE/'prepare-stderr.log').write_text(result.stderr)
    if result.returncode:raise RuntimeError(result.stderr)
    record=json.loads(result.stdout);(HERE/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('status','project','output','collection_games','final_evaluation_games')}))


if __name__ == '__main__':main()
