"""Stage a reviewed committed engine repair and build separate guarded libraries."""
import hashlib,io,json,shlex,subprocess,sys,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
ENGINE=REPO/'.local/engine-trigger-loop-fix-v1'

REMOTE=r'''
import hashlib,io,json,os,shlex,shutil,subprocess,sys,tarfile
from pathlib import Path
s=json.loads(sys.argv[1]);revision=sys.argv[2];expected=sys.argv[3]
blob=sys.stdin.buffer.read();assert hashlib.sha256(blob).hexdigest()==expected
base=Path(s['output']).parent;project=base/'project-trigger-v2'
if project.exists():raise FileExistsError(project)
shutil.copytree(s['project'],project,symlinks=True,copy_function=shutil.copyfile)
for current,dirs,files in os.walk(project,followlinks=False):
 Path(current).chmod(0o755)
 for name in files:
  f=Path(current)/name
  if not f.is_symlink():f.chmod(0o644)
oldlink=project/'.local/direct-standard';assert oldlink.is_symlink();oldlink.unlink();oldlink.mkdir()
engine=oldlink/('rustmage-'+revision[:12]);engine.mkdir()
with tarfile.open(fileobj=io.BytesIO(blob),mode='r:') as archive:archive.extractall(engine,filter='data')
(engine/'.direct-standard-revision').write_text(revision+'\n')
pinpath=project/'src/patchwork_automaton/data/runtime-engine.json'
pin=json.loads(pinpath.read_text());pin.update(runtime_revision=revision,previous_runtime_revision=pin['runtime_revision'],
 reason='Validated mandatory exile/return loop with unplaceable entry-damage trigger',
 archive_sha256=expected)
pinpath.write_text(json.dumps(pin,indent=2)+'\n')
cargo=project/'native/direct-standard/Cargo.toml'
lines=cargo.read_text().splitlines()
for i,line in enumerate(lines):
 if line.startswith('rustmage = '):lines[i]='rustmage = { path = "../../.local/direct-standard/rustmage-'+revision[:12]+'" }'
cargo.write_text('\n'.join(lines)+'\n')
target=oldlink/('target-'+revision[:12]+'-capped256')
control=base/'run-trigger-recovery-v1-build';control.mkdir()
env=dict(s['environment']);env.update(PYTHONPATH='src:tests:.',RUSTMAGE_ROOT=str(engine),
 CARGO_TARGET_DIR=str(target),
 PATH='/home/altf4/benchmarks/hartford-compact-ppo-20261101/native-toolchain-loop-v1/bin:/home/altf4/benchmarks/hartford-compact-ppo-20261101/rust-toolchain-1.98.0/bin:'+os.environ['PATH'],
 CARGO_HOME='/home/altf4/benchmarks/hartford-compact-ppo-20261101/cargo-home-loop-v1')
python=s['remote']['venv_python']
commands=[[python,str(engine/'scripts/resource_guard.py'),'--','cargo','build','--release','--offline','--locked',
 '--manifest-path',str(engine/'Cargo.toml'),'--lib'],
 [python,str(project/'scripts/build-direct-standard.py')]]
script=['#!/bin/bash','set -euo pipefail','source '+shlex.quote(s['remote']['environment_source'])]
script+=['export '+k+'='+shlex.quote(v) for k,v in env.items()]
script+=['cd '+shlex.quote(str(project))]
script+=['exec >'+shlex.quote(str(control/'build.log'))+' 2>&1',
 "trap 'code=$?; echo $code > "+shlex.quote(str(control/'exit-code'))+"' EXIT"]
script+=[shlex.join(c) for c in commands]
(control/'build.sh').write_text('\n'.join(script)+'\n')
record=dict(project=str(project),engine=str(engine),target=str(target),control=str(control),
 revision=revision,archive_sha256=expected,environment=env,commands=commands,
 library=str(target/'release/libpatchwork_direct_standard.so'),
 rustmage_library=str(target/'release/librustmage.so'))
(control/'stage.json').write_text(json.dumps(record,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s','hartford-inert-trigger-build',
 '/bin/bash '+shlex.quote(str(control/'build.sh'))],check=True)
print(json.dumps(record))
'''

def main():
    s=json.loads((HERE.parent/'launch.json').read_text())
    revision=subprocess.check_output(['git','-C',str(ENGINE),'rev-parse','HEAD'],text=True).strip()
    if revision=='4bedacf19687e98345f26c7b7492cd2bc149fe38':raise RuntimeError('Repair must be committed first')
    if subprocess.check_output(['git','-C',str(ENGINE),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Only clean committed repair snapshots can be staged')
    archive=subprocess.check_output(['git','-C',str(ENGINE),'archive',revision])
    digest=hashlib.sha256(archive).hexdigest()
    r=subprocess.run(s['remote']['ssh_argv_prefix']+[shlex.join(['python3','-c',REMOTE,json.dumps(s),revision,digest])],
        input=archive,capture_output=True,check=False,timeout=90)
    (HERE/'stage-build-stderr.log').write_bytes(r.stderr)
    if r.returncode:raise RuntimeError(r.stderr.decode())
    (HERE/'stage.json').write_bytes(r.stdout)
    print(r.stdout.decode())

if __name__=='__main__':main()
