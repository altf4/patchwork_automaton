"""Build the experimental typed Standard bridge with RustMage's resource guard.

The engine source is exported from the immutable commit used by the 250K model;
existing RustMage checkouts and shared libraries are never overwritten.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REVISION='046b25ef14671753a2eea6336bc1ce974b298253'
ENGINE=Path(os.environ.get('RUSTMAGE_ROOT',ROOT/'.local/rustmage')).resolve()
SNAPSHOT=ROOT/'.local/direct-standard/rustmage-046b25ef1467'
TARGET=ROOT/'.local/direct-standard/target'


def main():
    if not (SNAPSHOT/'Cargo.toml').is_file():
        if SNAPSHOT.exists():raise FileExistsError('Incomplete pinned source directory: '+str(SNAPSHOT))
        SNAPSHOT.mkdir(parents=True)
        archive=subprocess.Popen(['git','-C',str(ENGINE),'archive',REVISION],stdout=subprocess.PIPE)
        unpack=subprocess.run(['tar','-x','-C',str(SNAPSHOT)],stdin=archive.stdout)
        archive.stdout.close()
        if archive.wait() or unpack.returncode:raise RuntimeError('Failed to export pinned RustMage source')
        (SNAPSHOT/'.direct-standard-revision').write_text(REVISION+'\n')
    elif (SNAPSHOT/'.direct-standard-revision').is_file():
        if (SNAPSHOT/'.direct-standard-revision').read_text().strip()!=REVISION:
            raise ValueError('Pinned RustMage revision marker differs')
    else:
        raise ValueError('Existing source snapshot has no revision marker; refusing to overwrite')
    env=dict(os.environ,CARGO_TARGET_DIR=str(TARGET))
    subprocess.run([sys.executable,str(ENGINE/'scripts/resource_guard.py'),'--','cargo','build',
        '--release','--offline','--locked','--manifest-path',str(ROOT/'native/direct-standard/Cargo.toml')],cwd=ROOT,env=env,check=True)
    library=TARGET/'release/libpatchwork_direct_standard.so'
    digest=hashlib.sha256(library.read_bytes()).hexdigest()
    print(json.dumps(dict(revision=REVISION,library=str(library),sha256=digest)))

if __name__=='__main__':main()
