"""Build the experimental typed Standard bridge with RustMage's resource guard.

The engine source is exported from the pinned Standard-compatible commit;
existing RustMage checkouts and shared libraries are never overwritten.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from patchwork_automaton.runtime_engine import RUNTIME_REVISION, runtime_catalogue_version
from patchwork_automaton.backends.rustmage import RUSTMAGE_REVISION as FEATURE_REVISION
REVISION=RUNTIME_REVISION
ENGINE=Path(os.environ.get('RUSTMAGE_ROOT',ROOT/'.local/rustmage')).resolve()
SNAPSHOT=ROOT/'.local/direct-standard'/f'rustmage-{REVISION[:12]}'
TARGET=ROOT/'.local/direct-standard'/f'target-{REVISION[:12]}-capped256'


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
    traits_path=ROOT/'src/patchwork_automaton/data/rustmage-card-traits.json'
    traits=json.loads(traits_path.read_text())['manifest']
    coverage=json.loads((SNAPSHOT/'data/standard/coverage.json').read_text())
    schema=int((SNAPSHOT/'data/observation-schema.txt').read_text())
    if (traits['rustmage_revision']!=FEATURE_REVISION or traits['observation_schema']!=schema
            or runtime_catalogue_version(traits)!=coverage['catalogue_version']):
        raise ValueError('Direct engine snapshot and exported card traits differ; update both before building')
    sys.path.insert(0,str(ROOT/'src'))
    from patchwork_automaton.rl.hartford_preflight import direct_sources_sha256, file_sha256
    source_hash=direct_sources_sha256(ROOT)
    library=TARGET/'release/libpatchwork_direct_standard.so'
    manifest=library.with_suffix('.build.json')
    previous=json.loads(manifest.read_text()) if manifest.is_file() else {}
    started=time.time_ns()
    env=dict(os.environ,CARGO_TARGET_DIR=str(TARGET))
    subprocess.run([sys.executable,str(ENGINE/'scripts/resource_guard.py'),'--','cargo','build',
        '--release','--offline','--locked','--manifest-path',str(ROOT/'native/direct-standard/Cargo.toml')],cwd=ROOT,env=env,check=True)
    digest=hashlib.sha256(library.read_bytes()).hexdigest()
    if source_hash!=direct_sources_sha256(ROOT):
        raise RuntimeError('Native bridge inputs changed during the build; rebuild a frozen source tree')
    if (library.stat().st_mtime_ns<started and
            (previous.get('direct_sources_sha256')!=source_hash or previous.get('sha256')!=digest)):
        raise RuntimeError('Build did not produce a current native library; refusing to certify an older artifact')
    # Persist evidence for admission of future experiments. Existing binaries
    # without this record remain usable by their original playback workflows.
    record=dict(revision=REVISION,observation_schema=schema,
        catalogue_version=coverage['catalogue_version'],engine_source=str(SNAPSHOT),
        library=str(library),sha256=digest,traits_file_sha256=file_sha256(traits_path),
        coverage_sha256=file_sha256(SNAPSHOT/'data/standard/coverage.json'),
        direct_sources_sha256=source_hash)
    temporary=manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    temporary.replace(manifest)
    print(json.dumps(record))

if __name__=='__main__':main()
