"""Build the guarded native replay helper and verify/timestamp captured requests.

uv run python scripts/profile-rustmage-native.py --traces ... --output ...
Links existing release rlibs; never modifies or rebuilds the engine checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout',type=Path,default=Path('.local/rustmage'))
    parser.add_argument('--traces',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError('Positive repeat count required')
    root = args.checkout.resolve()
    deps = root/'target/release/deps'
    engine = root/'target/release/librustmage.rlib'
    shared = root/'target/release/librustmage.so'
    serde = list(deps.glob('libserde_json-*.rlib'))
    if len(serde)!=1 or not engine.is_file():
        raise RuntimeError('Expected one serde_json release rlib and the pinned engine rlib; build using scripts/rustmage-local.py')
    source = Path(__file__).with_suffix('.rs').resolve()
    traces = sorted(args.traces.resolve().glob('game-*.jsonl'))
    if not traces:
        raise ValueError('No native request traces found')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    binary = args.output.parent/'profile-rustmage-native'
    before = sha(shared)
    command = [sys.executable,str(root/'scripts/resource_guard.py'),'--',
        'rustc','--edition=2024','-O',str(source),'--extern',f'rustmage={engine}',
        '--extern',f'serde_json={serde[0]}','-L',f'dependency={deps}',
        '-o',str(binary.resolve())]
    log = args.output.with_suffix('.build.log')
    with log.open('w') as stream:
        subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
    runs = [json.loads(subprocess.check_output([str(binary.resolve()),*map(str,traces)],text=True))
            for _ in range(args.repeats)]
    if sha(shared)!=before:
        raise RuntimeError('Engine shared library changed during profiling')
    counts = runs[0]['stages']
    for run in runs:
        assert run['requests_verified']==runs[0]['requests_verified']
        assert run['response_bytes_verified']==runs[0]['response_bytes_verified']
        assert {k:v['calls'] for k,v in run['stages'].items()}=={k:v['calls'] for k,v in counts.items()}
    mean = {key:dict(calls=value['calls'],seconds=sum(r['stages'][key]['seconds'] for r in runs)/len(runs))
            for key,value in counts.items()}
    result = dict(runs=runs,mean_stages=mean,engine_shared_sha256=before,
        engine_rlib_sha256=sha(engine),serde_rlib_sha256=sha(serde[0]),
        replay_source_sha256=sha(source),replay_binary_sha256=sha(binary),
        build_command=command,build_log=str(log),
        rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),
        revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),games=runs[0]['games'],
        requests_verified_per_repeat=runs[0]['requests_verified'],repeats=len(runs))),flush=True)


if __name__ == '__main__':
    main()
