"""Combine validated, matching-fixture CPU benchmark runs into a portable report."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--run', action='append', required=True, help='LABEL=JSON_PATH')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unmeasured-backend-log', type=Path)
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text())
    if not validation['passed']:
        raise ValueError('Numerical validation failed')
    fixture_path = args.fixture / 'manifest.json'
    fixture = json.loads(fixture_path.read_text())
    assert validation['fixture_sha256'] == digest(fixture_path)
    runs, indexed = {}, {}
    for item in args.run:
        label, path = item.split('=', 1)
        if label in runs:
            raise ValueError(f'Duplicate run: {label}')
        run = json.loads(Path(path).read_text())
        if run['framework'] == 'tensorflow':
            assert run['fixture_sha256'] == digest(fixture_path)
        else:
            assert run['fixture_binary_sha256'] == fixture['binary_sha256']
        rows = {}
        for row in run['rows']:
            times = row['wall_seconds']
            assert times and all(math.isfinite(t) and t > 0 for t in times)
            assert math.isclose(row['median_ms'], statistics.median(times) * 1000)
            key = row['fixture'] + '/' + row['operation']
            assert key not in rows
            rows[key] = row
        indexed[label] = rows
        runs[label] = run
    baseline = indexed[args.baseline]
    assert all(set(rows) == set(baseline) for rows in indexed.values())
    comparisons = []
    for key, reference in baseline.items():
        measurements = {}
        for label, rows in indexed.items():
            row = rows[key]
            measurements[label] = dict(
                median_ms=row['median_ms'],
                min_trial_ms=min(row['wall_seconds']) * 1000,
                max_trial_ms=max(row['wall_seconds']) * 1000,
                median_cpu_ms=statistics.median(row['cpu_seconds']) * 1000,
                speedup_vs_baseline=reference['median_ms'] / row['median_ms'],
            )
        comparisons.append(dict(case=key, measurements=measurements))
        print(key, ' | '.join(f'{k}: {v["median_ms"]:.3f} ms ({v["speedup_vs_baseline"]:.2f}x)'
                             for k, v in measurements.items()))
    root = Path(__file__).resolve().parents[2]
    source_paths = list(Path(__file__).parent.glob('*.py')) + list(Path(__file__).parent.glob('src/*.rs'))
    source_paths += [Path(__file__).parent / name for name in ('Cargo.toml', 'Cargo.lock')]
    source_paths += [root / 'src/patchwork_automaton/neural' / name
                     for name in ('model.py', 'ppo.py', 'features.py', 'batching.py')]
    output = dict(
        schema=1, date=datetime.now(timezone.utc).date().isoformat(), baseline=args.baseline,
        scope='CPU model kernels on preloaded real examples. Not an end-to-end training-session benchmark.',
        machine=dict(platform=platform.platform(), python=platform.python_version(),
                     cpu=subprocess.check_output(['lscpu'], text=True),
                     rustc=subprocess.check_output(['rustc', '--version'], text=True).strip()),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        source_sha256={str(p.relative_to(root)): digest(p) for p in sorted(source_paths)},
        fixture={k: v for k, v in fixture.items() if k != 'arrays'},
        validation=validation, runs=runs, comparisons=comparisons,
    )
    if args.unmeasured_backend_log:
        output['unmeasured_backend_build_log'] = args.unmeasured_backend_log.read_text()
    args.output.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
