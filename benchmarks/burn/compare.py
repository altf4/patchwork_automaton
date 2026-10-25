"""Fail closed on numerical disagreement in predictions, gradients and Adam states."""
import argparse
import json
from pathlib import Path
import numpy as np


def read_archive(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    data = (root / 'arrays.bin').read_bytes()
    arrays = {key: np.frombuffer(data, '<f4', count=value['elements'], offset=value['offset']).reshape(value['shape'])
              for key, value in manifest['arrays'].items()}
    return manifest, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--burn', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ref_meta, ref = read_archive(args.reference)
    burn_meta, burn = read_archive(args.burn)
    if set(ref) != set(burn):
        raise ValueError(f'Array keys differ: {set(ref) ^ set(burn)}')
    rows = {}
    failures = []
    for key, expected in ref.items():
        actual = burn[key]
        if actual.shape != expected.shape:
            raise ValueError(f'Shape differs for {key}: {actual.shape} != {expected.shape}')
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError(f'Nonfinite result: {key}')
        error = actual.astype(np.float64) - expected.astype(np.float64)
        # Explicit tolerance is set before inspecting results; report absolute
        # errors and relative L2 too, including tensors with near-zero elements.
        okay = bool(np.allclose(actual, expected, atol=3e-5, rtol=5e-4))
        rows[key] = dict(max_abs=float(np.max(np.abs(error))),
                         relative_l2=float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-12)),
                         passed=okay)
        if not okay: failures.append(key)
    losses = {}
    for key, expected in ref_meta['losses'].items():
        actual = burn_meta['losses'][key]
        losses[key] = dict(reference=expected, burn=actual)
        if not all(np.isclose(actual[k], v, atol=3e-5, rtol=5e-4) for k, v in expected.items()):
            failures.append(key + '/loss-or-norm')
    result = dict(passed=not failures, tolerance=dict(atol=3e-5, rtol=5e-4),
                  fixture_sha256=ref_meta['fixture_sha256'],
                  arrays=len(rows), failures=failures, losses=losses, comparisons=rows)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    worst = sorted(rows.items(), key=lambda item: -item[1]['max_abs'])[:8]
    print(json.dumps(dict(passed=result['passed'], failures=failures, worst=worst, losses=losses), indent=2))
    if failures: raise SystemExit(1)


if __name__ == '__main__':
    main()
