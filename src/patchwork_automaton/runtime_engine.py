"""Separate validated runtime rules fixes from the saved feature ABI."""
import json
from pathlib import Path

from .backends.rustmage import RUSTMAGE_REVISION, OBSERVATION_SCHEMA


def runtime_pin():
    path = Path(__file__).with_name('data') / 'runtime-engine.json'
    if not path.exists():
        return {}
    pin = json.loads(path.read_text())
    if (pin.get('feature_revision') != RUSTMAGE_REVISION or
            pin.get('observation_schema') != OBSERVATION_SCHEMA or
            not isinstance(pin.get('runtime_revision'), str) or
            len(pin['runtime_revision']) != 40 or
            any(c not in '0123456789abcdef' for c in pin['runtime_revision'])):
        raise ValueError('Runtime engine pin does not preserve the feature ABI')
    for key in ('feature_catalogue_version', 'runtime_catalogue_version'):
        if (not isinstance(pin.get(key), str) or len(pin[key]) != 64 or
                any(c not in '0123456789abcdef' for c in pin[key])):
            raise ValueError('Runtime engine pin is missing explicit catalogue compatibility')
    return pin


RUNTIME_PIN = runtime_pin()
RUNTIME_REVISION = RUNTIME_PIN.get('runtime_revision', RUSTMAGE_REVISION)


def runtime_catalogue_version(feature_manifest):
    if not RUNTIME_PIN:
        return feature_manifest.get('catalogue_version')
    if RUNTIME_PIN['feature_catalogue_version'] != feature_manifest.get('catalogue_version'):
        raise ValueError('Runtime compatibility does not match saved feature catalogue')
    return RUNTIME_PIN['runtime_catalogue_version']
