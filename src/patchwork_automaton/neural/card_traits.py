"""Fixed-width, lossy card traits exported from evaluated typed RustMage rules.

Strings are identity lookups only. No Oracle text or language model participates
in feature generation or inference. Unknown identities retain a masked row.
"""
from importlib.resources import files
import hashlib
import json
from types import MappingProxyType

from ..backends.rustmage import OBSERVATION_SCHEMA, RUSTMAGE_REVISION
from ..semantics import CardSemantics, Coverage


def _validate_manifest(manifest):
    if (manifest.get('rustmage_revision') != RUSTMAGE_REVISION
            or manifest.get('observation_schema') != OBSERVATION_SCHEMA):
        raise ValueError('Card traits do not match the pinned RustMage revision/schema; '
            'regenerate with python3 scripts/export-rustmage-card-traits.py')


def _validate_content(data):
    payload = dict(feature_names=data['feature_names'], aliases=data['aliases'], unknown_traits=data['unknown_traits'],
        cards=[(c['identity'], c['features'], c['mana_value']) for c in data['cards']])
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if digest != data['manifest'].get('traits_sha256'):
        raise ValueError('Card trait contents do not match their manifest; '
            'regenerate with python3 scripts/export-rustmage-card-traits.py')


_DATA = json.loads(files('patchwork_automaton').joinpath('data/rustmage-card-traits.json').read_text())
MANIFEST = MappingProxyType(_DATA['manifest'])
_validate_manifest(MANIFEST)
_validate_content(_DATA)
FEATURE_NAMES = tuple(_DATA['feature_names'])
TRAIT_WIDTH = len(FEATURE_NAMES)
CARDS = tuple(c['identity'] for c in _DATA['cards'])
ALIASES = MappingProxyType(dict(_DATA['aliases']))
TRAITS = MappingProxyType({c['identity']: tuple(c['features']) for c in _DATA['cards']})
UNKNOWN_TRAITS = tuple(_DATA['unknown_traits'])
FACTS = MappingProxyType({c['identity']: MappingProxyType(dict(mana_value=c['mana_value'])) for c in _DATA['cards']})
_INDEX = {name: i + 1 for i, name in enumerate(CARDS)}
REGISTRY = MappingProxyType({name: CardSemantics(_INDEX[canonical], Coverage.PARTIAL)
    for name, canonical in ALIASES.items()})
registry = REGISTRY


def index_for(identity):
    return _INDEX.get(ALIASES.get(identity, identity), 0)


def traits_for(identity):
    return TRAITS.get(ALIASES.get(identity, identity), UNKNOWN_TRAITS)


if len(CARDS) != len(set(CARDS)) or len(FEATURE_NAMES) != len(set(FEATURE_NAMES)):
    raise ValueError('Duplicate card trait identity or feature')
if any(len(v) != TRAIT_WIDTH for v in (*TRAITS.values(), UNKNOWN_TRAITS)):
    raise ValueError('Card trait width mismatch')
if any(c not in TRAITS for c in ALIASES.values()):
    raise ValueError('Unknown card trait alias target')
del _DATA
