"""Numeric property identities, including indexed attributes omitted from enums."""
from collections.abc import Mapping
import json
from pathlib import Path
from types import MappingProxyType

PROPERTY_NAMES = json.loads(Path(__file__).with_name('properties.json').read_text())
PROPERTY_IDS = {name: int(key) for key, name in PROPERTY_NAMES.items()}


def property_id(name):
    return int(name[10:]) if name.startswith('attribute_') else PROPERTY_IDS.get(name, -1)


def get_property(properties, code, default=None):
    return properties.get(PROPERTY_NAMES.get(str(code), f'attribute_{code}'), default)


def indexed(properties, base, limit=255):
    """Yield sparse indexed properties in wire index order."""
    return tuple((property_id(key) - base, value) for key, value in sorted(
        properties.items(), key=lambda item: property_id(item[0]))
        if base <= property_id(key) < base + limit)


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value
