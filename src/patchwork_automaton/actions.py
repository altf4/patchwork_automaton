"""Encode caller-selected actions. No strategy or card-specific rules."""
from dataclasses import dataclass
import struct

from .protocol import ProtocolError, array
from .state import PROPERTY_NAMES

REVERSE = {name: int(key) for key, name in PROPERTY_NAMES.items()}


def action_entries(properties):
    for name, value in properties.items():
        code = REVERSE.get(name, -1)
        if name.startswith('attribute_'):
            code = int(name.removeprefix('attribute_'))
        if 202329600 <= code <= 202329855 and isinstance(value, dict):
            yield code, value


@dataclass(frozen=True)
class PromptChoice:
    code: int
    parameter: int = 0


@dataclass(frozen=True)
class CardChoice:
    card_id: int
    action_id: int
    targets: tuple[int, ...] = ()
    modes: bytes = b''
    extra_mana: int = 0
    flags: int = 1


def card_payload(card_id, action_id, timestamp, targets=(), *, modes=b'', extra_mana=0, flags=1):
    if len(targets) > 255 or len(modes) > 255:
        raise ProtocolError('Too many targets or modes')
    if not 0 <= extra_mana <= 255 or not 0 <= flags <= 255:
        raise ProtocolError('Action byte out of range')
    return (struct.pack('<ii', card_id, action_id) + array(modes)
            + struct.pack('<IBBBB', timestamp, len(targets), extra_mana, flags, 0)
            + struct.pack('<i', len(targets))
            + b''.join(struct.pack('<I', target) for target in targets))
