"""Explicit game controls, separate from responses to server decisions."""
from dataclasses import dataclass
from enum import IntEnum, IntFlag
import struct

from .protocol import string


class YieldControl(IntEnum):
    # Native FunctionKeyClickedMessage keys are toggles except DISABLE_ALL.
    DISABLE_ALL = 3
    THROUGH_THIS_TURN = 6
    AUTO_STACK_TRIGGERS = 7
    DISABLE_BLUFFING = 8
    THROUGH_OPPONENT_ACTIONS = 9


class YieldPreference(IntFlag):
    NORMAL_CARD_DRAW = 1
    ALL_COMBAT_DAMAGE = 2
    OTHERS_COMBAT_DAMAGE = 4
    REST_OF_ATTACK_PHASE_WHEN_NO_ATTACK = 8
    HIDE_NO_MANA_ABILITIES = 16
    OWN_MADNESS = 64
    OPPONENT_MADNESS = 128
    ALWAYS_USE_MADNESS = 256
    PROMPT_FOR_MANA_BURN = 512
    IGNORE_BUTCHER_WHEN_BLOCKING = 1024
    ALWAYS_BLUFF_MIRACLES = 4096


class Step(IntFlag):
    """Native GamePhase bit masks used by per-player stop preferences."""
    # Native mask is 1 << (GamePhase - Upkeep); untap has no stop.
    UPKEEP = 1
    DRAW = 2
    PRECOMBAT_MAIN = 4
    BEGIN_COMBAT = 8
    DECLARE_ATTACKERS = 16
    DECLARE_BLOCKERS = 32
    COMBAT_DAMAGE = 64
    END_COMBAT = 128
    POSTCOMBAT_MAIN = 256
    END = 512
    CLEANUP = 1024


@dataclass(frozen=True)
class PlayerStops:
    # Game-server player index, not the account user ID.
    player_id: int
    steps: int


@dataclass(frozen=True)
class ControlReceipt:
    name: str
    game_id: int
    revision: int
    status: str = 'sent'


def stop_payload(checksum, stops, yields, preferences):
    # The server compares this with its deployed string table. Do not invent it.
    if len(checksum) != 32 or any(c not in '0123456789abcdefABCDEF' for c in checksum):
        raise ValueError('Supply the deployed string-table checksum (32 hex characters)')
    if not 0 <= preferences <= 0xffffffff:
        raise ValueError('Yield preferences must be uint32')
    parts = [string(checksum, limit=32)]
    for entries in (tuple(stops), tuple(yields)):
        if len(entries) > 16 or len({e.player_id for e in entries}) != len(entries):
            raise ValueError('Supply at most one preference per player (maximum 16)')
        parts.append(struct.pack('<i', len(entries)))
        for entry in entries:
            if not 0 <= entry.player_id < 16 or not 0 <= entry.steps <= 0xffffffff:
                raise ValueError('Player indices must be 0..15 and step masks must be uint32')
            parts.append(struct.pack('<II', entry.player_id, entry.steps))
    return b''.join(parts) + struct.pack('<I', preferences)
