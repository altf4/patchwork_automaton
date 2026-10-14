"""Immutable observations. Raw attributes preserve unmodeled card mechanics."""
from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType
import uuid

from .game import DeckItem


@dataclass(frozen=True)
class CollectionItem:
    catalog_id: int
    permission: int
    quantity: int

    def encode(self):
        import struct
        return struct.pack('<iii', self.catalog_id, self.permission, self.quantity)


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class Deck:
    id: int
    name: str
    description: str
    cards: tuple[DeckItem, ...]

    @property
    def format(self) -> str:
        parts = self.description.split('?')
        if len(parts) < 2 or parts[0] != '1':
            raise ValueError('Deck metadata has no playable format; verify the entry and specify format explicitly')
        return parts[1]


@dataclass(frozen=True)
class DeckList:
    response_code: int
    decks: tuple[Deck, ...]


@dataclass(frozen=True)
class Match:
    id: int
    token: uuid.UUID
    state: int
    games: tuple[int, ...]
    metadata: Mapping = field(repr=False)

    @classmethod
    def from_wire(cls, event):
        return cls(event['event_id'], event['token'], event['state'],
                   tuple(g['game_id'] for g in event['games']), freeze(event))


@dataclass(frozen=True)
class DecisionID:
    session: uuid.UUID
    game_id: int
    revision: int
    timestamp: int


@dataclass(frozen=True)
class CardAction:
    card_id: int
    action_id: int
    label: str
    attributes: Mapping = field(repr=False)


@dataclass(frozen=True)
class Observation:
    decision_id: DecisionID | None
    revision: int
    prompt: Mapping | None
    cards: Mapping
    players: tuple
    actions: tuple[CardAction, ...]
    status: Mapping
    unknown_elements: tuple
    raw_state: bytes = field(repr=False)


@dataclass(frozen=True)
class GameMessage:
    opcode: int
    name: str
    payload: bytes = field(repr=False)


@dataclass(frozen=True)
class Submission:
    decision_id: DecisionID
    # Transport write completed. Neither acknowledgement nor a new state
    # proves that the server accepted the game action.
    status: str = 'sent'
