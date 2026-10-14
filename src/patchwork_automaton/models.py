"""Immutable observations. Raw attributes preserve unmodeled card mechanics."""
from dataclasses import dataclass, field
from collections.abc import Mapping
import uuid

from .game import DeckItem
from .properties import freeze
from .interactions import (AutoYield, Commander, Dungeon, DamageAssignment, GameObject, ManaEntry, ModeSpec,
                           Ordering, Selector, TargetGroup)


@dataclass(frozen=True)
class CollectionItem:
    catalog_id: int
    permission: int
    quantity: int

    def encode(self):
        import struct
        return struct.pack('<iii', self.catalog_id, self.permission, self.quantity)


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
    target_groups: tuple[TargetGroup, ...] = ()
    modes: ModeSpec | None = None
    cast_options: str = ''
    is_mana_ability: bool = False
    is_cast: bool = False
    attack_victim_id: int | None = None
    non_game: bool = False
    can_perform_locally: bool = False
    tap_locally: bool = False

    def legal_targets(self, observation, group_index, *, modes=None, chosen=()):
        from .actions import legal_targets_for
        return legal_targets_for(observation, self, group_index, modes=modes, chosen=chosen)

    def target_groups_for(self, modes=None):
        from .actions import selected_groups
        return selected_groups(self, modes)[1]


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
    selectors: tuple[Selector, ...] = ()
    objects: Mapping = field(default_factory=lambda: freeze({}))
    mana: tuple[ManaEntry, ...] = ()
    damage: tuple[DamageAssignment, ...] = ()
    ordering: tuple[Ordering, ...] = ()
    commanders: tuple[Commander, ...] = ()
    dungeons: tuple[Dungeon, ...] = ()
    auto_yields: tuple[AutoYield, ...] = ()

    @property
    def stack(self):
        return tuple(sorted((obj for obj in self.objects.values() if obj.zone == 1001728),
                     key=lambda obj: obj.properties.get('ORDERED_ZONE_INDEX', 0)))

    def actions_for(self, object_id):
        """Include advertised actions on split/adventure/room child objects."""
        pending, ids = [object_id], set()
        while pending:
            current = pending.pop()
            if current in ids:
                continue
            ids.add(current)
            obj = self.objects.get(current)
            if obj is not None:
                pending.extend(obj.facets)
        return tuple(action for action in self.actions if action.card_id in ids)


@dataclass(frozen=True)
class GameMessage:
    opcode: int
    name: str
    payload: bytes = field(repr=False)
    result: 'GameResult | None' = None


@dataclass(frozen=True)
class Submission:
    decision_id: DecisionID
    # Transport write completed. Neither acknowledgement nor a new state
    # proves that the server accepted the game action.
    status: str = 'sent'


@dataclass(frozen=True)
class ValidationError:
    catalog_id: int
    annotation: int
    code: int
    region: int
    card_specific: bool


@dataclass(frozen=True)
class SideboardResult:
    match_id: int
    accepted: bool
    errors: tuple[ValidationError, ...] = ()
    expired: bool = False


@dataclass(frozen=True)
class PlayerResult:
    code: int
    playing_time: int


@dataclass(frozen=True)
class GameResult:
    game_id: int
    moved_first: int
    status: int
    players: tuple[PlayerResult, ...]

    @property
    def winner_indices(self):
        return tuple(i for i,player in enumerate(self.players) if player.code == 2)
