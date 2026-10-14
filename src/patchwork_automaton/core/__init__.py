"""Backend-neutral, decision-level gameplay contracts (no rules or policy)."""
from dataclasses import dataclass, field, fields
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from ..properties import freeze


class _Immutable:
    def __post_init__(self):
        for descriptor in fields(self):
            value=getattr(self,descriptor.name)
            if isinstance(value,(Mapping,list,tuple)):
                object.__setattr__(self,descriptor.name,freeze(value))


@dataclass(frozen=True)
class Handle(_Immutable):
    backend: str
    kind: str
    value: str


@dataclass(frozen=True)
class DecisionToken(_Immutable):
    backend: str
    session: str
    game: Handle
    actor: Handle
    generation: int


@dataclass(frozen=True)
class Option(_Immutable):
    id: str
    kind: str
    label: str
    constraints: Mapping = field(default_factory=lambda: freeze({}))


@dataclass(frozen=True)
class Decision(_Immutable):
    token: DecisionToken
    kind: str
    message: str
    options: tuple[Option, ...]
    native: Mapping = field(default_factory=lambda: freeze({}), repr=False)


@dataclass(frozen=True)
class Response(_Immutable):
    option: str
    value: object = None


@dataclass(frozen=True)
class ObjectView(_Immutable):
    id: Handle
    name: str | None
    zone: str
    controller: Handle | None = None
    properties: Mapping = field(default_factory=lambda: freeze({}), repr=False)


@dataclass(frozen=True)
class Outcome(_Immutable):
    # Local actor perspective; None means not yet known, never assume a draw.
    won: bool | None = None
    draw: bool | None = None
    native: Mapping = field(default_factory=lambda: freeze({}), repr=False)


@dataclass(frozen=True)
class PlayerView(_Immutable):
    id: Handle
    name: str
    life: int | None
    hand_count: int | None = None
    library_count: int | None = None
    counters: Mapping = field(default_factory=lambda: freeze({}))
    native: Mapping = field(default_factory=lambda: freeze({}), repr=False)


@dataclass(frozen=True)
class Observation(_Immutable):
    backend: str
    game: Handle
    actor: Handle
    revision: int
    players: tuple
    objects: Mapping
    decision: Decision | None
    terminal: bool = False
    outcome: Outcome | None = None
    native: Mapping = field(default_factory=lambda: freeze({}), repr=False)
    turn: int | None = None
    phase: str | None = None
    active_player: Handle | None = None
    priority_player: Handle | None = None


@dataclass(frozen=True)
class Capabilities(_Immutable):
    backend: str
    features: frozenset[str]

    def supports(self, feature):
        return feature in self.features


@dataclass(frozen=True)
class CardSpec:
    name: str
    quantity: int
    sideboard: bool = False
    set_code: str | None = None
    collector_number: str | None = None
    # MTGO collection identity is separate from XMage printing identity.
    mtgo_catalog_id: int | None = None
    mtgo_annotation: int = 0
    mtgo_permission: int = 0

    def __post_init__(self):
        if not self.name or type(self.quantity) is not int or self.quantity <= 0:
            raise ValueError('Card name and positive integer quantity are required')
        if bool(self.set_code) != bool(self.collector_number):
            raise ValueError('Provide both set code and collector number, or neither')


@dataclass(frozen=True)
class DeckSpec:
    cards: tuple[CardSpec, ...]
    name: str = 'Patchwork deck'

    def __post_init__(self):
        object.__setattr__(self,'cards',tuple(self.cards))
        if not self.cards:
            raise ValueError('Deck cannot be empty')


@runtime_checkable
class GameHandle(Protocol):
    capabilities: Capabilities

    async def wait_for_decision(self, *, timeout=None) -> Observation: ...
    async def submit(self, token: DecisionToken, response: Response): ...
    async def snapshot(self) -> Observation: ...
    async def concede(self): ...
