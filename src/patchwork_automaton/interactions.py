"""Server-advertised interaction descriptors; no card rules or policy."""
from dataclasses import dataclass, field
from collections.abc import Mapping
from enum import IntEnum, IntFlag

from .properties import freeze, indexed


class PromptCode(IntFlag):
    OK = 1
    CANCEL = 2
    YES = 4
    NO = 8
    BLACK = 16
    BLUE = 32
    GREEN = 64
    RED = 128
    WHITE = 256
    COLORLESS = 512
    THREE_PILES = 4096
    PILE = 8192
    NUMBER = 16384
    DAMAGE_OK = 131072
    PLAYER = 524288
    NAME = 2097152
    WISH = 16777216


class SelectionSource(IntEnum):
    CARD_TYPES = 3  # low two bits signal that high bits are card type flags
    ARTIFACT_NAME = 16
    COUNTER_TYPE = 32
    CREATURE_NAME = 128
    NONCREATURE_NONLAND_NAME = 512
    BASIC_LAND = 2048
    LAND_TYPE = 4096
    PLANESWALKER_TYPE = 8192
    CREATURE_TYPE = 32768
    NONBASIC_NAME = 65536
    NONLAND_NONARTIFACT_NAME = 131072
    LAND_NAME = 524288
    NONLAND_NAME = 1048576
    CARD_NAME = 2097152
    NONBASIC_LAND_NAME = 4194304
    NONBASIC_LAND_TYPE = 8388608


@dataclass(frozen=True)
class NamedOption:
    value: int
    label: str


@dataclass(frozen=True)
class Selector:
    id: int
    kind: str
    code: int
    minimum: int | None = None
    maximum: int | None = None
    initial: int | None = None
    source: int | None = None
    options: tuple[NamedOption, ...] = ()
    except_values: tuple[int, ...] = ()
    invert: bool = False
    catalog_required: bool = False
    raw: bytes = field(default=b'', repr=False)


@dataclass(frozen=True)
class TargetGroup:
    index: int
    related_mode: int
    label: str
    minimum: int
    maximum: int
    legal_targets: tuple[int, ...]
    deferred: bool = False
    distributed_total: int | None = None
    distribution_minimum: int = 1
    requirements: tuple[str, ...] = ()
    attributes: Mapping = field(default_factory=lambda: freeze({}), repr=False)


@dataclass(frozen=True)
class ModeOption:
    index: int
    label: str
    property_id: int
    maximum: int
    weight: int = 1


@dataclass(frozen=True)
class ModeSpec:
    minimum: int
    maximum: int
    options: tuple[ModeOption, ...]
    fixed_mapping: str | None = None


@dataclass(frozen=True)
class DamageTarget:
    id: int
    initial: int
    minimum: int
    maximum: int


@dataclass(frozen=True)
class DamageAssignment:
    source_id: int
    total: int
    targets: tuple[DamageTarget, ...]
    flags: int = 0


@dataclass(frozen=True)
class Ordering:
    source_id: int
    targets: tuple[int, ...]


@dataclass(frozen=True)
class ManaEntry:
    player: int
    id: int
    amount: int
    color: int
    attributes: Mapping = field(repr=False)

    @property
    def color_flag(self):
        return {0:PromptCode.COLORLESS, 1:PromptCode.WHITE, 2:PromptCode.BLUE,
                3:PromptCode.BLACK, 4:PromptCode.RED, 5:PromptCode.GREEN}.get(self.color)


@dataclass(frozen=True)
class GameObject:
    id: int
    zone: int | None
    controller: int | None
    owner: int | None
    texture: int | None
    counters: Mapping
    properties: Mapping = field(repr=False)
    facets: tuple[int, ...] = ()


REQUIREMENTS = (
    'ALL_TARGETS_MUST_SHARE_CREATURE_TYPE', 'ACTION_TARGETS_MUST_SHARE_CARD_TYPE',
    'TARGETS_MUST_HAVE_DIFFERENT_CONTROLLERS', 'TARGETS_MUST_HAVE_SAME_CONTROLLER',
    'TARGETTING_UPDATE_SUM_VALUES', 'TARGETTING_UPDATE_DIFFERING_VALUES',
    'TARGETING_UPDATE_SAME_VALUES', 'TARGETS_MUST_HAVE_EQUAL_TOUGHNESS',
    'TARGETS_MUST_HAVE_DIFFERENT_NAMES', 'ACTION_TARGETS_MUST_SHARE_PERMANENT_TYPE',
)


def target_groups(action, card):
    groups = []
    containers = [(0, action)]
    containers += [(202812160 + i, item) for i, item in indexed(action, 202812160)]
    containers += [(202790144 + i, item) for i, item in indexed(action, 202790144)]
    for related, container in containers:
        minimum, maximum = action.get('MIN_TARGETS', 0), action.get('MAX_TARGETS', 0)
        label = action.get('TARGET_NAME_STRING', '')
        for index, info in indexed(container, 202482432):
            minimum, maximum = info.get('MIN_TARGETS', minimum), info.get('MAX_TARGETS', maximum)
            label = info.get('TARGET_NAME_STRING', label)
            legal = dict(indexed(container, 202553856)).get(index)
            if not legal:
                legal = dict(indexed(card, 202553856)).get(index, {})
            targets = tuple(value & 0xffffffff for _, value in indexed(legal, 604983296))
            total = None
            if info.get('ACTION_DIVIDE_EFFECT_UP'):
                total = container.get('ACTION_DAMAGEAMOUNT', maximum)
                if total < 0:
                    total = None
            merged = dict(container)
            merged.update(info)
            groups.append(TargetGroup(index, related, label, min(minimum, maximum), maximum, targets,
                bool(container.get('ALTACTION_DONT_CHOOSE_TARGETS_NOW') or info.get('TARGETS_AT_RANDOM')),
                total, 1, tuple(key for key in REQUIREMENTS if merged.get(key)), freeze(merged)))
    return tuple(groups)


def mode_spec(action, card):
    labels = dict(indexed(action, 1142185472))
    availability = dict(indexed(action, 605465856))
    effects = dict(indexed(action, 202812160))
    if 'ACTION_MODAL_MAX_CHOICES' not in action or not labels and not availability and not effects:
        return None
    minimum, maximum = action.get('ACTION_MODAL_MIN_CHOICES', 0), action.get('ACTION_MODAL_MAX_CHOICES', 0)
    repeat = bool(action.get('ACTION_MODAL_REPEAT_CHOICES'))
    weights = dict(indexed(action, 605528576))
    options = []
    fixed = None
    group_name = card.get('ACTION_GROUP_HEADER_STRING') or action.get('ACTION_GROUP_HEADER_STRING', '')
    if group_name:
        # A menu of separate actions already encodes each permitted combination.
        fixed = ''.join(str(i) * (n * maximum if repeat else n) for i, n in sorted(availability.items()))
    for i in sorted(labels.keys() | availability.keys() | effects.keys()):
        count = availability.get(i, 1)
        weight = weights.get(i, 1)
        capacity = count * maximum if repeat else count
        options.append(ModeOption(i, labels.get(i, ''), 202812160 + i if i in effects else 0,
                                  capacity // weight if weight > 0 else 0, weight))
    if availability and not fixed:
        maximum = min(maximum, max((n * maximum for n in availability.values()), default=0)
                      if repeat else sum(bool(n) for n in availability.values()))
    return ModeSpec(minimum, maximum, tuple(options), fixed)


def prompt_selectors(prompt, player_ids=(), catalog=None):
    result = []
    for entry in prompt.get('selectors', ()):
        kind = entry.get('kind', 'unknown')
        code = entry.get('code', 0)
        options = tuple(NamedOption(*option) for option in entry.get('options', ()))
        required = False
        if kind == 'player':
            options = tuple(NamedOption(i, f'Player {i}') for i in range(len(player_ids)))
        elif kind == 'list':
            required = catalog is None
            if catalog is not None:
                options = catalog.options(entry['source'])
                required = not bool(options)
            exclusions = entry.get('except_values', ())
            if exclusions:
                options = tuple(o for o in options if (o.value in exclusions) == entry.get('invert', False))
        elif kind == 'wish':
            required = catalog is None or not catalog.definitions
            if catalog is not None:
                options = catalog.wish_options()
        result.append(Selector(entry.get('id', len(result)), kind, code,
            entry.get('minimum'), entry.get('maximum'), entry.get('initial'), entry.get('source'),
            options, tuple(entry.get('except_values', ())), entry.get('invert', False), required,
            bytes.fromhex(entry.get('bytes', ''))))
    return tuple(result)
