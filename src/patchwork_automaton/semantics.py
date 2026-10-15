"""Opt-in numerical card/ability annotations, not a rules or legality engine.

Enum IDs and encoded column order are a versioned training-data contract.
Names are registry keys only; no freeform text is emitted by the encoder.
"""
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from types import MappingProxyType

SCHEMA_VERSION = 2


class Coverage(IntEnum):
    UNKNOWN = 0
    PARTIAL = 1
    COMPLETE = 2


class Keyword(IntEnum):
    FLYING = 1
    REACH = 2
    FIRST_STRIKE = 3
    DOUBLE_STRIKE = 4
    TRAMPLE = 5
    MENACE = 6
    VIGILANCE = 7
    HASTE = 8
    DEATHTOUCH = 9
    LIFELINK = 10
    INDESTRUCTIBLE = 11
    HEXPROOF = 12
    DEFENDER = 13
    INFECT = 14
    PROWESS = 15


class Color(IntEnum):
    UNKNOWN = 0
    WHITE = 1
    BLUE = 2
    BLACK = 3
    RED = 4
    GREEN = 5
    COLORLESS = 6
    GENERIC = 7  # Cost requirement only; not a kind of produced mana.


class AbilityKind(IntEnum):
    UNKNOWN = 0
    SPELL = 1
    ACTIVATED = 2
    TRIGGERED = 3
    STATIC = 4
    MANA = 5


class EffectKind(IntEnum):
    UNKNOWN = 0
    DAMAGE = 1
    DRAW = 2
    DISCARD = 3
    ADD_MANA = 4
    GAIN_LIFE = 5
    LOSE_LIFE = 6
    DESTROY = 7
    EXILE = 8
    COUNTER_SPELL = 9
    MOVE_ZONE = 10
    CREATE_TOKEN = 11
    MODIFY_STATS = 12
    ADD_COUNTER = 13
    REMOVE_COUNTER = 14
    GRANT_KEYWORD = 15
    REMOVE_KEYWORD = 16
    TAP = 17
    UNTAP = 18
    SCRY = 19
    MILL = 20
    SEARCH = 21
    CHANGE_CONTROL = 22
    PREVENT_DAMAGE = 23
    COPY = 24
    CANT_GAIN_LIFE = 25
    DAMAGE_CANT_BE_PREVENTED = 26
    SURVEIL = 27
    REVEAL = 28
    SHUFFLE = 29
    ENTER_TAPPED = 30
    REPLACE_ZONE_CHANGE = 31


class CostKind(IntEnum):
    UNKNOWN = 0
    MANA = 1
    TAP_SOURCE = 2
    SACRIFICE = 3
    DISCARD = 4
    PAY_LIFE = 5
    REMOVE_COUNTER = 6
    EXILE = 7


class QuantityKind(IntEnum):
    UNKNOWN = 0
    CONSTANT = 1
    X = 2
    SOURCE_POWER = 3
    MATCHING_COUNT = 4
    LANDFALL_DAMAGE = 5  # One, or three if a land entered under your control this turn.
    PREVENTED_DAMAGE = 6


class Recipient(IntEnum):
    UNKNOWN = 0
    SELF_PLAYER = 1
    OPPONENT = 2
    SOURCE = 3
    TARGET = 4
    EACH_MATCHING = 5
    SPELL_CASTER = 6
    ACTIVE_PLAYER = 7
    DEFENDING_PLAYER = 8
    TARGET_CONTROLLER = 9
    YOUR_PERMANENTS = 10
    CHOSEN_SOURCE_CONTROLLER = 11


class ObjectFilter(IntFlag):
    UNKNOWN = 0
    PLAYER = 1
    CREATURE = 2
    PLANESWALKER = 4
    BATTLE = 8
    ARTIFACT = 16
    ENCHANTMENT = 32
    LAND = 64
    SPELL = 128
    CARD = 256
    ANY_TARGET = PLAYER | CREATURE | PLANESWALKER | BATTLE


class Trigger(IntEnum):
    UNKNOWN = 0
    NONE = 1
    SOURCE_ENTERS = 2
    SOURCE_ATTACKS = 3
    SOURCE_DIES = 4
    LAND_ENTERS = 5
    SPELL_CAST = 6
    UPKEEP = 7


class Condition(IntEnum):
    UNKNOWN = 0
    NONE = 1
    CONTROL_ARTIFACT = 2
    OPPONENT_LOST_LIFE = 3
    COUNT_THRESHOLD = 4
    SPELL_MANA_VALUE_AT_MOST = 5
    NO_MANA_SPENT = 6
    YOU_CAST_NONCREATURE = 7
    TARGET_WAS_NONBASIC = 8
    TARGET_DIES_THIS_TURN = 9
    REVEALED_CARD_IS_LAND = 10


class Duration(IntEnum):
    UNKNOWN = 0
    IMMEDIATE = 1
    END_OF_TURN = 2
    WHILE_SOURCE_REMAINS = 3
    INDEFINITE = 4


class Zone(IntEnum):
    UNKNOWN = 0
    NONE = 1
    HAND = 2
    BATTLEFIELD = 3
    GRAVEYARD = 4
    EXILE = 5
    LIBRARY = 6


class RuleTag(IntEnum):
    """Reviewed mechanical qualifiers, never freeform text or live legality."""
    SACRIFICE_SOURCE = 1
    SEARCH_MOUNTAIN_OR_PLAINS = 2
    SEARCH_SWAMP_OR_MOUNTAIN = 3
    FROM_YOUR_LIBRARY = 4
    SEARCH_RESULT = 5
    TOP_LIBRARY_CARD = 6
    DEFENDING_PLAYERS_CARD = 7
    SAME_CONTROLLER_AS_FIRST_TARGET = 8
    SECOND_TARGET = 9
    SEVEN_CARDS_IN_YOUR_GRAVEYARD = 10
    ACTIVATION_RESTRICTION = 11
    OPTIONAL_PAY_TWO_LIFE_OR_ENTER_TAPPED = 12
    ENTER_TAPPED_IF_THREE_OTHER_LANDS = 13
    ALTERNATIVE_CAST_COST = 14
    CONDITION_THIS_TURN = 15
    POWER_AND_TOUGHNESS = 16
    ALL_GRAVEYARDS = 17
    CARD_OR_TOKEN_TO_ANY_GRAVEYARD = 18
    NEXT_DAMAGE_FROM_CHOSEN_SOURCE = 19
    ONLY_IF_DAMAGE_PREVENTED = 20
    DELAYED_TRIGGER = 21
    THAT_CREATURES_CONTROLLER = 22
    PREVIOUS_TARGET_CONTROLLER = 23
    FUSE_FROM_HAND = 24
    CHOOSE_ONE_MODE = 25
    IF_REVEALED_CARD_IS_LAND = 26
    IF_TARGET_WAS_NONBASIC = 27
    IF_TARGET_DIES_THIS_TURN = 28


@dataclass(frozen=True)
class Quantity:
    kind: QuantityKind = QuantityKind.UNKNOWN
    value: int = 0

    def __post_init__(self):
        if type(self.value) is not int:
            raise ValueError('Quantity value must be an integer')


@dataclass(frozen=True)
class Cost:
    kind: CostKind
    quantity: Quantity = Quantity()
    color: Color = Color.UNKNOWN
    object_filter: ObjectFilter = ObjectFilter.UNKNOWN
    counter_id: int = 0


@dataclass(frozen=True)
class Effect:
    kind: EffectKind
    quantity: Quantity = Quantity()
    recipient: Recipient = Recipient.UNKNOWN
    object_filter: ObjectFilter = ObjectFilter.UNKNOWN
    color: Color = Color.UNKNOWN
    duration: Duration = Duration.UNKNOWN
    destination: Zone = Zone.UNKNOWN
    keyword: Keyword | None = None
    counter_id: int = 0
    # References an application-defined semantic definition, e.g. token template.
    definition_id: int = 0
    tags: tuple[RuleTag, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'tags', tuple(self.tags))
        if self.kind == EffectKind.ADD_MANA and self.color == Color.GENERIC:
            raise ValueError('Generic mana is a cost requirement, not produced mana')


@dataclass(frozen=True)
class Ability:
    # Stable registry ID, never a live backend action UUID.
    id: int
    kind: AbilityKind
    effects: tuple[Effect, ...] = ()
    costs: tuple[Cost, ...] = ()
    coverage: Coverage = Coverage.UNKNOWN
    trigger: Trigger = Trigger.UNKNOWN
    condition: Condition = Condition.UNKNOWN
    condition_threshold: int = 0
    # Mode 0 means unmodal. Separate records retain distinct modes/abilities.
    mode: int = 0
    tags: tuple[RuleTag, ...] = ()

    def __post_init__(self):
        if type(self.id) is not int or self.id <= 0:
            raise ValueError('Ability identity must be a positive integer')
        object.__setattr__(self, 'effects', tuple(self.effects))
        object.__setattr__(self, 'costs', tuple(self.costs))
        object.__setattr__(self, 'tags', tuple(self.tags))


@dataclass(frozen=True)
class CardSemantics:
    id: int
    coverage: Coverage = Coverage.UNKNOWN
    keywords: tuple[Keyword, ...] = ()
    keyword_coverage: Coverage = Coverage.UNKNOWN
    abilities: tuple[Ability, ...] = ()

    def __post_init__(self):
        if type(self.id) is not int or self.id < 0:
            raise ValueError('Card identity must be a nonnegative integer')
        object.__setattr__(self, 'keywords', tuple(self.keywords))
        object.__setattr__(self, 'abilities', tuple(self.abilities))
        ids = [ability.id for ability in self.abilities]
        if len(ids) != len(set(ids)):
            raise ValueError('Ability identities must be unique within a card')


UNKNOWN_CARD = CardSemantics(0)

# Ordered tensor columns. Costs/effects include the ability row to preserve
# grouping; effect order within an ability is preserved by row order.
ABILITY_COLUMNS = ('id', 'kind', 'coverage', 'trigger', 'condition', 'threshold', 'mode')
COST_COLUMNS = ('ability_row', 'kind', 'quantity_kind', 'quantity', 'color', 'filter', 'counter_id')
EFFECT_COLUMNS = ('ability_row', 'kind', 'quantity_kind', 'quantity', 'recipient',
                  'filter', 'color', 'duration', 'destination', 'keyword', 'counter_id', 'definition_id')


def encode_card(card: CardSemantics):
    """Return integer-only, unpadded features suitable for a tensor collator.

    These describe annotated base semantics, not current engine state or legality.
    Empty arrays need their declared column widths when converted to tensors.
    """
    return {
        'schema_version': SCHEMA_VERSION,
        'card_id': card.id,
        'coverage': int(card.coverage),
        'keyword_coverage': int(card.keyword_coverage),
        'keywords': tuple(int(k in card.keywords) for k in Keyword),
        'abilities': tuple((a.id, int(a.kind), int(a.coverage), int(a.trigger),
                            int(a.condition), a.condition_threshold, a.mode) for a in card.abilities),
        'costs': tuple((i, int(c.kind), int(c.quantity.kind), c.quantity.value,
                       int(c.color), int(c.object_filter), c.counter_id)
                      for i, a in enumerate(card.abilities) for c in a.costs),
        'effects': tuple((i, int(e.kind), int(e.quantity.kind), e.quantity.value,
                         int(e.recipient), int(e.object_filter), int(e.color),
                         int(e.duration), int(e.destination), int(e.keyword or 0),
                         e.counter_id, e.definition_id)
                        for i, a in enumerate(card.abilities) for e in a.effects),
        'ability_tags': tuple(tuple(int(tag in a.tags) for tag in RuleTag) for a in card.abilities),
        'effect_tags': tuple(tuple(int(tag in e.tags) for tag in RuleTag)
                             for a in card.abilities for e in a.effects),
    }


ONE = Quantity(QuantityKind.CONSTANT, 1)
INITIAL_CARDS = MappingProxyType({
    'Mountain': CardSemantics(1, Coverage.COMPLETE, keyword_coverage=Coverage.COMPLETE,
        abilities=(Ability(1, AbilityKind.MANA,
            effects=(Effect(EffectKind.ADD_MANA, ONE, Recipient.SELF_PLAYER,
                            color=Color.RED, duration=Duration.IMMEDIATE),),
            costs=(Cost(CostKind.TAP_SOURCE, ONE),), coverage=Coverage.COMPLETE,
            trigger=Trigger.NONE, condition=Condition.NONE),)),
    'Lightning Bolt': CardSemantics(2, Coverage.COMPLETE, keyword_coverage=Coverage.COMPLETE,
        abilities=(Ability(2, AbilityKind.SPELL,
            effects=(Effect(EffectKind.DAMAGE, Quantity(QuantityKind.CONSTANT, 3),
                            Recipient.TARGET, ObjectFilter.ANY_TARGET,
                            duration=Duration.IMMEDIATE),),
            costs=(Cost(CostKind.MANA, ONE, Color.RED),), coverage=Coverage.COMPLETE,
            trigger=Trigger.NONE, condition=Condition.NONE),)),
    'Goblin Piker': CardSemantics(3, Coverage.COMPLETE, keyword_coverage=Coverage.COMPLETE,
        abilities=(Ability(3, AbilityKind.SPELL,
            effects=(Effect(EffectKind.MOVE_ZONE, ONE, Recipient.SOURCE,
                            duration=Duration.IMMEDIATE, destination=Zone.BATTLEFIELD),),
            costs=(Cost(CostKind.MANA, ONE, Color.GENERIC), Cost(CostKind.MANA, ONE, Color.RED)),
            coverage=Coverage.COMPLETE, trigger=Trigger.NONE, condition=Condition.NONE),)),
    'Raging Goblin': CardSemantics(4, Coverage.COMPLETE, keywords=(Keyword.HASTE,),
        keyword_coverage=Coverage.COMPLETE,
        abilities=(Ability(4, AbilityKind.SPELL,
            effects=(Effect(EffectKind.MOVE_ZONE, ONE, Recipient.SOURCE,
                            duration=Duration.IMMEDIATE, destination=Zone.BATTLEFIELD),),
            costs=(Cost(CostKind.MANA, ONE, Color.RED),), coverage=Coverage.COMPLETE,
            trigger=Trigger.NONE, condition=Condition.NONE),)),
})


def lookup_visible(obj, registry=None):
    """Resolve an ObjectView's known identity only; never inspect hidden text/IDs.

    The caller must supply the actor's view, not merged observations from seats.
    Unknown cards remain playable by backend options but have unknown semantics.
    """
    if not obj.name or any(obj.properties.get(key) for key in ('hideInfo', 'faceDown', 'face_down')):
        return UNKNOWN_CARD
    if registry is None:
        from .burn_semantics import CARD_REGISTRY
        registry = CARD_REGISTRY
    return registry.get(obj.name, UNKNOWN_CARD)
