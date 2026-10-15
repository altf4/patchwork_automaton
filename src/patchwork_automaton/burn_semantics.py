"""Reviewed numerical identities for the pinned 60+15 Modern Burn deck.

Sources and limitations: data/burn-semantics-review.json. These are base-card
features, not executable rules. PARTIAL records preserve unresolved dependencies.
IDs 1–4 stay assigned to the original smoke-test cards; new IDs are append-only.
"""
from dataclasses import replace
from types import MappingProxyType
from .semantics import (
    INITIAL_CARDS, CardSemantics, Ability, AbilityKind as A, Coverage as V,
    Keyword as K, Effect, EffectKind as E, Cost, CostKind as C, Color,
    Quantity, QuantityKind as Q, Recipient as R, ObjectFilter as F,
    Trigger as T, Condition as D, Duration as U, Zone as Z, RuleTag as G,
)


def q(n): return Quantity(Q.CONSTANT, n)
def mana(n, color): return Cost(C.MANA, q(n), color)
def effect(kind, n=None, recipient=R.UNKNOWN, filter=F.UNKNOWN, **kw):
    return Effect(kind, q(n) if n is not None else Quantity(), recipient, filter, **kw)
def damage(n, recipient=R.TARGET, filter=F.ANY_TARGET, **kw):
    return effect(E.DAMAGE, n, recipient, filter, duration=U.IMMEDIATE, **kw)
def ability(kind, effects=(), costs=(), *, coverage=V.COMPLETE, **kw):
    return Ability(1, kind, tuple(effects), tuple(costs), coverage,
                   trigger=kw.pop('trigger', T.NONE), condition=kw.pop('condition', D.NONE), **kw)
def card(identity, abilities, *, keywords=(), coverage=V.COMPLETE):
    return CardSemantics(identity, coverage, tuple(keywords), V.COMPLETE,
                         tuple(replace(a, id=identity*100+i+1) for i,a in enumerate(abilities)))

RED = mana(1, Color.RED)
WHITE = mana(1, Color.WHITE)
GENERIC = mana(1, Color.GENERIC)
RR = (mana(2, Color.RED),)
RW = (RED, WHITE)
TAP = Cost(C.TAP_SOURCE, q(1))
SAC = Cost(C.SACRIFICE, q(1), object_filter=F.LAND)
LIFE = Cost(C.PAY_LIFE, q(1))
PLAYER_PW = F.PLAYER | F.PLANESWALKER
PERMANENTS = F.CREATURE | F.ARTIFACT | F.ENCHANTMENT | F.LAND | F.PLANESWALKER | F.BATTLE


def permanent(costs):
    return ability(A.SPELL, (effect(E.MOVE_ZONE, 1, R.SOURCE, destination=Z.BATTLEFIELD,
                                     duration=U.IMMEDIATE),), costs)
def mana_ability(color, costs=(TAP,), **kw):
    return ability(A.MANA, (effect(E.ADD_MANA, 1, R.SELF_PLAYER, color=color,
                                  duration=U.IMMEDIATE),), costs, **kw)
def dual(costs=(TAP,)):
    # Alternative mana abilities, each yielding one mana, never red AND white.
    return (mana_ability(Color.RED, costs), mana_ability(Color.WHITE, costs))
def fetch(identity, search_tag):
    return card(identity, (ability(A.ACTIVATED, (
        effect(E.SEARCH, 1, R.SELF_PLAYER, F.LAND, destination=Z.LIBRARY,
               tags=(search_tag, G.FROM_YOUR_LIBRARY)),
        effect(E.MOVE_ZONE, 1, R.SELF_PLAYER, F.LAND, destination=Z.BATTLEFIELD,
               tags=(G.SEARCH_RESULT,)),
        effect(E.SHUFFLE, recipient=R.SELF_PLAYER, destination=Z.LIBRARY)),
        (TAP, LIFE, SAC), tags=(G.SACRIFICE_SOURCE,), coverage=V.PARTIAL),), coverage=V.PARTIAL)


BURN_CARDS = MappingProxyType({
    'Arid Mesa': fetch(5, G.SEARCH_MOUNTAIN_OR_PLAINS),
    'Barbarian Ring': card(6, (
        ability(A.MANA, (effect(E.ADD_MANA, 1, R.SELF_PLAYER, color=Color.RED,
                                duration=U.IMMEDIATE), damage(1, R.SELF_PLAYER, F.PLAYER)), (TAP,)),
        ability(A.ACTIVATED, (damage(2),), (RED, TAP, SAC),
                condition=D.COUNT_THRESHOLD, condition_threshold=7,
                tags=(G.SEVEN_CARDS_IN_YOUR_GRAVEYARD, G.ACTIVATION_RESTRICTION, G.SACRIFICE_SOURCE)))),
    'Bloodstained Mire': fetch(7, G.SEARCH_SWAMP_OR_MOUNTAIN),
    'Boltwave': card(8, (ability(A.SPELL, (damage(3, R.OPPONENT, F.PLAYER),), (RED,)),)),
    'Boros Charm': card(9, (
        ability(A.SPELL, (damage(4, filter=PLAYER_PW),), RW, mode=1, tags=(G.CHOOSE_ONE_MODE,)),
        ability(A.SPELL, (effect(E.GRANT_KEYWORD, recipient=R.YOUR_PERMANENTS, filter=PERMANENTS,
                                 keyword=K.INDESTRUCTIBLE, duration=U.END_OF_TURN),), RW, mode=2,
                tags=(G.CHOOSE_ONE_MODE,)),
        ability(A.SPELL, (effect(E.GRANT_KEYWORD, 1, R.TARGET, F.CREATURE,
                                 keyword=K.DOUBLE_STRIKE, duration=U.END_OF_TURN),), RW, mode=3,
                tags=(G.CHOOSE_ONE_MODE,)))),
    'Eidolon of the Great Revel': card(10, (permanent(RR),
        ability(A.TRIGGERED, (damage(2, R.SPELL_CASTER, F.PLAYER),),
                trigger=T.SPELL_CAST, condition=D.SPELL_MANA_VALUE_AT_MOST, condition_threshold=3))),
    'Elegant Parlor': card(11, (*dual(),
        ability(A.STATIC, (effect(E.ENTER_TAPPED, 1, R.SOURCE),)),
        ability(A.TRIGGERED, (effect(E.SURVEIL, 1, R.SELF_PLAYER, duration=U.IMMEDIATE),),
                trigger=T.SOURCE_ENTERS))),
    'Goblin Guide': card(12, (permanent((RED,)),
        ability(A.TRIGGERED, (
            effect(E.REVEAL, 1, R.DEFENDING_PLAYER, F.CARD,
                   tags=(G.TOP_LIBRARY_CARD, G.DEFENDING_PLAYERS_CARD)),
            effect(E.MOVE_ZONE, 1, R.DEFENDING_PLAYER, F.LAND, destination=Z.HAND,
                   tags=(G.TOP_LIBRARY_CARD, G.DEFENDING_PLAYERS_CARD, G.IF_REVEALED_CARD_IS_LAND))),
                trigger=T.SOURCE_ATTACKS, coverage=V.PARTIAL)),
        keywords=(K.HASTE,), coverage=V.PARTIAL),
    'Inspiring Vantage': card(13, (*dual(),
        ability(A.STATIC, (effect(E.ENTER_TAPPED, 1, R.SOURCE,
                                  tags=(G.ENTER_TAPPED_IF_THREE_OTHER_LANDS,)),)))),
    'Lava Spike': card(14, (ability(A.SPELL, (damage(3, filter=PLAYER_PW),), (RED,)),)),
    'Monastery Swiftspear': card(15, (permanent((RED,)),
        ability(A.TRIGGERED, (effect(E.MODIFY_STATS, 1, R.SOURCE, F.CREATURE,
                                    duration=U.END_OF_TURN, tags=(G.POWER_AND_TOUGHNESS,)),),
                trigger=T.SPELL_CAST, condition=D.YOU_CAST_NONCREATURE)), keywords=(K.HASTE, K.PROWESS)),
    'Sacred Foundry': card(16, (*dual(),
        ability(A.STATIC, (effect(E.ENTER_TAPPED, 1, R.SOURCE,
                    tags=(G.OPTIONAL_PAY_TWO_LIFE_OR_ENTER_TAPPED,)),), coverage=V.PARTIAL)), coverage=V.PARTIAL),
    'Searing Blaze': card(17, (ability(A.SPELL, (
        Effect(E.DAMAGE, Quantity(Q.LANDFALL_DAMAGE), R.TARGET, PLAYER_PW, duration=U.IMMEDIATE),
        Effect(E.DAMAGE, Quantity(Q.LANDFALL_DAMAGE), R.TARGET, F.CREATURE, duration=U.IMMEDIATE,
               tags=(G.SAME_CONTROLLER_AS_FIRST_TARGET, G.SECOND_TARGET))), RR,
        coverage=V.PARTIAL),), coverage=V.PARTIAL),
    'Skewer the Critics': card(18, (
        ability(A.SPELL, (damage(3),), (mana(2, Color.GENERIC), RED)),
        ability(A.SPELL, (damage(3),), (RED,), condition=D.OPPONENT_LOST_LIFE,
                tags=(G.ALTERNATIVE_CAST_COST, G.CONDITION_THIS_TURN)))),
    'Skullcrack': card(19, (ability(A.SPELL, (
        effect(E.CANT_GAIN_LIFE, recipient=R.EACH_MATCHING, filter=F.PLAYER, duration=U.END_OF_TURN),
        effect(E.DAMAGE_CANT_BE_PREVENTED, duration=U.END_OF_TURN),
        damage(3, filter=PLAYER_PW)), (GENERIC, RED)),)),
    'Sunbaked Canyon': card(20, (*dual((TAP, LIFE)),
        ability(A.ACTIVATED, (effect(E.DRAW, 1, R.SELF_PLAYER, duration=U.IMMEDIATE),),
                (GENERIC, TAP, SAC), tags=(G.SACRIFICE_SOURCE,)))),
    'Deflecting Palm': card(21, (ability(A.SPELL, (
        effect(E.PREVENT_DAMAGE, recipient=R.SELF_PLAYER, duration=U.END_OF_TURN,
               tags=(G.NEXT_DAMAGE_FROM_CHOSEN_SOURCE,)),
        Effect(E.DAMAGE, Quantity(Q.PREVENTED_DAMAGE), R.CHOSEN_SOURCE_CONTROLLER, F.PLAYER,
               duration=U.IMMEDIATE, tags=(G.ONLY_IF_DAMAGE_PREVENTED,))), RW, coverage=V.PARTIAL),), coverage=V.PARTIAL),
    'Lightning Helix': card(22, (ability(A.SPELL, (damage(3),
        effect(E.GAIN_LIFE, 3, R.SELF_PLAYER, duration=U.IMMEDIATE)), RW),)),
    'Molten Rain': card(23, (ability(A.SPELL, (
        effect(E.DESTROY, 1, R.TARGET, F.LAND, duration=U.IMMEDIATE),
        damage(2, R.TARGET_CONTROLLER, F.PLAYER, tags=(G.IF_TARGET_WAS_NONBASIC,))),
        (GENERIC, *RR), coverage=V.PARTIAL),), coverage=V.PARTIAL),
    'Rest in Peace': card(24, (permanent((GENERIC, WHITE)),
        ability(A.TRIGGERED, (effect(E.EXILE, recipient=R.EACH_MATCHING, filter=F.CARD,
                destination=Z.EXILE, tags=(G.ALL_GRAVEYARDS,)),), trigger=T.SOURCE_ENTERS),
        ability(A.STATIC, (effect(E.REPLACE_ZONE_CHANGE, recipient=R.EACH_MATCHING,
                destination=Z.EXILE, duration=U.WHILE_SOURCE_REMAINS,
                tags=(G.CARD_OR_TOKEN_TO_ANY_GRAVEYARD,)),), coverage=V.PARTIAL)), coverage=V.PARTIAL),
    'Roiling Vortex': card(25, (permanent((GENERIC, RED)),
        ability(A.TRIGGERED, (damage(1, R.ACTIVE_PLAYER, F.PLAYER),), trigger=T.UPKEEP),
        ability(A.TRIGGERED, (damage(5, R.SPELL_CASTER, F.PLAYER),), trigger=T.SPELL_CAST,
                condition=D.NO_MANA_SPENT),
        ability(A.ACTIVATED, (effect(E.CANT_GAIN_LIFE, recipient=R.OPPONENT, filter=F.PLAYER,
                                     duration=U.END_OF_TURN),), (RED,)))),
    'Searing Blood': card(26, (ability(A.SPELL, (
        damage(2, filter=F.CREATURE),
        damage(3, R.TARGET_CONTROLLER, F.PLAYER,
               tags=(G.DELAYED_TRIGGER, G.THAT_CREATURES_CONTROLLER, G.IF_TARGET_DIES_THIS_TURN))),
        RR, coverage=V.PARTIAL),), coverage=V.PARTIAL),
    'Smash to Smithereens': card(27, (ability(A.SPELL, (
        effect(E.DESTROY, 1, R.TARGET, F.ARTIFACT, duration=U.IMMEDIATE),
        damage(3, R.TARGET_CONTROLLER, F.PLAYER, tags=(G.PREVIOUS_TARGET_CONTROLLER,))),
        (GENERIC, RED), coverage=V.PARTIAL),), coverage=V.PARTIAL),
    'Wear // Tear': card(28, (
        ability(A.SPELL, (effect(E.DESTROY, 1, R.TARGET, F.ARTIFACT),), (GENERIC, RED), mode=1),
        ability(A.SPELL, (effect(E.DESTROY, 1, R.TARGET, F.ENCHANTMENT),), (WHITE,), mode=2),
        ability(A.SPELL, (effect(E.DESTROY, 1, R.TARGET, F.ARTIFACT),
                          effect(E.DESTROY, 1, R.TARGET, F.ENCHANTMENT, tags=(G.SECOND_TARGET,))),
                (GENERIC, RED, WHITE), mode=3, tags=(G.FUSE_FROM_HAND,), coverage=V.PARTIAL)), coverage=V.PARTIAL),
})

CARD_REGISTRY = MappingProxyType({**INITIAL_CARDS, **BURN_CARDS,
    'Wear/Tear': BURN_CARDS['Wear // Tear'],
    'Wear': replace(BURN_CARDS['Wear // Tear'], abilities=BURN_CARDS['Wear // Tear'].abilities[:1]),
    'Tear': replace(BURN_CARDS['Wear // Tear'], abilities=BURN_CARDS['Wear // Tear'].abilities[1:2])})
