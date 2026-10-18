"""Numerical schema for the pinned Foundations mono-red experiment.

Names are exact identity lookups, never language-model input. Native live IDs
bind references only. Large/arbitrary domains fail explicitly: this encoder is
for this deck, not a claim of support for RustMage's whole catalogue.
"""
from dataclasses import dataclass
import hashlib
import itertools
import json

import numpy as np

from ..backends.rustmage import OBSERVATION_SCHEMA, RUSTMAGE_REVISION, _plain, domain_action
from ..baselines.foundations import DATA, FACTS
from ..core import Response
from ..errors import UnsupportedInteraction
from ..semantics import CardSemantics, Coverage
from .features import numeric, onehot
from .history import HISTORY_WIDTH

CARDS = tuple(c['name'] for c in DATA['decks']['red']['cards'])
CARD_IDS = {name: i + 1 for i, name in enumerate(CARDS)}
REGISTRY = {name: CardSemantics(i, Coverage.PARTIAL) for name, i in CARD_IDS.items()}
ZONES = ('player', 'hand', 'battlefield', 'graveyard', 'exile', 'stack', 'command', 'library')
STEPS = ('Pregame', 'Upkeep', 'Draw', 'Main1', 'BeginCombat', 'Attackers', 'Blockers',
         'FirstStrikeDamage', 'CombatDamage', 'EndCombat', 'Main2', 'End', 'Cleanup')
DECISIONS = ('mulligan', 'bottom', 'opening_hand', 'priority', 'pay_costs', 'kicker',
             'targets', 'attackers', 'blockers', 'damage', 'discard', 'order_triggers',
             'optional_effect', 'mana_lot_spend')
ACTION_FIELDS = {
    'Concede': (), 'Keep': (), 'Mulligan': (), 'Bottom': ('card',),
    'FinishOpeningHand': (), 'Pass': (), 'PlayLand': ('card',), 'TapMana': ('card',),
    'AnnounceCast': ('card',), 'AnnounceActivate': ('card', 'ability', 'counter_payment'),
    'PayCosts': ('payment',), 'Cast': ('card', 'target', 'payment'),
    'BeginCast': ('card', 'payment'),
    'Activate': ('card', 'ability', 'target', 'payment', 'counter_payment'),
    'BeginActivate': ('card', 'ability', 'payment', 'counter_payment'),
    'ChooseKicker': ('pay',), 'ChooseTarget': ('target',), 'FinishTargets': (),
    'Attack': ('card',), 'FinishAttackers': (), 'Block': ('blocker', 'attacker'),
    'FinishBlockers': (), 'AssignDamage': ('target', 'amount'), 'Discard': ('card',),
    'PutTrigger': ('id',), 'ChooseOptional': ('accept',), 'ChooseManaLotSpend': ('amount',),
}
ACTIONS = tuple(ACTION_FIELDS)
# Reviewed, deliberately partial semantic labels. Dynamic P/T and keywords come
# from the engine, including Lavarunner's graveyard condition and temporary buffs.
TAGS = ('red_mana', 'tap_cost', 'sacrifice_source', 'damage_any', 'damage_opponent',
        'entry_trigger', 'noncreature_cast_trigger', 'dragon_cast_trigger',
        'two_instants_sorceries_graveyard', 'grant_haste', 'grant_power',
        'until_end_turn', 'kicker', 'grant_first_strike', 'damage_creature',
        'damage_planeswalker', 'exile_if_dies_this_turn', 'exile_attached_equipment')
LABELS = {
    'Mountain': ('red_mana', 'tap_cost'),
    'Fanatical Firebrand': ('tap_cost', 'sacrifice_source', 'damage_any'),
    'Ghitu Lavarunner': ('two_instants_sorceries_graveyard', 'grant_haste', 'grant_power'),
    'Firebrand Archer': ('noncreature_cast_trigger', 'damage_opponent'),
    'Viashino Pyromancer': ('entry_trigger', 'damage_opponent'),
    'Firespitter Whelp': ('noncreature_cast_trigger', 'dragon_cast_trigger', 'damage_opponent'),
    'Burst Lightning': ('damage_any', 'kicker'),
    'Kindled Fury': ('grant_power', 'grant_first_strike', 'until_end_turn'),
    'Sure Strike': ('grant_power', 'grant_first_strike', 'until_end_turn'),
    'Scorching Dragonfire': ('damage_creature', 'damage_planeswalker', 'exile_if_dies_this_turn'),
    'Fiery Annihilation': ('damage_creature', 'exile_if_dies_this_turn', 'exile_attached_equipment'),
}
DAMAGE = {'Fanatical Firebrand': 1, 'Firebrand Archer': 1, 'Viashino Pyromancer': 2,
          'Firespitter Whelp': 1, 'Burst Lightning': 2, 'Scorching Dragonfire': 3,
          'Fiery Annihilation': 5}
POWER_BONUS = {'Ghitu Lavarunner': 1, 'Kindled Fury': 1, 'Sure Strike': 3}
SEMANTICS = {name: [1.] + [float(tag in LABELS[name]) for tag in TAGS]
             + numeric(DAMAGE.get(name, 0)) + numeric(4 if name == 'Burst Lightning' else 0)
             + numeric(POWER_BONUS.get(name, 0))
             for name in CARDS}
SEMANTIC_WIDTH = len(SEMANTICS['Mountain'])
COUNTERS = ('plus_one', 'minus_one', 'stun', 'loyalty', 'lore')
FLAGS = ('tapped', 'summoning_sick', 'attacking', 'damaged_this_turn', 'dealt_damage_this_turn', 'modified')
PLAYER_STATS = ('life', 'hand_count', 'library_count', 'poison_counters', 'spells_cast_this_turn',
                'cards_drawn_this_turn', 'life_lost_this_turn', 'extra_land_plays_remaining')


def entity_row(obj, side, *, player=False):
    name = obj.get('name')
    row = onehot('player' if player else obj.get('zone', '').lower(), ZONES) + onehot(side, (0, 1))
    row += [float(player)] + [float(bool(obj.get(k))) for k in FLAGS]
    row += [float(bool(obj.get('kinds', 0) & 2**i)) for i in range(8)]
    row += [float(bool(obj.get('colors', 0) & 2**i)) for i in range(6)]
    row += [float(bool(obj.get('keywords', 0) & 2**i)) for i in range(28)]
    row += sum((numeric(obj.get(k)) for k in ('power', 'toughness', 'damage')), [])
    row += numeric(FACTS[name]['mana_value'] if name in CARD_IDS else None)
    row += sum((numeric(obj.get('counters', {}).get(k, 0)) for k in COUNTERS), [])
    row += sum((numeric(obj.get(k) if player else None) for k in PLAYER_STATS), [])
    row += sum((numeric(v) for v in (obj.get('mana', (0,)*6) if player else (0,)*6)), [])
    return row


ENTITY_WIDTH = len(entity_row({}, 0))
# Each ordered source/target/other role has its own identity and raw facts.
# This preserves e.g. Block(blocker,attacker) even though the attention reference
# also pools linked objects. IDs themselves never enter a tensor.
ROLE_WIDTH = 1 + len(CARDS) + ENTITY_WIDTH
CANDIDATE_WIDTH = len(ACTIONS) + 3*ROLE_WIDTH + 2*2 + 6*2 + 2*2 + len(COUNTERS)*2
CONTEXT_WIDTH = len(STEPS) + len(DECISIONS) + 3 + 2*8 + len(CARDS) + 2*3 + 6*4
FEATURE_SPEC = dict(entities=ENTITY_WIDTH, semantics=SEMANTIC_WIDTH,
                    context=CONTEXT_WIDTH, candidates=CANDIDATE_WIDTH, history=HISTORY_WIDTH)


def canonical(action):
    return json.dumps(_plain(action), sort_keys=True, separators=(',', ':'))


def expand_options(decision, *, allow_concede=False, limit=4096):
    """Exactly enumerate this deck's small domains; deduplicate representatives.

    The bound is an explicit unsupported-interaction error, never a truncation.
    No choices are pruned for strategic reasons. Voluntary resignation is an
    explicit experiment setting, identical in training and evaluation.
    """
    choices = {}
    for option in decision.options:
        if option.kind == 'finite':
            action = option.constraints['action']
            if action['kind'] == 'Concede' and not allow_concede: continue
            choices.setdefault(canonical(action), (Response(option.id), action))
            continue
        c = option.constraints
        if option.id == 'domain:mana_payment':
            # Monored produces only R. Retain a general exact small-domain
            # enumerator for tests, with an explicit product-size guard.
            ranges = [range(lo, hi+1) for lo, hi in zip(c['required'], c['available'])]
            size = 1
            for values in ranges: size *= len(values)
            if size > limit: raise UnsupportedInteraction('Payment domain exceeds mono-red enumeration bound')
            values = (v for v in itertools.product(*ranges)
                      if sum(v)-sum(c['required']) == c['generic'])
        elif option.id in ('domain:damage_assignment', 'domain:mana_lot_spend'):
            if c.get('values') is not None: values = c['values']
            else:
                if c['maximum']-c['minimum']+1 > limit:
                    raise UnsupportedInteraction('Numeric domain exceeds mono-red enumeration bound')
                values = range(c['minimum'], c['maximum']+1)
        else:
            raise UnsupportedInteraction('Domain outside mono-red schema: ' + option.id)
        for value in values:
            action = domain_action(option, value)
            choices.setdefault(canonical(action), (Response(option.id, value), action))
    if not choices: raise UnsupportedInteraction('No playable action')
    return tuple(choices.values())


@dataclass
class EncodedRustDecision:
    arrays: dict
    responses: tuple
    actions: tuple
    token: object


class RustMageEncoder:
    registry = REGISTRY
    card_count = len(CARDS) + 1
    feature_spec = FEATURE_SPEC

    def __init__(self, *, allow_concede=False):
        self.allow_concede = allow_concede

    def manifest(self):
        result = dict(backend='rustmage', feature_version=2, observation_schema=OBSERVATION_SCHEMA,
            rustmage_revision=RUSTMAGE_REVISION, deck=DATA['decks']['red'], cards=CARDS,
            facts={n: FACTS[n] for n in CARDS}, semantic_tags=TAGS, semantic_labels=LABELS,
            semantics=SEMANTICS, actions=ACTION_FIELDS, decisions=DECISIONS, steps=STEPS,
            zones=ZONES, counters=COUNTERS, flags=FLAGS, player_stats=PLAYER_STATS,
            feature_spec=FEATURE_SPEC, allow_concede=self.allow_concede,
            scope='Foundations mono-red mirrors and passive 60-Mountain opponent')
        result['fingerprint'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result

    def encode(self, obs, *, choices=None):
        if obs.backend != 'rustmage' or obs.decision is None:
            raise UnsupportedInteraction('RustMage decision required')
        raw = obs.native
        if raw['schema'] != OBSERVATION_SCHEMA or raw['decision'] not in DECISIONS:
            raise UnsupportedInteraction('Unsupported RustMage schema/decision: ' + str(raw['decision']))
        if raw['step'] not in STEPS: raise UnsupportedInteraction('Unsupported step')
        me = obs.actor.value
        players = sorted(obs.players, key=lambda p: p.id != obs.actor)
        objects = list(obs.objects.values())
        if any(o.name not in CARD_IDS for o in objects):
            raise UnsupportedInteraction('Card outside Foundations mono-red vocabulary')
        rows = [entity_row(p.native, int(p.id != obs.actor), player=True) for p in players]
        rows += [entity_row(o.properties, int(o.controller != obs.actor)) for o in objects]
        ids = [0, 0] + [CARD_IDS[o.name] for o in objects]
        lookup = {p.id.value: i for i, p in enumerate(players)}
        lookup.update({int(o.id.value): i+2 for i, o in enumerate(objects)})
        stack = list(raw['stack']) + list(raw['waiting_triggers'])
        pending = raw.get('announcing') or raw.get('resolving') or {}
        stack_by_id = {s['id']: s for s in stack}
        n = len(rows)
        def resolve(target):
            if not target: return None
            if 'Player' in target: return lookup[target['Player']]
            if 'Card' in target: return lookup.get(target['Card'])
            if 'Spell' in target: return lookup.get(target['Spell'])
            if 'Stack' in target: return lookup.get(stack_by_id.get(target['Stack'], {}).get('card'))
            raise UnsupportedInteraction('Unknown target representation')
        def role(index, name=None):
            if index is not None:
                return [1.] + onehot(ids[index], range(1, len(CARDS)+1)) + rows[index]
            return [float(name in CARD_IDS)] + onehot(CARD_IDS.get(name), range(1, len(CARDS)+1)) + [0.]*ENTITY_WIDTH
        relations = np.zeros((n, n, 4), np.float32)
        for i, o in enumerate(objects, 2):
            relations[i, lookup[o.controller.value], 0] = 1
            attached = o.properties.get('attached_to')
            if attached and attached['card'] in lookup: relations[i, lookup[attached['card']], 1] = 1
        for s in stack + ([pending] if pending else []):
            source = lookup.get(s.get('card'))
            if source is not None:
                for t in s.get('targets', ()):
                    dest = resolve(t['target'])
                    if dest is not None: relations[source, dest, 2] = 1
        for left, right in zip(raw['stack'], raw['stack'][1:]):
            i, j = lookup.get(left.get('card')), lookup.get(right.get('card'))
            if i is not None and j is not None: relations[i, j, 3] = 1
        for blocker, attacker in raw.get('blocks', ()):
            if blocker in lookup and attacker in lookup: relations[lookup[blocker], lookup[attacker], 1] = 1
        if choices is None:
            choices = expand_options(obs.decision, allow_concede=self.allow_concede)
        candidates, refs = [], []
        for response, action in choices:
            kind = action['kind']
            if kind not in ACTION_FIELDS or set(action) != {'kind', *ACTION_FIELDS[kind]}:
                raise UnsupportedInteraction('Unknown native action fields: ' + repr(action))
            source_item = stack_by_id.get(action.get('id'), {}) if kind == 'PutTrigger' else pending
            source = lookup.get(action.get('card', action.get('blocker', source_item.get('card'))))
            target = resolve(action.get('target'))
            other = lookup.get(action.get('attacker'))
            source_name = source_item.get('source_card')
            row = onehot(kind, ACTIONS) + role(source, source_name) + role(target) + role(other)
            row += numeric(action.get('ability')) + numeric(action.get('amount'))
            row += sum((numeric(v) for v in action.get('payment', (None,)*6)), [])
            row += numeric(int(action['pay']) if 'pay' in action else None)
            row += numeric(int(action['accept']) if 'accept' in action else None)
            row += sum((numeric(action.get('counter_payment', {}).get(k, 0)) for k in COUNTERS), [])
            candidates.append(row)
            ref = np.zeros(n, np.float32)
            for index in (source, target, other):
                if index is not None: ref[index] += 1
            refs.append(ref / max(ref.sum(), 1))
        own_objects = [o for o in objects if o.controller == obs.actor]
        hand = [o for o in own_objects if o.zone == 'hand']
        battlefield = [o for o in own_objects if o.zone == 'battlefield']
        numbers = (raw['turn'], len(raw['stack']), len(raw['waiting_triggers']), len(hand),
            sum(o.name == 'Mountain' for o in hand), sum(o.name == 'Mountain' and not o.properties['tapped'] for o in battlefield),
            sum(o.name != 'Mountain' for o in battlefield), sum(FACTS[o.name]['mana_value'] == 2 for o in hand))
        context = onehot(raw['step'], STEPS) + onehot(raw['decision'], DECISIONS)
        context += [float(raw['active'] == me), float(bool(pending)), float(bool(raw.get('announcing_payment')))]
        context += sum((numeric(v) for v in numbers), [])
        context += onehot(CARD_IDS.get(pending.get('source_card')), range(1, len(CARDS)+1))
        context += numeric(raw.get('targeting_group')) + numeric(pending.get('x'))
        context += numeric(int((pending.get('cast') or {}).get('kicker') or False))
        payment = raw.get('mana_payment') or {}
        context += sum((numeric(v) for key in ('required', 'available') for v in payment.get(key, (None,)*6)), [])
        arrays = dict(entities=np.asarray(rows, np.float32), card_ids=np.asarray(ids, np.int32),
            entity_mask=np.ones(n, bool), semantics=np.asarray([[([0.]*SEMANTIC_WIDTH if i < 2 else SEMANTICS[objects[i-2].name])] for i in range(n)], np.float32),
            semantic_mask=np.ones((n, 1), bool), relations=relations, context=np.asarray(context, np.float32),
            candidates=np.asarray(candidates, np.float32), candidate_refs=np.asarray(refs, np.float32),
            candidate_mask=np.ones(len(choices), bool), selection=np.zeros(CANDIDATE_WIDTH, np.float32),
            history=np.zeros((1, HISTORY_WIDTH), np.float32), history_cards=np.zeros(1, np.int32), history_mask=np.zeros(1, bool))
        for key, width in FEATURE_SPEC.items():
            if arrays[key].shape[-1] != width: raise AssertionError((key, arrays[key].shape, width))
        return EncodedRustDecision(arrays, tuple(c[0] for c in choices), tuple(c[1] for c in choices), obs.decision.token)
