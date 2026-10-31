"""General numerical card traits with a bounded RustMage action schema.

Card names resolve offline descriptors; names and catalogue IDs never enter model
features. Unknown cards retain their visible live characteristics and explicit
unknown static traits. Action support is versioned separately from card traits.
"""
from dataclasses import dataclass
import hashlib
import itertools
import json

import numpy as np

from ..backends.rustmage import OBSERVATION_SCHEMA, RUSTMAGE_REVISION, _plain, domain_action
from ..core import Response
from ..errors import UnsupportedInteraction
from ..semantics import CardSemantics, Coverage
from .features import numeric, onehot
from .history import HISTORY_WIDTH
from . import card_traits

CARDS = card_traits.CARDS
FACTS = card_traits.FACTS
CARD_IDS = {name: card_traits.index_for(name) for name in (*CARDS, *card_traits.ALIASES)}


class TraitRegistry(dict):
    # Descriptor IDs remain bookkeeping for history; model identity inputs are 0.
    trait_rows = (card_traits.UNKNOWN_TRAITS,) + tuple(card_traits.TRAITS[n] for n in CARDS)
    trait_width = card_traits.TRAIT_WIDTH
    identity_embeddings = False


REGISTRY = TraitRegistry({name: CardSemantics(i, Coverage.PARTIAL) for name, i in CARD_IDS.items()})


def visible_identity(obj):
    if any(obj.get(k) for k in ('hideInfo', 'faceDown', 'face_down')):
        return None
    identity = obj.get('card')
    if not isinstance(identity, str):
        identity = obj.get('name')
    return card_traits.ALIASES.get(identity, identity)


def static_traits(obj):
    return card_traits.traits_for(visible_identity(obj))


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
# One fixed feature schema for the entire catalogue, including future unknowns.
SEMANTICS = card_traits.TRAITS
SEMANTIC_WIDTH = card_traits.TRAIT_WIDTH
TYPE_BITS, COLOR_BITS, KEYWORD_BITS = 16, 6, 64
COUNTERS = ('plus_one', 'minus_one', 'stun', 'loyalty', 'lore')
FLAGS = ('tapped', 'summoning_sick', 'attacking', 'damaged_this_turn', 'dealt_damage_this_turn', 'modified')
PLAYER_STATS = ('life', 'hand_count', 'library_count', 'poison_counters', 'spells_cast_this_turn',
                'cards_drawn_this_turn', 'life_lost_this_turn', 'extra_land_plays_remaining')


def entity_row(obj, side, *, player=False):
    name = visible_identity(obj)
    row = onehot('player' if player else obj.get('zone', '').lower(), ZONES) + onehot(side, (0, 1))
    row += [float(player)] + [float(bool(obj.get(k))) for k in FLAGS]
    row += [float(bool(obj.get('kinds', 0) & 2**i)) for i in range(TYPE_BITS)]
    row += [float(bool(obj.get('colors', 0) & 2**i)) for i in range(COLOR_BITS)]
    row += [float(bool(obj.get('keywords', 0) & 2**i)) for i in range(KEYWORD_BITS)]
    row += sum((numeric(obj.get(k)) for k in ('power', 'toughness', 'damage')), [])
    row += numeric(FACTS.get(name, {}).get('mana_value'))
    row += sum((numeric(obj.get('counters', {}).get(k, 0)) for k in COUNTERS), [])
    row += sum((numeric(obj.get(k) if player else None) for k in PLAYER_STATS), [])
    row += sum((numeric(v) for v in (obj.get('mana', (0,)*6) if player else (0,)*6)), [])
    return row


ENTITY_WIDTH = len(entity_row({}, 0))
# Each ordered source/target/other role has static traits and live facts.
# This preserves e.g. Block(blocker,attacker) even though the attention reference
# also pools linked objects. IDs themselves never enter a tensor.
ROLE_WIDTH = 1 + SEMANTIC_WIDTH + ENTITY_WIDTH
CANDIDATE_WIDTH = len(ACTIONS) + 3*ROLE_WIDTH + 2*2 + 6*2 + 2*2 + len(COUNTERS)*2
CONTEXT_WIDTH = len(STEPS) + len(DECISIONS) + 3 + 2*8 + SEMANTIC_WIDTH + 2*3 + 6*4
FEATURE_SPEC = dict(entities=ENTITY_WIDTH, semantics=SEMANTIC_WIDTH,
                    context=CONTEXT_WIDTH, candidates=CANDIDATE_WIDTH, history=HISTORY_WIDTH + SEMANTIC_WIDTH)


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
    card_count = 1  # No per-card learned identity table; shared traits carry meaning.
    feature_spec = FEATURE_SPEC
    action_fields, actions, decisions = ACTION_FIELDS, ACTIONS, DECISIONS
    decision_aliases = {}
    entity_row = staticmethod(entity_row)

    def expand_options(self, decision):
        return expand_options(decision, allow_concede=self.allow_concede)

    def action_extras(self, action):
        return []

    def action_roles(self, action):
        return action

    def __init__(self, *, allow_concede=False):
        self.allow_concede = allow_concede

    def manifest(self):
        result = dict(backend='rustmage', feature_version=3, observation_schema=OBSERVATION_SCHEMA,
            rustmage_revision=RUSTMAGE_REVISION, card_encoding='shared_traits_without_identity_embeddings',
            card_traits=dict(card_traits.MANIFEST), trait_features=card_traits.FEATURE_NAMES,
            actions=self.action_fields, decisions=self.decisions, steps=STEPS,
            zones=ZONES, counters=COUNTERS, flags=FLAGS, player_stats=PLAYER_STATS,
            bit_widths=dict(kinds=TYPE_BITS, colors=COLOR_BITS, keywords=KEYWORD_BITS),
            feature_spec=self.feature_spec, allow_concede=self.allow_concede,
            scope='General catalogue card traits; bounded action schema, currently validated on Foundations')
        result['fingerprint'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result

    def encode(self, obs, *, choices=None):
        if obs.backend != 'rustmage' or obs.decision is None:
            raise UnsupportedInteraction('RustMage decision required')
        raw = obs.native
        if hasattr(raw, 'native_choices'):
            native_choices = raw.native_choices(self.allow_concede)
            if choices is None or choices == native_choices:
                return raw.encoded(obs.decision.token, self.allow_concede)
        decision = self.decision_aliases.get(raw['decision'], raw['decision'])
        if raw['schema'] != OBSERVATION_SCHEMA or decision not in self.decisions:
            raise UnsupportedInteraction('Unsupported RustMage schema/decision: ' + str(raw['decision']))
        if raw['step'] not in STEPS: raise UnsupportedInteraction('Unsupported step')
        me = obs.actor.value
        players = sorted(obs.players, key=lambda p: p.id != obs.actor)
        objects = list(obs.objects.values())
        rows = [self.entity_row(p.native, int(p.id != obs.actor), player=True) for p in players]
        rows += [self.entity_row(o.properties, int(o.controller != obs.actor)) for o in objects]
        ids = [0] * len(rows)
        trait_rows = [[0.] * SEMANTIC_WIDTH for _ in players]
        trait_rows += [list(static_traits(o.properties)) for o in objects]
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
                return [1.] + trait_rows[index] + rows[index]
            traits = list(card_traits.traits_for(name)) if name is not None else [0.] * SEMANTIC_WIDTH
            return [float(name is not None)] + traits + [0.] * self.feature_spec['entities']
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
            choices = self.expand_options(obs.decision)
        candidates, refs = [], []
        for response, action in choices:
            kind = action['kind']
            if kind not in self.action_fields or set(action) != {'kind', *self.action_fields[kind]}:
                raise UnsupportedInteraction('Unknown native action fields: ' + repr(action))
            roles = self.action_roles(action)
            source_item = stack_by_id.get(action.get('id'), {}) if kind == 'PutTrigger' else pending
            source = lookup.get(roles.get('card', roles.get('blocker', source_item.get('card'))))
            target = resolve(roles.get('target'))
            other = lookup.get(roles.get('attacker'))
            source_name = source_item.get('source_card')
            row = onehot(kind, self.actions) + role(source, source_name) + role(target) + role(other)
            row += numeric(action.get('ability')) + numeric(action.get('amount'))
            row += sum((numeric(v) for v in action.get('payment', (None,)*6)), [])
            row += numeric(int(action['pay']) if 'pay' in action else None)
            row += numeric(int(action['accept']) if 'accept' in action else None)
            row += sum((numeric(action.get('counter_payment', {}).get(k, 0)) for k in COUNTERS), [])
            row += self.action_extras(action)
            candidates.append(row)
            ref = np.zeros(n, np.float32)
            for index in (source, target, other):
                if index is not None: ref[index] += 1
            refs.append(ref / max(ref.sum(), 1))
        own_objects = [o for o in objects if o.controller == obs.actor]
        hand = [o for o in own_objects if o.zone == 'hand']
        battlefield = [o for o in own_objects if o.zone == 'battlefield']
        numbers = (raw['turn'], len(raw['stack']), len(raw['waiting_triggers']), len(hand),
            sum(bool(o.properties.get('kinds', 0) & 2) for o in hand),
            sum(bool(o.properties.get('kinds', 0) & 2) and not o.properties.get('tapped', False) for o in battlefield),
            sum(not (o.properties.get('kinds', 0) & 2) for o in battlefield),
            sum(bool(o.properties.get('kinds', 0) & 1) for o in hand))
        context = onehot(raw['step'], STEPS) + onehot(decision, self.decisions)
        context += [float(raw['active'] == me), float(bool(pending)), float(bool(raw.get('announcing_payment')))]
        context += sum((numeric(v) for v in numbers), [])
        source_name = pending.get('source_card')
        context += list(card_traits.traits_for(source_name)) if source_name is not None else [0.] * SEMANTIC_WIDTH
        context += numeric(raw.get('targeting_group')) + numeric(pending.get('x'))
        context += numeric(int((pending.get('cast') or {}).get('kicker') or False))
        payment = raw.get('mana_payment') or {}
        context += sum((numeric(v) for key in ('required', 'available') for v in payment.get(key, (None,)*6)), [])
        arrays = dict(entities=np.asarray(rows, np.float32), card_ids=np.asarray(ids, np.int32),
            entity_mask=np.ones(n, bool), semantics=np.asarray([[traits] for traits in trait_rows], np.float32),
            semantic_mask=np.ones((n, 1), bool), relations=relations, context=np.asarray(context, np.float32),
            candidates=np.asarray(candidates, np.float32), candidate_refs=np.asarray(refs, np.float32),
            candidate_mask=np.ones(len(choices), bool), selection=np.zeros(self.feature_spec['candidates'], np.float32),
            history=np.zeros((1, self.feature_spec['history']), np.float32), history_cards=np.zeros(1, np.int32), history_mask=np.zeros(1, bool))
        for key, width in self.feature_spec.items():
            if arrays[key].shape[-1] != width: raise AssertionError((key, arrays[key].shape, width))
        return EncodedRustDecision(arrays, tuple(c[0] for c in choices), tuple(c[1] for c in choices), obs.decision.token)
