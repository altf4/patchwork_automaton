"""Versioned, text-free observation and candidate encoding for XMage.

Backend strings are used only for exact enum/identity lookups, never tokenized.
Unrecognized semantics stay unknown. Live UUIDs are only lookup references.
"""
from dataclasses import dataclass, replace
from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real

import numpy as np

from ..errors import UnsupportedInteraction
from ..semantics import (INITIAL_CARDS, UNKNOWN_CARD, Keyword, Coverage, AbilityKind,
    EffectKind, CostKind, QuantityKind, Recipient, ObjectFilter, Color, Duration,
    Zone, Trigger, Condition, RuleTag, encode_card, lookup_visible)
from ..burn_semantics import CARD_REGISTRY
from ..backends.xmage_prompts import (PURPOSES, PROMPT_RULE_VERSION, prompt_metadata,
                                    MANA_COLORS, TARGET_ZONES, prompt_source)

FEATURE_VERSION = 6
PHASES = ('unknown', 'untap', 'upkeep', 'draw', 'precombat_main', 'begin_combat',
          'declare_attackers', 'declare_blockers', 'first_combat_damage',
          'combat_damage', 'end_combat', 'postcombat_main', 'end_turn', 'cleanup')
DECISIONS = ('unknown', 'GAME_ASK', 'GAME_SELECT', 'GAME_TARGET', 'GAME_CHOOSE_ABILITY',
             'GAME_CHOOSE_PILE', 'GAME_CHOOSE_CHOICE', 'GAME_PLAY_MANA',
             'GAME_PLAY_XMANA', 'GAME_GET_AMOUNT', 'GAME_GET_MULTI_AMOUNT')
ZONES = ('unknown', 'player', 'hand', 'battlefield', 'graveyard', 'exile', 'stack',
         'command', 'sideboard', 'revealed', 'looked_at', 'library')
TYPES = ('LAND', 'CREATURE', 'INSTANT', 'SORCERY', 'ARTIFACT', 'ENCHANTMENT', 'PLANESWALKER', 'BATTLE')
COLORS = ('white', 'blue', 'black', 'red', 'green', 'colorless')
OPTION_KINDS = ('unknown', 'boolean', 'uuid', 'string', 'integer', 'distribution', 'digit')
ROLES = ('unknown', 'yes', 'no', 'play_land', 'cast', 'mana', 'target', 'ability', 'special', 'digit', 'cancel')
ABILITY_IDS = (0, *sorted({a.id for card in CARD_REGISTRY.values() for a in card.abilities}))
FLAGS = ('tapped', 'summoningSickness', 'canAttack', 'canBlock', 'attacking', 'blocking',
         'faceDown', 'isToken', 'isAbility')
STATS = ('power', 'toughness', 'damage', 'manaValue', 'loyalty', 'defense')
COUNTERS = ('+1/+1', '-1/-1', 'loyalty', 'poison', 'energy', 'stun')


def onehot(value, vocabulary):
    return [float(value == item) for item in vocabulary]


def numeric(value):
    """Signed logarithm plus known bit; no clipping of large game quantities."""
    if isinstance(value, bool):
        return [0., 0.]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return [0., 0.]
    if not math.isfinite(number):
        return [0., 0.]
    return [math.copysign(math.log1p(abs(number)), number) / 5., 1.]


def flags(properties, names):
    result = []
    for name in names:
        value = properties.get(name)
        result.extend((float(bool(value)) if value is not None else 0., float(value is not None)))
    return result


def counter_features(counters):
    if isinstance(counters, (list, tuple)):
        counters = {str(c.get('name', '')).lower(): c.get('count') for c in counters}
    if not isinstance(counters, Mapping):
        return [0.] * (2 * len(COUNTERS))
    counters = {str(k).lower(): v for k, v in counters.items()}
    return sum((numeric(counters.get(k, 0)) for k in COUNTERS), [])


def semantic_rows(card):
    """Ordered ability/cost/effect records; each categorical field is one-hot."""
    rows = []
    for index, ability in enumerate(card.abilities):
        for row_type, record in [('ability', ability), *[('cost', c) for c in ability.costs],
                                 *[('effect', e) for e in ability.effects]]:
            quantity = getattr(record, 'quantity', None)
            row = onehot(row_type, ('ability', 'cost', 'effect'))
            row += onehot(ability.kind, tuple(AbilityKind)) + onehot(ability.coverage, tuple(Coverage))
            row += onehot(ability.trigger, tuple(Trigger)) + onehot(ability.condition, tuple(Condition))
            row += numeric(ability.condition_threshold) + numeric(ability.mode) + numeric(index)
            row += onehot(record.kind if row_type == 'effect' else EffectKind.UNKNOWN, tuple(EffectKind))
            row += onehot(record.kind if row_type == 'cost' else CostKind.UNKNOWN, tuple(CostKind))
            row += onehot(quantity.kind if quantity else QuantityKind.UNKNOWN, tuple(QuantityKind))
            row += numeric(quantity.value if quantity and quantity.kind == QuantityKind.CONSTANT else None)
            row += onehot(getattr(record, 'recipient', Recipient.UNKNOWN), tuple(Recipient))
            mask = getattr(record, 'object_filter', ObjectFilter.UNKNOWN)
            row += [float(bool(mask & f)) for f in ObjectFilter]
            row += onehot(getattr(record, 'color', Color.UNKNOWN), tuple(Color))
            row += onehot(getattr(record, 'duration', Duration.UNKNOWN), tuple(Duration))
            row += onehot(getattr(record, 'destination', Zone.UNKNOWN), tuple(Zone))
            row += onehot(getattr(record, 'keyword', None), (None, *Keyword))
            row += [float(tag in ability.tags) for tag in RuleTag]
            row += [float(tag in getattr(record, 'tags', ())) if row_type == 'effect' else 0.
                    for tag in RuleTag]
            rows.append(row)
    return rows


SEMANTIC_WIDTH = len(semantic_rows(INITIAL_CARDS['Mountain'])[0])
# Entity features: zone, self/opponent/unknown controller, player, hidden,
# coverage, base keywords, current stats/flags/types/colors/counters/player data.
ENTITY_WIDTH = (len(ZONES) + 3 + 2 + 6 + len(Keyword) + 2 * len(Keyword)
                + 2 * len(STATS) + 2 * len(FLAGS) + len(TYPES) + 1
                + 2 * len(COLORS) + 2 * len(COUNTERS) + 2 * (3 + len(COLORS)))
CANDIDATE_WIDTH = (len(OPTION_KINDS) + len(ROLES) + 3 + 2 + 2 * 3 + 4 + len(AbilityKind)
                   + len(ABILITY_IDS) + SEMANTIC_WIDTH)
ABILITY_ID_SLICE = slice(CANDIDATE_WIDTH - SEMANTIC_WIDTH - len(ABILITY_IDS),
                         CANDIDATE_WIDTH - SEMANTIC_WIDTH)
HAND_MANA_BUCKETS = tuple(range(9))  # 8 includes all values >= 8.
CONTEXT_WIDTH = (len(PHASES) + len(DECISIONS) + 3 + 3 + 2 * 4 + len(PURPOSES) + 4
                 + 2 * len(MANA_COLORS) + 2 + len(TARGET_ZONES) + 2
                 + 2 * (len(TYPES) + len(HAND_MANA_BUCKETS)) + 3)


def hand_summary(obs):
    """Observable counts, never a hand-quality label or policy threshold."""
    hand = [o for o in obs.objects.values() if o.zone=='hand' and o.controller==obs.actor]
    player = next(p for p in obs.players if p.id==obs.actor)
    complete = len(hand)==player.hand_count and not any(not o.name or any(o.properties.get(k)
        for k in ('hideInfo','faceDown','face_down')) for o in hand)
    typed = complete and all(isinstance(o.properties.get('cardTypes'),(list,tuple,set,frozenset)) for o in hand)
    mana = [o.properties.get('manaValue') for o in hand]
    known_mana = complete and all(isinstance(v,Real) and not isinstance(v,bool)
                                  and math.isfinite(v) and v>=0 and float(v).is_integer() for v in mana)
    result = sum((numeric(sum(t in o.properties['cardTypes'] for o in hand) if typed else None) for t in TYPES), [])
    result += sum((numeric(sum(min(v,8)==bucket for v in mana) if known_mana else None)
                   for bucket in HAND_MANA_BUCKETS), [])
    return result


@dataclass
class EncodedDecision:
    arrays: dict
    options: tuple
    token: object
    entities: tuple


class Encoder:
    def __init__(self, registry=CARD_REGISTRY):
        self.registry = registry
        self.card_count = max((card.id for card in registry.values()), default=0) + 1

    def manifest(self):
        from .history import EVENTS, HISTORY_WIDTH
        from ..backends.burn_actions import RULES
        cards = {name: encode_card(card) for name, card in sorted(self.registry.items())}
        payload = dict(version=FEATURE_VERSION, cards=cards, phases=PHASES, decisions=DECISIONS,
                       zones=ZONES, types=TYPES, colors=COLORS, flags=FLAGS, stats=STATS,
                       counters=COUNTERS, option_kinds=OPTION_KINDS, roles=ROLES,
                       events=EVENTS, ability_ids=ABILITY_IDS,
                       prompt_purposes=PURPOSES, prompt_rule_version=PROMPT_RULE_VERSION,
                       prompt_mana_colors=MANA_COLORS,target_zones=TARGET_ZONES,hand_mana_buckets=HAND_MANA_BUCKETS,
                       action_rules=[(name, rule, identity) for (name, rule), identity in sorted(RULES.items())],
                       semantic_enums={enum.__name__: {v.name: int(v) for v in enum}
                           for enum in (RuleTag, EffectKind, QuantityKind, Recipient, Condition,
                                        AbilityKind, CostKind, Color, ObjectFilter, Trigger,
                                        Duration, Zone, Coverage, Keyword)},
                       widths=[ENTITY_WIDTH, SEMANTIC_WIDTH, CANDIDATE_WIDTH, CONTEXT_WIDTH, HISTORY_WIDTH])
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return {'feature_version': FEATURE_VERSION, 'fingerprint': digest, 'schema': payload}

    def _entity(self, obs, obj=None, player=None):
        hidden = obj is not None and (not obj.name or any(obj.properties.get(k)
                                  for k in ('hideInfo', 'faceDown', 'face_down')))
        card = lookup_visible(obj, self.registry) if obj is not None else UNKNOWN_CARD
        # Do not extract raw characteristics from face-down/hidden payloads.
        props = obj.properties if obj is not None and not hidden else {}
        controller = obj.controller if obj is not None else player.id
        side = 0 if controller == obs.actor else 1 if controller is not None else 2
        zone = obj.zone if obj is not None else 'player'
        row = onehot(zone if zone in ZONES else 'unknown', ZONES) + onehot(side, range(3))
        row += [float(player is not None), float(hidden)]
        row += onehot(card.coverage, tuple(Coverage)) + onehot(card.keyword_coverage, tuple(Coverage))
        row += [float(k in card.keywords) for k in Keyword]
        # XMage currently supplies card text rather than trustworthy keyword flags.
        # Only use explicit booleans when present; printed flags are separate.
        row += flags(props, [k.name for k in Keyword])
        row += sum((numeric(props.get(k)) for k in STATS), [])
        row += flags(props, FLAGS)
        row += [float(k in (props.get('cardTypes') or ())) for k in TYPES]
        row += [float('cardTypes' in props)]
        row += flags(props.get('color') or {}, COLORS)
        row += counter_features(player.counters if player is not None else props.get('counters'))
        if player is not None:
            row += numeric(player.life) + numeric(player.hand_count) + numeric(player.library_count)
            pool = player.native.get('manaPool') or {}
            row += sum((numeric(pool.get(k)) for k in COLORS), [])
        else:
            row += [0.] * (2 * (3 + len(COLORS)))
        assert len(row) == ENTITY_WIDTH
        return row, card.id, semantic_rows(card)

    def encode(self, obs):
        if obs.backend != 'xmage':
            raise UnsupportedInteraction('The first neural encoder supports XMage; MTGO bindings are not implemented')
        if len(obs.players) != 2:
            raise UnsupportedInteraction('The neural encoder requires two player views')
        if obs.terminal or obs.decision is None or not obs.decision.options:
            raise UnsupportedInteraction('A nonterminal decision with candidates is required')
        players = sorted(obs.players, key=lambda p: p.id != obs.actor)
        objects = tuple(obs.objects.values())
        handles = tuple(p.id for p in players) + tuple(o.id for o in objects)
        by_id = {h.value: i for i, h in enumerate(handles)}
        rows = [self._entity(obs, player=p) for p in players] + [self._entity(obs, obj=o) for o in objects]
        count = len(rows)
        semantic_count = max(1, max(len(r[2]) for r in rows))
        semantics = np.zeros((count, semantic_count, SEMANTIC_WIDTH), np.float32)
        semantic_mask = np.zeros((count, semantic_count), bool)
        for i, (_, _, records) in enumerate(rows):
            if records:
                semantics[i, :len(records)] = records
                semantic_mask[i, :len(records)] = True
        relations = np.zeros((count, count, 4), np.float32)
        for i, obj in enumerate(objects, start=2):
            if obj.controller and obj.controller.value in by_id:
                relations[i, by_id[obj.controller.value], 0] = 1
            # Parent/target links are identifiers, never text features.
            if not obj.name or any(obj.properties.get(k) for k in ('hideInfo', 'faceDown', 'face_down')):
                continue
            parent = obj.properties.get('parentId')
            if parent in by_id: relations[i, by_id[parent], 1] = 1
            targets = obj.properties.get('targets') or ()
            for target in targets:
                if isinstance(target, str) and target in by_id: relations[i, by_id[target], 2] = 1
        # Stack order comes from the gateway's ordered map, not arbitrary object IDs.
        stack = [i for i, o in enumerate(objects, start=2) if o.zone == 'stack']
        for first, second in zip(stack, stack[1:]): relations[first, second, 3] = 1
        decision = obs.decision
        def side(handle): return 2 if handle is None else int(handle != obs.actor)
        context = onehot(obs.phase if obs.phase in PHASES else 'unknown', PHASES)
        context += onehot(decision.kind if decision.kind in DECISIONS else 'unknown', DECISIONS)
        context += onehot(side(obs.active_player), range(3)) + onehot(side(obs.priority_player), range(3))
        context += numeric(obs.turn) + numeric(decision.native.get('min')) + numeric(decision.native.get('max'))
        context += numeric(len((decision.native.get('options') or {}).get('chosenTargets') or ()))
        metadata = prompt_metadata(decision.kind, decision.native)
        context += onehot(metadata['purpose'], PURPOSES)
        context += numeric(metadata.get('next_hand_size')) + numeric(metadata.get('bottom_remaining'))
        context += sum((numeric(metadata.get('mana_cost',{}).get(c)) for c in MANA_COLORS), [])
        context += numeric(metadata.get('life_payment'))
        target_zone = (decision.native.get('options') or {}).get('targetZone')
        context += onehot(target_zone if target_zone in TARGET_ZONES else 'unknown',TARGET_ZONES)
        context += flags(decision.native,('flag',)) if decision.kind=='GAME_TARGET' else [0.,0.]
        context += hand_summary(obs)
        context += [1., 0., 0.]  # action, integer digit, distribution digit
        assert len(context) == CONTEXT_WIDTH
        candidates = []
        references = np.zeros((len(decision.options), count), np.float32)
        playable = (obs.native.get('canPlayObjects') or {}).get('objects') or {}
        for i, option in enumerate(decision.options):
            if option.constraints.get('action_kind')=='spend_mana':object_id=option.constraints.get('source')
            elif option.kind=='uuid':object_id=option.constraints.get('source') or option.id.removeprefix('object:')
            elif option.kind=='boolean':object_id=prompt_source(decision.native)
            else:object_id=None
            if object_id in by_id: references[i, by_id[object_id]] = 1
            role, coverage, ability = self._binding(obs, option, object_id, playable)
            candidates.append(candidate_row(option.kind, role, option.constraints, coverage=coverage, ability=ability))
        arrays = dict(entities=np.asarray([r[0] for r in rows], np.float32),
            card_ids=np.asarray([r[1] for r in rows], np.int32),
            entity_mask=np.ones(count, bool), semantics=semantics, semantic_mask=semantic_mask,
            relations=relations, context=np.asarray(context, np.float32),
            candidates=np.asarray(candidates, np.float32), candidate_refs=references,
            candidate_mask=np.ones(len(candidates), bool),
            selection=np.zeros(CANDIDATE_WIDTH, np.float32))
        from .history import HISTORY_WIDTH
        arrays.update(history=np.zeros((1, HISTORY_WIDTH), np.float32),
                      history_cards=np.zeros(1, np.int32), history_mask=np.zeros(1, bool))
        return EncodedDecision(arrays, decision.options, decision.token, handles)

    def _binding(self, obs, option, object_id, playable):
        if option.constraints.get('action_kind')=='spend_mana':
            from ..semantics import Ability,Cost,Quantity
            color=Color.__members__.get(option.constraints.get('mana_color'))
            if (color not in (Color.WHITE,Color.BLUE,Color.BLACK,Color.RED,Color.GREEN,Color.COLORLESS)
                    or object_id!=obs.actor.value or obs.decision.kind!='GAME_PLAY_MANA'):
                return 'unknown',Coverage.UNKNOWN,None
            # Existing numeric columns describe consumption rather than mana
            # production. This protocol action is not a registered card ability;
            # its ID encodes as unknown. The quantity unlocked is not asserted.
            descriptor=Ability(9000+int(color),AbilityKind.UNKNOWN,
                costs=(Cost(CostKind.MANA,Quantity(),color),),coverage=Coverage.PARTIAL)
            return 'mana',Coverage.PARTIAL,descriptor
        if option.constraints.get('action_kind') == 'cancel':
            return 'cancel', Coverage.COMPLETE, None
        if 'semantic_ability' in option.constraints:
            # Metadata is emitted by the finite protocol decoder. Revalidate
            # against visible source identities and this encoder's registry.
            for handle, obj in obs.objects.items():
                if option.constraints.get('source') and handle.value != option.constraints['source']: continue
                card = lookup_visible(obj, self.registry)
                if not card.id or card.id != option.constraints.get('semantic_card'): continue
                ability = next((a for a in card.abilities if a.id == option.constraints['semantic_ability']), None)
                if ability:
                    role = 'mana' if ability.kind == AbilityKind.MANA else 'cast' if ability.kind == AbilityKind.SPELL else 'ability'
                    return role, ability.coverage, ability
            return 'unknown', Coverage.UNKNOWN, None
        if option.constraints.get('selection_kind') == 'trigger_order':
            obj = next((o for h,o in obs.objects.items() if h.value == object_id), None)
            card = lookup_visible(obj, self.registry) if obj else UNKNOWN_CARD
            triggers = [a for a in card.abilities if a.kind == AbilityKind.TRIGGERED]
            if len(triggers) == 1: return 'ability', triggers[0].coverage, triggers[0]
            return 'ability', Coverage.UNKNOWN, None
        if option.id in ('yes', 'no', 'special'):
            return option.id, Coverage.COMPLETE, None
        if option.kind == 'uuid' and obs.decision.kind == 'GAME_TARGET':
            return 'target', Coverage.COMPLETE, None
        stats = playable.get(object_id) or {}
        categories = [k for k in ('basicPlayAbilities', 'basicCastAbilities', 'basicManaAbilities', 'other') if stats.get(k)]
        if obs.decision.kind in ('GAME_PLAY_MANA', 'GAME_PLAY_XMANA'):
            categories = [k for k in categories if k == 'basicManaAbilities']
            if not categories:
                obj = next((o for h,o in obs.objects.items() if h.value==object_id),None)
                card = lookup_visible(obj,self.registry) if obj else UNKNOWN_CARD
                mana = [a for a in card.abilities if a.kind==AbilityKind.MANA]
                if len(mana)==1:return 'mana',mana[0].coverage,mana[0]
                if mana:return 'mana',Coverage.PARTIAL,None
        if len(categories) != 1:
            return ('ability' if obs.decision.kind == 'GAME_CHOOSE_ABILITY' else 'unknown'), Coverage.UNKNOWN, None
        category = categories[0]
        role = {'basicPlayAbilities': 'play_land', 'basicCastAbilities': 'cast', 'basicManaAbilities': 'mana', 'other': 'ability'}.get(category, 'unknown')
        if role == 'play_land': return role, Coverage.COMPLETE, None
        obj = next((o for h, o in obs.objects.items() if h.value == object_id), None)
        card = lookup_visible(obj, self.registry) if obj else UNKNOWN_CARD
        kind = AbilityKind.MANA if role == 'mana' else AbilityKind.ACTIVATED if role == 'ability' else AbilityKind.SPELL
        abilities = [a for a in card.abilities if a.kind == kind]
        # Identity can be known even when some of the card's rules are partial.
        if len(abilities) == 1 and len(stats[category]) == 1:
            return role, abilities[0].coverage, abilities[0]
        return role, Coverage.UNKNOWN, None


def candidate_row(kind, role, constraints=None, *, coverage=Coverage.UNKNOWN, ability=None,
                  number=None, position=None, prefix=None):
    constraints = constraints or {}
    row = onehot(kind if kind in OPTION_KINDS else 'unknown', OPTION_KINDS) + onehot(role, ROLES)
    row += onehot(coverage, tuple(Coverage))
    row += [float(constraints.get('legality') == 'advertised'), float('legality' in constraints)]
    row += numeric(constraints.get('minimum')) + numeric(constraints.get('maximum')) + numeric(number)
    row += numeric(position) + numeric(prefix) + onehot(ability.kind if ability else AbilityKind.UNKNOWN, tuple(AbilityKind))
    row += onehot(ability.id if ability and ability.id in ABILITY_IDS else 0, ABILITY_IDS)
    # Fixed-width summary of this selected ability's typed costs/effects. The
    # identity retains distinctions between ordered records; full ordered card
    # semantics are also available through the source entity reference.
    records = semantic_rows(replace(UNKNOWN_CARD, abilities=(ability,))) if ability else []
    row += np.sum(records, axis=0).tolist() if records else [0.] * SEMANTIC_WIDTH
    assert len(row) == CANDIDATE_WIDTH
    return row


def collate(items):
    """Pad variable entity, semantic and candidate axes without truncation."""
    if not items: raise ValueError('Cannot collate an empty batch')
    n = max(len(x['entities']) for x in items)
    a = max(len(x['candidates']) for x in items)
    s = max(x['semantics'].shape[1] for x in items)
    h = max(len(x['history']) for x in items)
    from .history import HISTORY_WIDTH
    shapes = {'entities': (n, ENTITY_WIDTH), 'card_ids': (n,), 'entity_mask': (n,),
        'semantics': (n, s, SEMANTIC_WIDTH), 'semantic_mask': (n, s), 'relations': (n, n, 4),
        'context': (CONTEXT_WIDTH,), 'candidates': (a, CANDIDATE_WIDTH),
        'candidate_refs': (a, n), 'candidate_mask': (a,), 'selection': (CANDIDATE_WIDTH,),
        'history': (h, HISTORY_WIDTH), 'history_cards': (h,), 'history_mask': (h,)}
    batch = {key: np.zeros((len(items), *shape), dtype=items[0][key].dtype) for key, shape in shapes.items()}
    for i, item in enumerate(items):
        if not np.any(item['candidate_mask']): raise ValueError('Decision has no allowed candidate')
        for key, value in item.items():
            if key in batch: batch[key][(i, *[slice(0, d) for d in value.shape])] = value
    return batch
