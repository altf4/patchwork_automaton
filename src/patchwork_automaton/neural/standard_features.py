"""Numerical action support for the three Standard main decks.

Shares the catalogue trait encoder with Foundations. This explicit, separately
fingerprinted action schema also describes mana colors, library destinations,
entry choices and referenced objects. No card-name embedding or policy fallback.
"""
import hashlib
import json
from collections.abc import Mapping

from ..backends.rustmage import domain_action
from ..core import Response
from ..errors import UnsupportedInteraction
from .features import numeric, onehot
from .rustmage_features import (ACTION_FIELDS, DECISIONS, FEATURE_SPEC,
                               RustMageEncoder, canonical, entity_row)

BASICS = ('Plains', 'Island', 'Swamp', 'Mountain', 'Forest')
DESTINATIONS = ('Top', 'Bottom', 'Graveyard', 'Shuffle', 'SecondFromTop')
FACES = ('Normal', 'Adventure', 'Omen', 'Back')
FIELDS = dict(ACTION_FIELDS, **{
    'ActivateMana': ('card', 'ability', 'payment', 'counter_payment', 'production'),
    'AnnounceMana': ('card', 'ability', 'counter_payment', 'production'),
    'ChooseMana': ('production',), 'TapGrantedMana': ('card', 'grantor', 'production'),
    'AnnounceExile': ('card',), 'AnnounceAdventure': ('card',),
    'AnnounceBack': ('card',), 'AnnounceOmen': ('card',),
    'AnnouncePlot': ('card',), 'PayPlot': ('payment',), 'AnnouncePlotted': ('card',),
    'AnnounceFlashback': ('card',), 'AnnounceFlashSurcharge': ('card',),
    'AnnounceWarp': ('card',), 'AnnounceDisguise': ('card',),
    'AnnounceHarmonize': ('card',),
    'AnnounceFree': ('card', 'face', 'flash_surcharge'),
    'AnnounceExileFree': ('card', 'face', 'flash_surcharge'),
    'AnnounceExileLife': ('card', 'face', 'flash_surcharge'),
    'BeginFaceUp': ('object', 'cost'), 'PayFaceUp': ('payment',), 'AbortFaceUpPayment': (),
    'PlayGraveyardPermanent': ('card', 'permission', 'permanent_type'),
    'AnnounceFreeGraveyardPermanent': ('card', 'face', 'flash_surcharge', 'permission', 'permanent_type'),
    'AnnounceDisguiseGraveyard': ('card', 'permission', 'permanent_type'),
    'AnnounceFreeFromPermission': ('card', 'face', 'flash_surcharge', 'permission'),
    'AnnounceGrantedFlashback': ('card', 'flash_surcharge'),
    'ChooseEntryValue': ('value',), 'ChooseEntryLife': ('pay',),
    'ChooseEntryTimestamp': ('object',), 'ChooseEntryReplacement': ('source', 'intrinsic'),
    'ChooseManaRollback': ('keep',), 'ChooseOffspring': ('pay',),
    'ChooseCasualty': ('pay',), 'ChooseBargain': ('sacrifice',),
    'ChooseGift': ('beneficiary',), 'ChooseBlight': ('pay',),
    'ChooseTeamwork': ('pay',), 'ChooseAdditionalLife': (),
    'ChooseAdditionalCost': ('object',), 'ChooseCastCost': ('clause',),
    'ChooseSpellCost': ('cost',), 'ChooseActivationCost': ('cost',),
    'ChooseMode': ('mode',), 'ChooseX': ('value',),
    'ChooseSourceCounterAmount': ('amount',), 'AllocateTargetAmount': ('amount',),
    'ChooseSearch': ('card',), 'ChooseSearchBattlefield': ('card',),
    'FinishSearch': (), 'ChooseSearchZones': ('library', 'graveyard'),
    'ChooseHandLibrarySearchZones': ('hand', 'library'),
    'SelectLibrary': ('card',), 'FinishLibrarySelection': (),
    'OrderLibraryBottom': ('card',), 'ArrangeLibrary': ('card', 'destination'),
    'ChooseLibraryDestination': ('object', 'destination'),
    'ChooseHandCard': ('object',), 'ChooseRevealedDiscard': ('object',),
    'ChooseExplore': ('object', 'graveyard'), 'ChooseManifest': ('object',),
    'Sacrifice': ('object',), 'KeepLegend': ('object',), 'KeepRole': ('object',),
    'ChooseAuraAttachment': ('object',), 'ChooseLinkedExile': ('object',),
    'BeginOnBattlefield': ('object',), 'BeginOrderedCosts': (),
    'PayResolutionMana': ('payment',), 'DeclineResolutionMana': (),
    'PayResolutionLife': (), 'DeclineResolutionLife': (),
    'PayAttackCost': ('payment',), 'AbortAttackPayment': (),
    'CancelAttackDeclaration': (), 'DeclineEffectPayment': (),
    'CastExiledDuringResolution': ('object', 'face'), 'FinishResolutionCasts': (),
    'FinishGraveyardSymbolCost': (),
})
STANDARD_DECISIONS = DECISIONS + tuple(d for d in (
    'entry_value', 'entry_life', 'entry_timestamps', 'entry_replacement', 'aura_entry',
    'mana_choice', 'mana_rollback', 'spell_mode', 'spell_cost', 'activation_cost',
    'additional_cost', 'cast_costs', 'offspring', 'casualty', 'casualty_payment',
    'bargain', 'gift', 'x', 'resolution_x', 'source_counter_amount', 'target_amount',
    'library_search', 'library_search_destination', 'library_choice',
    'library_selection', 'hand_library_search', 'library_graveyard_search',
    'hand_card_choice', 'revealed_discard', 'effect_discard', 'explore', 'manifest',
    'legend_rule', 'role_rule', 'linked_exile', 'resolution_mana', 'effect_mana',
    'resolution_life', 'attack_cost', 'plot_payment', 'resolution_cast',
    'ordered_costs', 'graveyard_symbol_cost', 'blight_intent', 'blight_payment',
    'teamwork_intent', 'teamwork_payment',
    'effect_choice', 'retarget', 'resolution_casts', 'face_up_payment',
) if d not in DECISIONS)
BOOL_FIELDS = ('keep', 'graveyard', 'intrinsic', 'hand', 'library', 'flash_surcharge')
NUMBER_FIELDS = ('mode', 'cost', 'clause')
PERMANENT_TYPES = ('Creature', 'Artifact', 'Enchantment', 'Planeswalker', 'Land', 'Battle')
EXTRA_WIDTH = 12 + 2 * len(NUMBER_FIELDS) + 2 + len(BASICS) + len(DESTINATIONS) + 2 * len(BOOL_FIELDS) + len(FACES) + 14 + len(PERMANENT_TYPES)


def expand_standard_options(decision, *, allow_concede=False, limit=4096):
    """Enumerate complete small domains; guard valid choices, not Cartesian waste."""
    choices = {}

    def add(response, action):
        choices.setdefault(canonical(action), (response, action))
        if len(choices) > limit:
            raise UnsupportedInteraction('Standard candidate domain exceeds enumeration bound')

    def payments(required, available, generic, prefix=()):
        index = len(prefix)
        if index == 6:
            if generic == 0:
                yield prefix
            return
        capacity = sum(hi - lo for lo, hi in zip(required[index + 1:], available[index + 1:]))
        low = max(0, generic - capacity)
        high = min(generic, available[index] - required[index])
        for extra in range(low, high + 1):
            yield from payments(required, available, generic - extra, prefix + (required[index] + extra,))

    for option in decision.options:
        if option.kind == 'finite':
            action = option.constraints['action']
            if action['kind'] != 'Concede' or allow_concede:
                add(Response(option.id), action)
            continue
        c = option.constraints
        if option.id == 'domain:mana_payment':
            values = payments(c['required'], c['available'], c['generic'])
        elif option.id == 'domain:x_choice':
            values = (v for interval in c['intervals'] for v in range(interval['minimum'], interval['maximum'] + 1))
        elif option.id == 'domain:source_counter_choice':
            values = range(c['maximum'] + 1)
        elif option.id in ('domain:damage_assignment', 'domain:target_amount', 'domain:mana_lot_spend'):
            values = c['values'] if c.get('values') is not None else range(c['minimum'], c['maximum'] + 1)
        else:
            raise UnsupportedInteraction('Unsupported Standard domain: ' + option.id)
        for value in values:
            add(Response(option.id, value), domain_action(option, value))
    if not choices:
        raise UnsupportedInteraction('No playable Standard action')
    return tuple(choices.values())


class StandardEncoder(RustMageEncoder):
    action_fields, actions, decisions = FIELDS, tuple(FIELDS), STANDARD_DECISIONS
    decision_aliases = {'optional_search': 'optional_effect', 'attack_payment': 'attack_cost'}
    feature_spec = dict(FEATURE_SPEC,
        entities=FEATURE_SPEC['entities'] + len(BASICS),
        candidates=FEATURE_SPEC['candidates'] - len(ACTION_FIELDS) + len(FIELDS) + 3 * len(BASICS) + EXTRA_WIDTH,
        context=FEATURE_SPEC['context'] - len(DECISIONS) + len(STANDARD_DECISIONS))

    def expand_options(self, decision):
        return expand_standard_options(decision, allow_concede=self.allow_concede)

    def encode(self, obs, *, choices=None):
        # The 250K schema uses the equivalent optional-object Bargain columns
        # for Harmonize; keep the original responses for game application.
        if obs.backend == 'rustmage' and obs.decision is not None and obs.native['decision'] == 'harmonize':
            if choices is None:
                choices = self.expand_options(obs.decision)
            adapted = []
            for response, action in choices:
                if action.get('kind') != 'ChooseHarmonizer' or set(action) != {'kind', 'object'}:
                    raise UnsupportedInteraction('Unexpected Harmonize action: ' + repr(action))
                adapted.append((response, {'kind': 'ChooseBargain', 'sacrifice': action['object']}))
            obs = obs._replace(native=dict(obs.native, decision='bargain'))
            return super().encode(obs, choices=tuple(adapted))
        return super().encode(obs, choices=choices)

    @staticmethod
    def entity_row(obj, side, *, player=False):
        entry = obj.get('entry_choice') or {}
        return entity_row(obj, side, player=player) + onehot(entry.get('BasicLandType'), BASICS)

    def action_roles(self, action):
        roles = dict(action)
        obj = action.get('object') or action.get('sacrifice') or action.get('source')
        if obj is not None:
            roles['card'] = obj['card']
        if action.get('beneficiary') is not None:
            roles['target'] = {'Player': action['beneficiary']}
        if action.get('permission') is not None:
            roles['attacker'] = action['permission']['card']
        return roles

    def action_extras(self, action):
        value = action.get('value')
        basic = None
        if isinstance(value, Mapping):
            if set(value) != {'BasicLandType'} or value['BasicLandType'] not in BASICS:
                raise UnsupportedInteraction('Unsupported entry choice: ' + repr(value))
            basic, value = value['BasicLandType'], None
        face = action.get('face')
        if face is not None and face not in FACES:
            raise UnsupportedInteraction('Unsupported cast face: ' + repr(face))
        cost = action.get('cost')
        cost = cost if isinstance(cost, Mapping) else {}
        return (sum((numeric(v) for v in action.get('production', (None,) * 6)), [])
                + sum((numeric(action.get(k)) for k in NUMBER_FIELDS), []) + numeric(value)
                + onehot(basic, BASICS) + onehot(action.get('destination'), DESTINATIONS)
                + sum((numeric(int(action[k]) if k in action else None) for k in BOOL_FIELDS), [])
                + onehot(face, FACES)
                + sum((numeric(v) for v in cost.get('colored', (None,) * 6)), [])
                + numeric(cost.get('generic'))
                + onehot(action.get('permanent_type'), PERMANENT_TYPES))

    def manifest(self):
        result = super().manifest()
        result.update(feature_version=5, decision_aliases=self.decision_aliases, scope='Standard Landfall, Prowess and Lifegain main decks',
                      extra_action_fields=dict(production=6, numbers=NUMBER_FIELDS,
                          booleans=BOOL_FIELDS, basics=BASICS, destinations=DESTINATIONS,
                          faces=FACES, dynamic_cost=('colored', 'generic'), permanent_types=PERMANENT_TYPES), live_entry_basic_types=BASICS,
                      observation_pipeline='python')
        result.pop('fingerprint')
        result['fingerprint'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result
