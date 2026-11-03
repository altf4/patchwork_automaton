"""Additional visible-trigger encoding on the current Standard schema.

The shared choice helper uses only cards exposed in the actor's legal pools. It
does not add card-name IDs; cards with identical public traits can still be
indistinguishable to the model.
"""
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
from typing import NamedTuple

import numpy as np

from ..core import Handle
from .features import numeric
from .rustmage_features import visible_identity
from .standard_features import StandardEncoder


CHOICE_POOLS = (
    ('search_choice', 'cards'),
    ('library_choice', 'remaining'),
    ('library_selection', 'remaining'),
)
CHOICE_ACTIONS = frozenset((
    'ChooseSearch', 'ChooseSearchBattlefield', 'SelectLibrary',
    'OrderLibraryBottom', 'ArrangeLibrary',
))


class _ChoiceObject(NamedTuple):
    id: Handle
    name: str
    zone: str
    controller: Handle
    properties: Mapping


def _choice_objects(obs, choices):
    """Create entity rows for legal cards that the current choice reveals."""
    legal_ids = {action.get('card') for _, action in choices
                 if action['kind'] in CHOICE_ACTIONS and type(action.get('card')) is int}
    if not legal_ids:
        return {}
    existing = {int(obj.id.value) for obj in obs.objects.values()}
    additions = {}
    for context, field in CHOICE_POOLS:
        pool = obs.native.get(context)
        if not isinstance(pool, Mapping):
            continue
        for card in pool.get(field) or ():
            if not isinstance(card, Mapping):
                continue
            identifier = card.get('id')
            name = visible_identity(card)
            controller = card.get('controller')
            zone = card.get('zone')
            if (type(identifier) is not int or identifier not in legal_ids
                    or identifier in existing or identifier in additions
                    or not isinstance(name, str) or controller not in ('P0', 'P1')
                    or not isinstance(zone, str)):
                continue
            handle = Handle('rustmage', 'object', str(identifier))
            additions[identifier] = _ChoiceObject(handle, name, zone.lower(),
                Handle('rustmage', 'player', controller), card)
    return additions


def _with_visible_objects(obs, additions):
    if not additions:
        return obs
    objects = dict(obs.objects)
    objects.update({item.id: item for item in additions.values()})
    # A NativeObservation can otherwise return its cached v5 arrays. Materialize
    # the same player-private JSON tree so the Python encoder sees new entities.
    raw = dict(obs.native) if hasattr(obs.native, 'native_choices') else obs.native
    if hasattr(obs, '_replace'):
        return obs._replace(objects=objects, native=raw)
    return replace(obs, objects=objects, native=raw)


def _visible_trigger_ability(raw, action):
    if action['kind'] != 'PutTrigger':
        return None
    for trigger in raw.get('waiting_triggers') or ():
        if trigger.get('id') != action.get('id'):
            continue
        source_identity = trigger.get('source') or {}
        source_card = source_identity.get('card', trigger.get('source_card'))
        source_id = (source_identity.get('object') or {}).get('card', trigger.get('card'))
        if (any(trigger.get(flag) for flag in ('hideInfo', 'faceDown', 'face_down'))
                or not isinstance(source_card, str)):
            return None
        source = next((obj for obj in raw.get('objects') or ()
                       if obj.get('id') == source_id), None)
        if source is not None and visible_identity(source) is None:
            return None
        kind = trigger.get('kind') or {}
        detail = kind.get('TriggeredAbility') if isinstance(kind, Mapping) else None
        ability = detail.get('ability') if isinstance(detail, Mapping) else None
        return ability if type(ability) is int else None
    return None


class StandardChoiceEncoder(StandardEncoder):
    """v6 candidate encoder; opt in only when starting a new training run."""

    def encode(self, obs, *, choices=None):
        if choices is None and obs.decision is not None:
            choices = self.expand_options(obs.decision)
        additions = _choice_objects(obs, choices or ())
        encoded = super().encode(_with_visible_objects(obs, additions), choices=choices)
        role_width = 1 + self.feature_spec['semantics'] + self.feature_spec['entities']
        offset = len(self.actions) + 3 * role_width
        candidates = None
        for index, (_, action) in enumerate(choices or ()):
            ability = _visible_trigger_ability(obs.native, action)
            if ability is None:
                continue
            if candidates is None:
                candidates = encoded.arrays['candidates'].copy()
            candidates[index, offset:offset + 2] = np.asarray(numeric(ability), np.float32)
        if candidates is not None:
            encoded.arrays = dict(encoded.arrays, candidates=candidates)
        return encoded

    def manifest(self):
        result = super().manifest()
        result.update(feature_version=6,
            choice_local_visible_card_pools=CHOICE_POOLS,
            visible_trigger_ability=True)
        result.pop('fingerprint')
        result['fingerprint'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result
