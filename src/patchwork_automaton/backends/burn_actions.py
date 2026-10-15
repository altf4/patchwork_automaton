"""Finite XMage display-rule bindings for the reviewed Burn deck.

This is a protocol adapter, not a text model. Only exact reviewed rules (after
removing the UI's numbered prefix) match. Unknown rules remain unbound.
Native ability IDs and the preceding object click locate instances when possible.
"""
from collections.abc import Mapping
import re

from ..burn_semantics import CARD_REGISTRY
from ..semantics import AbilityKind, CostKind, lookup_visible


def rules():
    result = {}
    def add(name, index, *labels):
        for label in labels:
            result[name, label] = CARD_REGISTRY[name].abilities[index].id
    for name in ('Mountain', 'Elegant Parlor', 'Inspiring Vantage', 'Sacred Foundry'):
        add(name, 0, '{T}: Add {R}.')
        if name != 'Mountain': add(name, 1, '{T}: Add {W}.')
    for i, color in enumerate(('R', 'W')):
        add('Sunbaked Canyon', i, '{T}, Pay 1 life: Add {' + color + '}.')
    add('Sunbaked Canyon', 2, '{1}, {T}, Sacrifice Sunbaked Canyon: Draw a card.')
    for name, types in (('Arid Mesa', 'Mountain or Plains'), ('Bloodstained Mire', 'Swamp or Mountain')):
        add(name, 0, f'{{T}}, Pay 1 life, Sacrifice {name}: Search your library for a {types} card, put it onto the battlefield, then shuffle.')
    add('Barbarian Ring', 0, '{T}: Add {R}. Barbarian Ring deals 1 damage to you.')
    add('Barbarian Ring', 1, '{R}, {T}, Sacrifice Barbarian Ring: It deals 2 damage to any target. Activate only if seven or more cards are in your graveyard.',
        '{R}, {T}, Sacrifice Barbarian Ring: Barbarian Ring deals 2 damage to any target. Activate only if seven or more cards are in your graveyard.')
    for i, label in enumerate(('Boros Charm deals 4 damage to target player or planeswalker.',
                              'Permanents you control gain indestructible until end of turn.',
                              'Target creature gains double strike until end of turn.')):
        add('Boros Charm', i, label)
    add('Roiling Vortex', 3, "{R}: Your opponents can't gain life this turn.")
    # Spell names are exact UI identities; alternate/fused casts are separate.
    for name, card in CARD_REGISTRY.items():
        if name in ('Raging Goblin', 'Goblin Piker'): continue
        spells = [a for a in card.abilities if a.kind == AbilityKind.SPELL]
        if len(spells) == 1:
            result[name, 'Cast ' + name] = spells[0].id
    return result


RULES = rules()


def mana_click_available(obj):
    """Reject provably unavailable reviewed mana sources, not bad strategies.

    Unknown cards retain server-validated click behavior. In particular, this
    does not reject damage-dealing lands or infer that losing life is bad.
    """
    card = lookup_visible(obj)
    if not card.id: return True
    abilities = [a for a in card.abilities if a.kind == AbilityKind.MANA]
    if not abilities: return False
    return any(not (obj.properties.get('tapped') and any(c.kind == CostKind.TAP_SOURCE for c in a.costs))
               for a in abilities)


def bind_ability(value, text, view, objects, source=None):
    label = re.sub(r'^\d+\.\s*', '', str(text)).strip()
    if label == 'Cancel': return {'action_kind': 'cancel'}
    visible = {h.value: o for h, o in objects.items() if lookup_visible(o).id}
    source_card = source.get('card') if isinstance(source, Mapping) else None
    source = source.get('id') if isinstance(source, Mapping) else source
    # Prefer an engine-provided ability-to-object association over click context.
    matches = set()
    for object_id, stats in ((view.get('canPlayObjects') or {}).get('objects') or {}).items():
        for category in ('basicPlayAbilities', 'basicCastAbilities', 'basicManaAbilities', 'other'):
            for entry in stats.get(category) or ():
                if isinstance(entry, Mapping) and str(entry.get('id')) == str(value): matches.add(object_id)
    if len(matches) == 1:
        source = next(iter(matches))
        source_card = None
    elif len(matches) > 1: return {}
    if source in visible:
        candidates = [visible[source]]
    elif source_card:
        # Casting can replace the clicked hand-card UUID with a stack UUID.
        candidates = [o for o in visible.values() if lookup_visible(o).id == source_card and o.zone == 'stack']
    else:
        candidates = list(visible.values()) if source is None else []
    found = []
    for obj in candidates:
        card = lookup_visible(obj)
        ability_id = RULES.get((obj.name, label))
        if ability_id is not None:
            found.append((obj.id.value, card.id, ability_id))
    identities = {(card, ability) for _, card, ability in found}
    if len(identities) != 1: return {}  # Never choose by option position or UUID.
    card, ability = next(iter(identities))
    result = {'semantic_card': card, 'semantic_ability': ability}
    if len(found) == 1: result['source'] = found[0][0]
    return result
