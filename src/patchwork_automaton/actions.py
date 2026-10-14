"""Encode caller-selected actions. No strategy or card-specific rules."""
from dataclasses import dataclass
from collections import Counter
import struct

from .protocol import ProtocolError, array
from .properties import PROPERTY_NAMES, get_property
from .interactions import PromptCode
from .errors import UnsupportedInteraction

REVERSE = {name: int(key) for key, name in PROPERTY_NAMES.items()}


def action_entries(properties):
    for name, value in properties.items():
        code = REVERSE.get(name, -1)
        if name.startswith('attribute_'):
            code = int(name.removeprefix('attribute_'))
        if 202329600 <= code <= 202329855 and isinstance(value, dict):
            yield code, value


@dataclass(frozen=True)
class PromptChoice:
    code: int
    parameter: int = 0
    extra: bool = False


@dataclass(frozen=True)
class CardChoice:
    card_id: int
    action_id: int
    targets: tuple[int, ...] = ()
    modes: bytes = b''
    extra_mana: int = 0
    flags: int = 1
    target_count: int | None = None


@dataclass(frozen=True)
class SelectorChoice:
    selector_id: int
    value: int


@dataclass(frozen=True)
class ManaChoice:
    color: PromptCode
    mana_id: int = 0


@dataclass(frozen=True)
class CardSelection:
    card_id: int
    action_id: int
    target_groups: tuple[tuple[int, ...], ...] = ()
    modes: tuple[int, ...] | None = None
    distributions: tuple[tuple[int, ...], ...] = ()
    retain_priority: bool = False
    allow_non_game: bool = False


@dataclass(frozen=True)
class DamageChoice:
    source_id: int
    assignments: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class OrderingChoice:
    source_id: int
    targets: tuple[int, ...]


def card_payload(card_id, action_id, timestamp, targets=(), *, modes=b'', extra_mana=0, flags=1, target_count=None):
    target_count = len(targets) if target_count is None else target_count
    if not 0 <= target_count <= 255 or len(targets) > 511 or len(modes) > 255:
        raise ProtocolError('Too many targets or modes')
    if not 0 <= extra_mana <= 255 or not 0 <= flags <= 255:
        raise ProtocolError('Action byte out of range')
    return (struct.pack('<ii', card_id, action_id) + array(modes)
            + struct.pack('<IBBBB', timestamp, target_count, extra_mana, flags, 0)
            + struct.pack('<i', len(targets))
            + b''.join(struct.pack('<I', target) for target in targets))


def damage_payload(source_id, assignments, *, ordering=False):
    if len(assignments) > 65535:
        raise ProtocolError('Too many damage assignments')
    buffer = b''.join(struct.pack('<II', target & 0xffffffff, amount & 0xffffffff) for target, amount in assignments)
    return struct.pack('<iHHi', source_id, len(assignments), 0 if ordering else 1,
                       len(assignments) * 2) + buffer


def selected_groups(action, modes):
    spec = action.modes
    if spec is None:
        if modes:
            raise ValueError('This action has no modes')
        return b'', action.target_groups
    if spec.fixed_mapping is not None:
        if modes is not None and ''.join(str(i) for i in sorted(modes)) != spec.fixed_mapping:
            raise ValueError('Choose the advertised action matching this fixed mode combination')
        return spec.fixed_mapping.encode('ascii'), action.target_groups
    counts = Counter(modes or ())
    options = {option.index: option for option in spec.options}
    if any(i not in options or i > 9 or i < 0 for i in counts):
        raise ValueError('Invalid mode index')
    if any(n > options[i].maximum for i, n in counts.items()):
        raise ValueError('Mode repetition exceeds advertised capacity')
    weight = sum(n * options[i].weight for i, n in counts.items())
    if not spec.minimum <= weight <= spec.maximum:
        raise ValueError('Selected mode weight is outside advertised limits')
    groups = [g for g in action.target_groups if g.related_mode == 0]
    for i in sorted(counts):
        mode_groups = [g for g in action.target_groups if g.related_mode == options[i].property_id]
        for _ in range(counts[i]):
            groups.extend(mode_groups)
    return ''.join(str(i) * counts[i] for i in sorted(counts)).encode('ascii'), tuple(groups)


def _requirement_check(observation, descriptor, selected, earlier, action):
    objects = observation.objects
    def props(target):
        obj = objects.get(target) or objects.get(target - 0x100000000 if target > 0x7fffffff else target)
        if obj is None:
            raise UnsupportedInteraction('Target metadata is missing for constraint validation')
        # A GamePlayer controls itself; its Thing metadata may omit CONTROLLER.
        result = dict(obj.properties)
        if 'CONTROLLER' not in result:
            result['CONTROLLER'] = obj.controller
        return result
    def values(key):
        return [props(target).get(key, 0) for target in selected]
    requirements = descriptor.requirements
    if 'TARGETS_MUST_HAVE_DIFFERENT_CONTROLLERS' in requirements:
        controllers = [props(target)['CONTROLLER'] for target in selected]
        if len(set(controllers)) != len(controllers):
            raise ValueError('Targets must have different controllers')
    if 'TARGETS_MUST_HAVE_SAME_CONTROLLER' in requirements:
        if len({props(target)['CONTROLLER'] for target in selected}) > 1:
            raise ValueError('Targets must have the same controller')
    if 'TARGETS_MUST_HAVE_EQUAL_TOUGHNESS' in requirements and len(set(values('TOUGHNESS'))) > 1:
        raise ValueError('Targets must have equal toughness')
    for requirement, field in (('TARGETTING_UPDATE_SUM_VALUES', 'TARGETTING_MAX_SUM_VALUE'),
                                ('TARGETTING_UPDATE_DIFFERING_VALUES', None)):
        if requirement not in requirements:
            continue
        code = action.attributes.get(requirement, descriptor.attributes.get(requirement, 0))
        numbers = [get_property(props(target), code, 0) for target in selected]
        if field and sum(numbers) > action.attributes.get(field, 0x7fffffff):
            raise ValueError('Target values exceed the advertised sum')
        if not field and len(set(numbers)) != len(numbers):
            raise ValueError('Target property values must differ')
    if 'TARGETING_UPDATE_SAME_VALUES' in requirements:
        code = action.attributes.get('TARGETING_UPDATE_SAME_VALUES', 0)
        bitmap = action.attributes.get('TARGETING_UPDATE_TARGETLIST_BITMAP', 0)
        prior = [target for i, group in enumerate(earlier) if bitmap & (1 << i) for target in group]
        if len({get_property(props(target), code, 0) for target in (*prior, *selected)}) > 1:
            raise ValueError('Target property values must match')
    for requirement, key in (('TARGETS_MUST_HAVE_DIFFERENT_NAMES','card_names'),
                              ('ALL_TARGETS_MUST_SHARE_CREATURE_TYPE','creature_types'),
                              ('ACTION_TARGETS_MUST_SHARE_CARD_TYPE','card_types'),
                              ('ACTION_TARGETS_MUST_SHARE_PERMANENT_TYPE','permanent_types')):
        if requirement not in requirements or len(selected) < 2:
            continue
        metadata = [props(target) for target in selected]
        if key == 'card_names':
            for item in metadata:
                if key not in item and 'CARDNAME_STRING' in item:
                    item[key] = (item['CARDNAME_STRING'],)
        if any(key not in p for p in metadata):
            raise UnsupportedInteraction(f'Load card definitions to validate {requirement}')
        if key == 'card_names':
            for i,item in enumerate(metadata):
                names = set(item[key])
                for earlier_item in metadata[:i]:
                    prior = set(earlier_item[key])
                    # Native client compares complete name sets (Rooms can
                    # have two names) and treats two nameless objects alike.
                    if (not names and not prior) or names and prior and (names <= prior or prior <= names):
                        raise ValueError('Targets must have different names')
        elif key == 'creature_types':
            first = metadata[0]
            def universal(item):
                return item.get('changeling') or item.get('all_creature_types')
            for item in metadata[1:]:
                if (not universal(first) and not first[key]
                        or not universal(item) and not item[key]):
                    raise ValueError('Targets do not have a creature type')
                if (not universal(first) and not universal(item)
                        and not set(first[key]) & set(item[key])):
                    raise ValueError('Targets do not share a creature type')
        elif any(not set(metadata[0][key]) & set(p[key]) for p in metadata[1:]):
            raise ValueError('Targets do not share the required type')


def encode_card_selection(observation, choice, action):
    modes, groups = selected_groups(action, choice.modes)
    if len(groups) != len(choice.target_groups):
        raise ValueError('Supply one selection for each active target group')
    if choice.distributions and len(choice.distributions) != len(groups):
        raise ValueError('Supply distribution entries for each target group')
    targets, amounts, earlier = [], [], []
    divided = any(g.distributed_total is not None for g in groups)
    source = observation.objects.get(action.card_id)
    distinct = bool(action.attributes.get('ACTION_TARGET_MUST_NOT_BE_PREVIOUS_TARGETS')
                    or source and source.properties.get('ACTION_TARGET_MUST_NOT_BE_PREVIOUS_TARGETS'))
    for i, (group, selected) in enumerate(zip(groups, choice.target_groups)):
        if not (group.deferred and not selected) and not group.minimum <= len(selected) <= group.maximum:
            raise ValueError('Target count is outside advertised limits')
        if len(set(selected)) != len(selected) or any(target not in group.legal_targets for target in selected):
            raise ValueError('Target is duplicated or not advertised as legal')
        if distinct and any(target in prior for target in selected for prior in earlier):
            raise ValueError('A previous target cannot be chosen again')
        _requirement_check(observation, group, selected, earlier, action)
        distribution = choice.distributions[i] if choice.distributions else ()
        if group.distributed_total is not None and not (group.deferred and not selected):
            if (len(distribution) != len(selected) or any(n < group.distribution_minimum for n in distribution)
                    or sum(distribution) != group.distributed_total):
                raise ValueError('Target distribution does not match the advertised total')
        elif distribution:
            raise ValueError('This group does not take a distribution')
        if i:
            targets.append(0xfffffffe); amounts.append(0xfffffffe)
        targets.extend(selected)
        amounts.extend(distribution if group.distributed_total is not None else (0 for _ in selected))
        earlier.append(selected)
    count = len(targets)
    if divided:
        targets.extend(amounts)
    return card_payload(choice.card_id, choice.action_id, observation.decision_id.timestamp, targets,
                        modes=modes, flags=(0 if choice.retain_priority else 1) | (2 if divided else 0),
                        target_count=count)


def legal_targets_for(observation, action, group_index, *, modes=None, chosen=()):
    """Filter a group's advertised candidates after partial caller selections.

    This mirrors descriptor constraints, not the Magic rules engine. Missing
    metadata needed for a relationship raises UnsupportedInteraction.
    """
    _, groups = selected_groups(action, modes)
    if not 0 <= group_index < len(groups) or len(chosen) > len(groups):
        raise ValueError('Invalid target group index or partial selection')
    group = groups[group_index]
    selected = chosen[group_index] if group_index < len(chosen) else ()
    if len(selected) >= group.maximum:
        return ()
    source = observation.objects.get(action.card_id)
    distinct = bool(action.attributes.get('ACTION_TARGET_MUST_NOT_BE_PREVIOUS_TARGETS')
                    or source and source.properties.get('ACTION_TARGET_MUST_NOT_BE_PREVIOUS_TARGETS'))
    candidates = []
    for target in group.legal_targets:
        if target in selected or distinct and any(target in values for values in chosen):
            continue
        try:
            _requirement_check(observation, group, (*selected, target), chosen[:group_index], action)
        except UnsupportedInteraction:
            raise
        except ValueError:
            continue
        candidates.append(target)
    return tuple(candidates)


def encode_choice(observation, choice, *, player_index=None):
    """Return message name, bytes, and a partial-submission key (or None)."""
    timestamp = observation.decision_id.timestamp
    if isinstance(choice, SelectorChoice):
        selector = next((s for s in observation.selectors if s.id == choice.selector_id), None)
        if selector is None or selector.kind == 'unknown':
            raise UnsupportedInteraction('Selector is not advertised or modeled')
        value = choice.value
        if selector.minimum is not None and not selector.minimum <= value <= selector.maximum:
            raise ValueError('Number is outside advertised limits')
        if selector.options and value not in {option.value for option in selector.options}:
            raise ValueError('Selection is not advertised')
        if selector.kind in ('card','player','color','pile') and not selector.options:
            raise ValueError('Selector has no enabled options')
        if selector.kind == 'list' and selector.except_values:
            if (value in selector.except_values) != selector.invert:
                raise ValueError('Selection violates the list filter')
        if not 0 <= value <= 0xffffffff:
            raise ValueError('Selection value must be uint32')
        code, parameter = (value, 0) if selector.kind == 'color' else (selector.code, value)
        return 'PromptActionMessage', struct.pack('<III', code, parameter,
                      0 if selector.kind == 'color' else timestamp), None
    if isinstance(choice, ManaChoice):
        color = int(choice.color)
        if color not in (16, 32, 64, 128, 256, 512):
            raise ValueError('Choose exactly one mana color')
        if not any(m['flags'] & color and (player_index is None or m['player'] in (player_index,255))
                   for m in (observation.prompt or {}).get('mana', ())):
            raise UnsupportedInteraction('Mana payment color is not advertised')
        if not 0 <= choice.mana_id <= 0xffffffff:
            raise ValueError('Mana entry ID must be uint32')
        if choice.mana_id and not any(m.id == choice.mana_id and m.player == player_index and m.amount > 0
                                      and m.color_flag == color
                                      for m in observation.mana):
            raise ValueError('Mana pool entry is unavailable')
        return 'PromptActionMessage', struct.pack('<III', color, choice.mana_id, 0), ('mana', color, choice.mana_id)
    if isinstance(choice, (CardChoice, CardSelection)):
        action = next((a for a in observation.actions if a.card_id == choice.card_id and a.action_id == choice.action_id), None)
        if action is None or action.non_game and not getattr(choice, 'allow_non_game', False):
            raise UnsupportedInteraction('Card action is not advertised as a game action')
        if isinstance(choice, CardSelection):
            payload = encode_card_selection(observation, choice, action)
        else:
            payload = card_payload(choice.card_id, choice.action_id, timestamp, choice.targets,
                modes=choice.modes, extra_mana=choice.extra_mana, flags=choice.flags,
                target_count=choice.target_count)
        return 'NewCardActionMessage', payload, ('card', choice.card_id, choice.action_id) if action.can_perform_locally or action.non_game else None
    if isinstance(choice, DamageChoice):
        assignment = next((a for a in observation.damage if a.source_id == choice.source_id), None)
        if assignment is None:
            raise UnsupportedInteraction('Damage assignment is not advertised')
        chosen = dict(choice.assignments)
        available = {target.id: target for target in assignment.targets}
        if len(chosen) != len(choice.assignments) or set(chosen) != set(available):
            raise ValueError('Supply each advertised damage recipient exactly once')
        if any(not available[target].minimum <= amount <= available[target].maximum for target, amount in chosen.items()):
            raise ValueError('Damage is outside recipient bounds')
        if sum(chosen.values()) != assignment.total:
            raise ValueError('Damage does not match advertised total')
        return 'DamageAssignmentMessage', damage_payload(choice.source_id, tuple((t.id, chosen[t.id])
            for t in assignment.targets)), ('damage', choice.source_id)
    if isinstance(choice, OrderingChoice):
        order = next((o for o in observation.ordering if o.source_id == choice.source_id), None)
        if order is None:
            raise UnsupportedInteraction('Ordering is not advertised')
        if Counter(choice.targets) != Counter(order.targets):
            raise ValueError('Ordering must be a permutation of advertised targets')
        return 'DamageAssignmentMessage', damage_payload(choice.source_id,
            tuple((target, i + 1) for i, target in enumerate(choice.targets)), ordering=True), ('ordering', choice.source_id)
    raise TypeError('Expected a supported game choice')
