"""Optional local names/types needed by server list selectors.

    Card-name list selectors send name tokens, while card/wish selectors send
    texture numbers. Keeping those namespaces separate prevents silent mistakes.
"""
import json
from pathlib import Path

from .interactions import NamedOption, SelectionSource
from .properties import PROPERTY_IDS, freeze, indexed


def type_label(value):
    return value.removeprefix('CREATURETYPE_').replace('_', ' ').title()


class ChoiceCatalog:
    def __init__(self, *, definitions=None, sources=None, allowed_name_tokens=None):
        self.definitions = freeze({int(key): value for key, value in (definitions or {}).items()})
        defaults = json.loads(Path(__file__).with_name('data').joinpath('interaction-enums.json').read_text())
        defaults.update(sources or {})
        self.sources = {int(key): tuple(NamedOption(int(value), type_label(label) if int(key) == 32768 else label) for value, label in options)
                        for key, options in defaults.items()}
        self.allowed_name_tokens = None if allowed_name_tokens is None else frozenset(allowed_name_tokens)

    @classmethod
    def from_json(cls, path, *, allowed_name_tokens=None):
        data = json.loads(Path(path).read_text())
        return cls(definitions=data.get('definitions'), sources=data.get('sources'),
                   allowed_name_tokens=allowed_name_tokens)

    def properties(self, texture):
        return self.definitions.get(texture, {})

    def state_properties(self, raw):
        # Match the native definition lookup. MORPHED_CARD_TEXTURE_NUMBER and
        # OTHER_FACE describe related/hidden faces, not the current definition.
        texture = raw.get('CARDTEXTURE_NUMBER') or raw.get('EFFECT_TEXTURE_NUMBER')
        result = dict(self.properties(texture))
        result.update(raw)
        if 'creature_types' in result:
            result['creature_types'] = tuple(type_label(t) for t in result['creature_types'])
        types = {'IS_ARTIFACT':'Artifact', 'IS_BATTLE':'Battle', 'IS_CREATURE':'Creature',
                 'IS_ENCHANTMENT':'Enchantment', 'IS_LAND':'Land', 'IS_PLANESWALKER':'Planeswalker',
                 'INSTANT':'Instant', 'SORCERY':'Sorcery', 'IS_KINDRED':'Kindred',
                 'IS_PLANE':'Plane', 'IS_PHENOMENON':'Phenomenon', 'IS_VANGUARD':'Vanguard',
                 'IS_CONSPIRACY':'Conspiracy'}
        if any(key in raw for key in types):
            result['card_types'] = tuple(label for key, label in types.items() if raw.get(key))
            result['permanent_types'] = tuple(label for label in result['card_types']
                if label in ('Artifact','Battle','Creature','Enchantment','Land','Planeswalker'))
        ranged = indexed(raw, 605139200)
        if ranged:
            labels = {o.value:o.label for o in self.options(32768)}
            values = ['*' if value == PROPERTY_IDS['GROUPMATCH_WILDCARD'] else labels.get(value) for _,value in ranged]
            if all(value is not None for value in values):
                result['creature_types'] = tuple(values)
            else:
                result.pop('creature_types', None)
        # These are server state flags, not a rules simulation from card text.
        face_down = any(raw.get(key) for key in ('FACE_DOWN','FACE_DOWN_BECAUSE_MORPH',
            'FACE_DOWN_BECAUSE_TEZZERET','FACE_DOWN_BECAUSE_YEDORA','MANIFEST_FACE_DOWN',
            'FACE_DOWN_BECAUSE_CLOAK','FACE_DOWN_BECAUSE_DISGUISE','FORETOLD_ORDINAL'))
        result['face_down'] = face_down
        result['changeling'] = bool(result.get('CHANGELING') and not raw.get('REMOVE_ALL_TEXT') and not face_down)
        result['all_creature_types'] = bool(raw.get('GROUPMATCH_WILDCARD') or '*' in result.get('creature_types', ()))
        if result['changeling'] and 'creature_types' not in result:
            result['creature_types'] = ()
        if face_down:
            result['card_names'] = ()
            if not ranged:
                result['creature_types'] = ()
                result['all_creature_types'] = False
        elif result.get('IS_ROOM') and raw.get('ZONE') in (1000961, 1001728):
            # Locked doors have no name on the battlefield; on stack only the
            # chosen half contributes. Missing half definitions remain explicit.
            names = []
            battlefield = raw['ZONE'] == 1000961
            for side in ('LEFT', 'RIGHT'):
                if raw.get(side+'_UNLOCKED' if battlefield else 'CHOSE_'+side):
                    half = result.get(side+'_HALF_ATTRIBUTES_LIST')
                    if not hasattr(half, 'get'):
                        break
                    if 'CARDNAME_STRING' not in half:
                        break
                    names.append(half['CARDNAME_STRING'])
            else:
                result['card_names'] = tuple(names)
        elif 'CARDNAME_STRING' in result:
            result['card_names'] = (result['CARDNAME_STRING'],)
        return result

    def options(self, source):
        if source in self.sources:
            return self.sources[source]
        if source & 3 == 3:
            flags = {4:'IS_ARTIFACT', 8:'IS_BATTLE', 16:'IS_CREATURE', 32:'IS_ENCHANTMENT',
                     64:'INSTANT', 128:'IS_LAND', 256:'IS_PLANESWALKER', 512:'SORCERY', 1024:'IS_KINDRED'}
            return tuple(NamedOption(PROPERTY_IDS[name], name.replace('IS_', '').title())
                         for flag, name in flags.items() if source & flag and name in PROPERTY_IDS)
        conditions = {
            SelectionSource.CARD_NAME: lambda p: True,
            SelectionSource.ARTIFACT_NAME: lambda p: p.get('ARTIFACT') or p.get('IS_ARTIFACT'),
            SelectionSource.CREATURE_NAME: lambda p: p.get('CREATURE') or p.get('IS_CREATURE'),
            SelectionSource.NONCREATURE_NONLAND_NAME: lambda p: not (p.get('CREATURE') or p.get('IS_CREATURE')) and not p.get('IS_LAND'),
            SelectionSource.NONBASIC_NAME: lambda p: not (p.get('BASIC') or p.get('IS_BASICLAND')),
            SelectionSource.NONLAND_NONARTIFACT_NAME: lambda p: not p.get('IS_LAND') and not (p.get('ARTIFACT') or p.get('IS_ARTIFACT')),
            SelectionSource.LAND_NAME: lambda p: p.get('IS_LAND'),
            SelectionSource.NONLAND_NAME: lambda p: not p.get('IS_LAND'),
            SelectionSource.NONBASIC_LAND_NAME: lambda p: p.get('IS_LAND') and not (p.get('BASIC') or p.get('IS_BASICLAND')),
        }
        if source == SelectionSource.BASIC_LAND:
            by_name = {}
            for texture, props in sorted(self.definitions.items()):
                if (props.get('BASIC') or props.get('IS_BASICLAND')) and props.get('IS_LAND') and not props.get('COLORLESS'):
                    by_name[props.get('CARDNAME_STRING', str(texture))] = NamedOption(texture, props.get('CARDNAME_STRING', str(texture)))
            return tuple(by_name.values())
        if source not in conditions:
            return ()
        by_token = {}
        for props in self.definitions.values():
            token = props.get('CARDNAME_TOKEN')
            if token is None or self.allowed_name_tokens is not None and token not in self.allowed_name_tokens:
                continue
            if conditions[source](props):
                by_token[token] = NamedOption(token, props.get('CARDNAME_STRING', str(token)))
        return tuple(sorted(by_token.values(), key=lambda option: option.label))

    def wish_options(self):
        return tuple(NamedOption(texture, props.get('CARDNAME_STRING', str(texture)))
                     for texture, props in self.definitions.items()
                     if self.allowed_name_tokens is None or props.get('CARDNAME_TOKEN') in self.allowed_name_tokens)
