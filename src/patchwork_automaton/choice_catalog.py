"""Optional local names/types needed by server list selectors.

    Card-name list selectors send name tokens, while card/wish selectors send
    texture numbers. Keeping those namespaces separate prevents silent mistakes.
"""
import json
from pathlib import Path

from .interactions import NamedOption, SelectionSource
from .properties import PROPERTY_IDS, freeze, get_property


class ChoiceCatalog:
    def __init__(self, *, definitions=None, sources=None, allowed_name_tokens=None):
        self.definitions = freeze({int(key): value for key, value in (definitions or {}).items()})
        defaults = json.loads(Path(__file__).with_name('data').joinpath('interaction-enums.json').read_text())
        defaults.update(sources or {})
        self.sources = {int(key): tuple(NamedOption(int(value), label) for value, label in options)
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
        result = dict(self.properties(raw.get('CARDTEXTURE_NUMBER')))
        result.update(raw)
        types = {'IS_ARTIFACT':'Artifact', 'IS_BATTLE':'Battle', 'IS_CREATURE':'Creature',
                 'IS_ENCHANTMENT':'Enchantment', 'IS_LAND':'Land', 'IS_PLANESWALKER':'Planeswalker',
                 'INSTANT':'Instant', 'SORCERY':'Sorcery', 'IS_KINDRED':'Kindred'}
        if any(key in raw for key in types):
            result['card_types'] = tuple(label for key, label in types.items() if raw.get(key))
            result['permanent_types'] = tuple(label for label in result['card_types']
                if label in ('Artifact','Battle','Creature','Enchantment','Land','Planeswalker'))
        subtypes = [(option, get_property(raw, option.value)) for option in self.options(32768)]
        if any(value is not None for _, value in subtypes):
            result['creature_types'] = tuple(option.label for option, value in subtypes if value)
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
