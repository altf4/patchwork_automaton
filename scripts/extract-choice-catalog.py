"""Build a local card-name/type choice catalog from installed client data.

Requires dnfile only for this discovery tool. Output is local client data,
not part of the distributable library. Does not connect to an account/server.
"""
import argparse
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import dnfile


def enums(assembly):
    pe = dnfile.dnPE(str(assembly))
    constants = {}
    for entry in pe.net.mdtables.Constant.rows:
        if entry.Type in (8, 9) and len(entry.Value.value) == 4:
            constants[id(entry.Parent.row)] = struct.unpack('<i' if entry.Type == 8 else '<I', entry.Value.value)[0]
    result = {}
    for typedef in pe.net.mdtables.TypeDef.rows:
        name = str(typedef.TypeName)
        if name.startswith('Magic'):
            result[name] = {str(field.row.Name): constants[id(field.row)] for field in typedef.FieldList
                            if id(field.row) in constants}
    return result


def build(assembly, directory):
    enum_values = enums(assembly)
    lookups = {}
    def resolve(element):
        if 'value' in element.attrib:
            return element.get('value')
        if 'id' in element.attrib:
            tag = element.tag
            if tag not in lookups:
                path = directory / (tag + '.xml')
                lookups[tag] = {e.get('id'): e.text for e in ET.parse(path).getroot()} if path.exists() else {}
            return lookups[tag].get(element.get('id'))
        return '1'
    definitions = {}
    types = {'IS_ARTIFACT':'Artifact', 'ARTIFACT':'Artifact', 'IS_CREATURE':'Creature', 'CREATURE':'Creature',
             'IS_ENCHANTMENT':'Enchantment', 'ENCHANTMENT':'Enchantment', 'IS_LAND':'Land',
             'IS_PLANESWALKER':'Planeswalker', 'PLANESWALKER':'Planeswalker', 'IS_BATTLE':'Battle',
             'INSTANT':'Instant', 'SORCERY':'Sorcery', 'IS_KINDRED':'Kindred'}
    name_tokens = enum_values.get('MagicCardNameToken', {})
    creature_types = enum_values.get('MagicCreatureType', {})
    for path in sorted(directory.glob('client_*.xml')):
        for element in ET.parse(path).getroot().iter('DigitalObject'):
            props = {}
            for attr in element:
                value = resolve(attr)
                if value is None:
                    continue
                try:
                    props[attr.tag] = int(value)
                except ValueError:
                    props[attr.tag] = value
            if props.get('DIGITAL_OBJECT_TYPE_CODE_STRING') != 'CARD' or not props.get('CARDTEXTURE_NUMBER'):
                continue
            token = props.get('CARDNAME_TOKEN')
            if isinstance(token, str):
                props['CARDNAME_TOKEN'] = name_tokens.get(token)
            props['card_types'] = sorted({label for tag, label in types.items() if props.get(tag)})
            props['permanent_types'] = [label for label in props['card_types']
                                         if label in ('Artifact','Creature','Enchantment','Land','Planeswalker','Battle')]
            props['creature_types'] = [key for key in creature_types if props.get(key)]
            for node in element.iter():
                value = resolve(node)
                if value in creature_types and value not in props['creature_types']:
                    props['creature_types'].append(value)
            definitions[props['CARDTEXTURE_NUMBER']] = props
    return {'definitions': definitions}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attributes', type=Path, help='Mtgo.Shared.Parser.Attributes.dll')
    parser.add_argument('card_data', type=Path, help='Client CardDataSource directory')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = build(args.attributes, args.card_data)
    args.output.write_text(json.dumps(data))
    print('Exported card definitions:', len(data['definitions']))
