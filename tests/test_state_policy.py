import json
from pathlib import Path
import struct
import unittest
from types import SimpleNamespace

from patchwork_automaton.protocol import Cursor, ProtocolError
from patchwork_automaton.state import attributes, apply_diff, checksum, interaction, State
from research.policy import SoloPolicy
from patchwork_automaton.actions import card_payload


class StatePolicyTests(unittest.TestCase):
    def test_nested_attributes_large_integer_and_string_reference(self):
        # ACTION_LIST0 contains string ref "Play" and a large ACTION_FLAGS.
        blob=struct.pack('<IIHI',202329600,1074895385,65535,0)
        blob+=struct.pack('<Ii',538023699+0x08000000,1074266113)+b'\0'*8
        c=Cursor(blob)
        props=attributes(c,['Play']);c.end()
        self.assertEqual(props['ACTION_LIST0']['ACTION_NAME_STRING'],'Play')
        self.assertEqual(props['ACTION_LIST0']['ACTION_FLAGS'],1074266113)

    def test_diff_copy_literal_and_source_bounds(self):
        self.assertEqual(apply_diff(b'abcdef',b'\x83\0\x02XY\x81\x02',6),b'abcXYf')
        with self.assertRaises(ProtocolError):
            apply_diff(b'abc',b'\x83\xff',3)
        with self.assertRaises(ProtocolError):
            apply_diff(b'abc',b'\x01X',2)

    def test_legacy_prompt_allows_unused_zero_padding(self):
        prompt=struct.pack('<iiIII6B',1,3,2,0x08000000,0x08000000,0,0,0,0,0,0)
        prompt+=b'Combat\0'.ljust(32,b'\0')+b'\0'*96+b'First main phase.\0\0'
        parsed=interaction(prompt,True)
        self.assertEqual(parsed['buttons'][0]['label'],'Combat')
        self.assertEqual(parsed['timestamp'],2)
        self.assertEqual(parsed['phase'],4)

    def test_full_state_checksum_and_per_player_diff(self):
        def packet(raw,flags,expected,prior=0,size=None):
            return struct.pack('<i',len(raw))+raw+b'\0'*36+struct.pack('<4I3IBBI',77,len(raw),size or len(raw),0,0,expected,prior,0,flags,1)
        state=State()
        raw=b''
        self.assertTrue(state.update(packet(raw,6,checksum(raw))))
        with self.assertRaises(ProtocolError):
            state.update(packet(raw,6,0))

    def test_policy_plays_land_before_spell_and_avoids_self_targets(self):
        info={'1':{'name':'Mountain','cmc':0},'2':{'name':'Creature','cmc':1,'type_line':'CREATURE'},
              '3':{'name':'Bolt','cmc':1,'type_line':'INSTANT'}}
        def card(n,ctn,zone,label,target=False):
            a={'ACTION_NAME_STRING':label}
            if target:a['ACTION_TARGET_INFO_LIST0']={'MIN_TARGETS':1}
            return {'id':n,'properties':{'CARDTEXTURE_NUMBER':ctn,'CONTROLLER':0,'ZONE':zone,'ACTION_LIST0':a}}
        s=SimpleNamespace(prompt={'player':0,'phase':4,'timestamp':1,'text':'Main phase','buttons':[]},
                          cards={1:card(1,2,1000448,'Play'),2:card(2,4,1000448,'Cast'),3:card(3,6,1000448,'Cast',True)})
        p=SoloPolicy(info)
        self.assertEqual(p.choose(s)[1],1)
        s.cards[1]=card(1,2,1000961,'Add R')
        self.assertEqual(p.choose(s)[1],2)
        del s.cards[2]
        self.assertIsNone(p.choose(s))

    def test_new_card_action_layout(self):
        payload=card_payload(123,202329600,42,[0])
        self.assertEqual(struct.unpack('<iiiIBBBB',payload[:20]),(123,202329600,0,42,1,0,1,0))
        self.assertEqual(struct.unpack('<iI',payload[20:]),(1,0))

    def test_fragmented_state_preserves_unknown_elements(self):
        raw = struct.pack('<ii', 11, 88) + b'xyz'
        def fragment(data, flags):
            return struct.pack('<i', len(data)) + data + b'\0' * 36 + struct.pack(
                '<4I3IBBI', 77, len(raw), len(raw), 1, 0, checksum(raw), 0, 0, flags, 1)
        state = State()
        self.assertFalse(state.update(fragment(raw[:5], 2)))
        self.assertTrue(state.update(fragment(raw[5:], 4)))
        self.assertEqual(state.unknown_elements, ((88, b'xyz'),))
        self.assertEqual(state.raw, raw)

    def test_unknown_ui_selector_retains_wire_bytes(self):
        prompt = struct.pack('<iiI3B', 1, 3, 1, 0, 0, 0) + b'Choose\0\1'
        prompt += struct.pack('<iB', 8, 99) + b'xyz'
        parsed = interaction(prompt)
        self.assertEqual(parsed['selectors'], [{'type':99, 'bytes':'78797a'}])

    def test_card_action_modes_extra_mana_and_limits(self):
        payload = card_payload(1, 202329600, 2, (0xfffffffe, 123), modes=b'\2\3', extra_mana=5)
        self.assertEqual(payload[8:14], struct.pack('<i', 2) + b'\2\3')
        self.assertEqual(struct.unpack('<IBBBB', payload[14:22]), (2, 2, 5, 1, 0))
        with self.assertRaises(ProtocolError):
            card_payload(1, 2, 3, tuple(range(256)))
        with self.assertRaises(ProtocolError):
            card_payload(1, 2, 3, extra_mana=256)
