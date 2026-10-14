import struct
import unittest
import uuid

from patchwork_automaton.game import (DeckItem, decks_response, solitaire_payload,
    game_status, event_sync, challenge_payload, accept_challenge_payload)
from patchwork_automaton.protocol import Cursor, ProtocolError, string


class GameTests(unittest.TestCase):
    def test_decks_with_sideboard_and_delimited_metadata(self):
        items = [DeckItem(123, 0, 0, 60, False), DeckItem(124, 0, 0, 15, True)]
        payload = struct.pack("<ii", 1, 99) + string("test", True) + string("1?CLEGACY?0?123", True)
        payload += string("ABC", False) + struct.pack("<i", 2) + b"".join(i.encode() for i in items)
        payload += struct.pack("<6i", 0, 0, 0, 0, 5, 0)[:-4]
        response, decks = decks_response(payload)
        self.assertEqual(response, 5)
        self.assertEqual(sum(i['quantity'] for i in decks[0]['cards'] if not i['sideboard']), 60)
        self.assertTrue(decks[0]['cards'][1]['sideboard'])

    def test_solitaire_layout_and_arbitrary_format_identifier(self):
        token = uuid.uuid4()
        cards = [vars(DeckItem(123, 0, 0, 60, False))]
        c = Cursor(solitaire_payload(token, cards, "CLEGACY"))
        self.assertEqual(c.take(16), token.bytes_le)
        self.assertEqual((c.string(), c.string(), c.string()), ("SEL", "CLEGACY", "SOLO"))
        self.assertEqual(c.string(), "")
        self.assertEqual(c.take(2), b"\0\0")
        self.assertEqual(struct.unpack("<ii?iii", c.take(21)), (6, 60, False, 0, 0, 0))
        self.assertEqual(c.string(), "bgname")
        self.assertEqual(c.integer(), 1)
        self.assertEqual(c.take(17), DeckItem(123, 0, 0, 60, False).encode())
        self.assertEqual(c.take(8), b"\0" * 8)
        c.end()
        self.assertIn(string('CFUTURE', True), solitaire_payload(token, cards, 'CFUTURE'))
        with self.assertRaises(ProtocolError):
            solitaire_payload(token, cards, '')

    def test_game_state_header_tracks_full_fragments_and_timestamp(self):
        payload = struct.pack("<i", 3) + b"abc" + struct.pack("<9i", *([0] * 8), 0)
        payload += struct.pack("<4I3iBBI", 77, 3, 3, 1, 0, 1, 0, 0, 6, 123)
        status = game_status(payload)
        self.assertEqual(status['game_id'], 77)
        self.assertEqual(status['game_state_timestamp'], 123)
        self.assertTrue(status['full_state'] and status['head'] and status['tail'])
        with self.assertRaises(ProtocolError):
            game_status(payload[:-1])

    def test_match_sync_response_uses_int32_handicap_not_request_bool(self):
        token = uuid.uuid4()
        p = token.bytes_le + struct.pack("<ii", 11, 0) + b"\0" * 16 + struct.pack("<i", 6)
        p += string("CLEGACY", True) + struct.pack("<i", 1)
        p += struct.pack("<ii", 42, 0) + string("synthetic", True) + struct.pack("<i?", 1, False)
        p += struct.pack("<iii", 1, 0, 8) + string("", True) + struct.pack("<ii", 0, 42)
        p += b"\0" * 18 + struct.pack("<i", 0) + string("", True) * 2
        p += b"\1" + struct.pack("<i", 0)  # MatchSyncData present / HostGSHServerID
        p += token.bytes_le + struct.pack("<ii", 11, 0)
        p += string("SEL", True) + string("CLEGACY", True) + string("SOLO", True)
        p += struct.pack("<i", 0) + string("", True)
        p += struct.pack("<?iiii?", True, 0, 6, 60, 0, False)
        p += b"\0" * 18 + struct.pack("<i?i", 42, False, 0)
        p += struct.pack("<iiiii", 1, 1, 1, 77, 9) + struct.pack("<i", 0)
        p += struct.pack("<i", 0) + string("", True) + b"\0" * 9
        p += struct.pack("<ii", 0, 0) + b"\0\0\0"
        c = Cursor(p)
        event = event_sync(c)
        c.end()
        self.assertEqual(event['player_format'], 'SOLO')
        self.assertEqual(event['maximum_players'], 1)
        self.assertEqual(event['players'], 1)
        self.assertEqual(event['games'], [{'game_id': 77, 'game_state': 9}])

    def test_direct_challenge_and_acceptance_wire_layouts(self):
        token = uuid.uuid4()
        cards = (DeckItem(123, 0, 0, 60, False),)
        c = Cursor(challenge_payload(token, cards, 42, 'CFREEFORM'))
        self.assertEqual(c.take(16), token.bytes_le)
        self.assertEqual((c.string(), c.string(), c.string()), ('SGM', 'CFREEFORM', 'D1V1'))
        self.assertEqual(c.string(), '')
        self.assertEqual(c.take(2), b'\0\0')
        self.assertEqual(struct.unpack('<ii?iii', c.take(21)), (6, 60, False, 0, 0, 0))
        self.assertEqual(c.string(), 'bgname')
        self.assertEqual((c.integer(), c.integer()), (0, 1))
        self.assertEqual(c.take(17), cards[0].encode())
        self.assertEqual(c.integer(), 42)
        self.assertEqual(c.string(), '')
        self.assertEqual(c.integer(), 0)
        c.end()
        c = Cursor(accept_challenge_payload(token, cards))
        self.assertEqual(c.take(16), token.bytes_le)
        self.assertEqual(c.take(12), b'\0' * 12)
        self.assertEqual((c.string(), c.string()), ('bgname', ''))
        self.assertEqual((c.integer(), c.integer()), (0, 1))
        self.assertEqual(c.take(17), cards[0].encode())
        self.assertEqual((c.integer(), c.integer()), (0, 1))
        c.end()
