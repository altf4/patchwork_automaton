"""Synthetic sideboarding and game-transition workflows through framed transport."""
import asyncio
from dataclasses import replace
import struct
import unittest
import uuid

from patchwork_automaton import DeckItem, Disconnected, MTGOClient
from patchwork_automaton.game import challenge_payload
from patchwork_automaton.protocol import Cursor, ProtocolError, array
from test_client import Backend, state_packet


class MatchBackend(Backend):
    response = 'FlsGameSubmitSideboardSuccessMessage'
    errors = ()

    def write(self, frame):
        op,payload=self.catalog.parse(frame)
        if op == 940:
            self.frames.append((op,payload))
            body=payload[:20]
            if self.response == 'FlsGameSubmitSideboardFailureMessage':
                body+=struct.pack('<i',len(self.errors))+b''.join(struct.pack('<iIii?',*e) for e in self.errors)
            self.feed(self.response,body)
        elif op == 1024:
            self.frames.append((op,payload))
            token,match_id,game_id=payload[:16],*struct.unpack_from('<ii',payload,16)
            state=state_packet()
            state=state[:-34]+struct.pack('<I',game_id)+state[-30:]
            self.feed('GsMessageMessage',token+struct.pack('<ii',match_id,game_id)
                      +array(self.catalog.frame('GamePlayStatusMessage',state)))
        else:
            super().write(frame)


class MatchLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def setup_match(self):
        backend=MatchBackend()
        client=MTGOClient(connector=backend.connect,timeout=1)
        self.addAsyncCleanup(client.close)
        await client.login('synthetic','synthetic')
        match,=await client.current_matches()
        game=await client.connect_game(match)
        return client,backend,match,game

    async def test_sideboard_complete_deck_and_detach_prior_game(self):
        client,backend,match,game=await self.setup_match()
        await game._receive(backend.token.bytes_le,4632,b'')
        cards=(DeckItem(123,0,0,56,False),DeckItem(124,0,0,4,False),DeckItem(125,0,0,15,True))
        result=await client.submit_sideboard(match,cards)
        self.assertTrue(result.accepted)
        self.assertFalse(result.expired)
        sideboard=next(data for op,data in backend.frames if op==940)
        c=Cursor(sideboard)
        self.assertEqual(c.take(16),match.token.bytes_le)
        self.assertEqual((c.integer(),c.integer()),(11,3))
        self.assertEqual(c.take(51),b''.join(card.encode() for card in cards))
        c.end()
        self.assertTrue(any(op==1154 for op,_ in backend.frames))
        self.assertTrue(client.connected)
        with self.assertRaises(Disconnected):
            game._check()

    async def test_sideboard_preserves_validation_errors(self):
        client,backend,match,game=await self.setup_match()
        backend.response='FlsGameSubmitSideboardFailureMessage'
        backend.errors=((123,0x80000000,9,2,True),(124,0,10,3,False))
        await game._receive(backend.token.bytes_le,4632,b'')
        result=await client.submit_sideboard(match,(DeckItem(123,0,0,60,False),))
        self.assertFalse(result.accepted)
        self.assertEqual((result.errors[0].catalog_id,result.errors[0].annotation,result.errors[0].code),(123,0x80000000,9))
        self.assertEqual(result.errors[1].region,3)
        self.assertTrue(client.connected)

    async def test_sideboard_expiration(self):
        client,backend,match,game=await self.setup_match()
        backend.response='FlsGameSideboardTimeExpiredMessage'
        await game._receive(backend.token.bytes_le,4632,b'')
        result=await client.submit_sideboard(match,(DeckItem(123,0,0,60,False),))
        self.assertTrue(result.expired)
        self.assertFalse(result.accepted)

    async def test_sideboard_rejects_live_game_unknown_match_and_empty_deck(self):
        client,backend,match,game=await self.setup_match()
        before=len(backend.frames)
        cards=(DeckItem(123,0,0,60,False),)
        with self.assertRaises(ValueError):
            await client.submit_sideboard(match,cards)
        with self.assertRaises(ValueError):
            await client.submit_sideboard(replace(match,token=uuid.uuid4()),cards)
        with self.assertRaises(ValueError):
            await client.submit_sideboard(match,())
        self.assertEqual(len(backend.frames),before)

    async def test_next_game_uses_new_start_and_never_returns_ended_game(self):
        client,backend,match,game=await self.setup_match()
        await game._receive(backend.token.bytes_le,4632,b'')
        client._starts[match.token]=(11,77)  # Old notification must be discarded.
        waiting=asyncio.create_task(client.wait_for_game(match,after_game_id=77))
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        backend.feed('FlsMatchGameStartedMessage',match.token.bytes_le+struct.pack('<ii',11,78))
        new=await waiting
        self.assertEqual((new.id,new.observation.status['game_id']),(78,78))
        self.assertIsNot(new,game)

    async def test_next_game_uses_fresh_match_sync_even_with_old_argument(self):
        client,backend,match,game=await self.setup_match()
        waiting=asyncio.create_task(client.wait_for_game(match,after_game_id=77))
        await asyncio.sleep(0)
        client._matches[match.token]=replace(match,games=(77,78))
        client._matches_changed.set()
        new=await waiting
        self.assertEqual(new.id,78)

    async def test_next_game_rejects_mismatched_match_id(self):
        client,backend,match,game=await self.setup_match()
        client._starts[match.token]=(999,78)
        with self.assertRaises(ProtocolError):
            await client.wait_for_game(match,after_game_id=77)

    async def test_best_of_three_uses_native_first_two_wins_match_type(self):
        cards=(DeckItem(123,0,0,60,False),)
        c=Cursor(challenge_payload(uuid.uuid4(),cards,123,'CMODERN',best_of=3))
        c.take(16)
        self.assertEqual((c.string(),c.string(),c.string()),('BES','CMODERN','D1V1'))
        with self.assertRaises(ValueError):
            challenge_payload(uuid.uuid4(),cards,123,'CMODERN',best_of=5)
