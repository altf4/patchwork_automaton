"""Exercise the public API through an in-memory framed backend."""
import asyncio
from dataclasses import replace
import struct
import unittest
import uuid

from patchwork_automaton import (AuthenticationError, CardChoice, CollectionItem, DeckItem,
    Disconnected, EventOverflow, Match, MTGOClient, Observation, PromptChoice,
    RequestRejected, StaleDecision, UnsupportedInteraction)
from patchwork_automaton.actions import REVERSE, card_payload
from patchwork_automaton.client import Game
from patchwork_automaton.protocol import Catalog, Cursor, ProtocolError, array, string
from patchwork_automaton.state import checksum


def login_packet(code=1, flags=0):
    user = struct.pack('<i', 42) + string('synthetic', True)
    user += string('', True) * 2 + struct.pack('<i', 0) + string('', True) * 3
    user += struct.pack('<i', 0) + string('', True) + b'\0'
    user += b'\0' * 51 + array(b'x' * 64) + string('', True)
    user += struct.pack('<4i', 0, 0, 0, flags) + struct.pack('<i', 0)
    user += b'\0' * 18 + struct.pack('<i', 0) + string('', True)
    return user + b'\1' + struct.pack('<II', 1700000000, 0) + b'x' * 16 + struct.pack('<i', code)


def match_packet(token, match_id=11, game_id=77):
    p = token.bytes_le + struct.pack('<ii', match_id, 0) + b'\0' * 16 + struct.pack('<i', 6)
    p += string('CLEGACY', True) + struct.pack('<i', 1)
    p += struct.pack('<ii', 42, 0) + string('synthetic', True) + struct.pack('<i?', 1, False)
    p += struct.pack('<iii', 1, 0, 8) + string('', True) + struct.pack('<ii', 0, 42)
    p += b'\0' * 18 + struct.pack('<i', 0) + string('', True) * 2
    p += b'\1' + struct.pack('<i', 0) + token.bytes_le + struct.pack('<ii', match_id, 0)
    p += string('SEL', True) + string('CLEGACY', True) + string('SOLO', True)
    p += struct.pack('<i', 0) + string('', True) + struct.pack('<?iiii?', True, 0, 6, 60, 0, False)
    p += b'\0' * 18 + struct.pack('<i?i', 42, False, 0)
    p += struct.pack('<iiiii', 1, 1, 1, game_id, 9) + struct.pack('<i', 0)
    return p + struct.pack('<i', 0) + string('', True) + b'\0' * 9 + struct.pack('<ii', 0, 0) + b'\0' * 3


def state_packet(timestamp=1, text='Choose', unknown=False):
    ui = struct.pack('<iBB', 13, 0, 1) + b'Keep\0'
    prompt = struct.pack('<iiI3B', 1, 3, timestamp, 0, 0, 0) + text.encode() + b'\0\1'
    prompt += struct.pack('<iB', len(ui) + 5, 1) + ui
    elements = [struct.pack('<ii', len(prompt) + 8, 9) + prompt]
    props = struct.pack('<Ii', REVERSE['THINGNUMBER'] + 0x08000000, 123)
    props += struct.pack('<IIH', REVERSE['ACTION_LIST0'], REVERSE['ACTION_NAME_STRING'], 4) + b'Cast' + b'\0' * 8
    card = struct.pack('<i', 0) + props
    elements.append(struct.pack('<ii', len(card) + 8, 5) + card)
    if unknown:
        elements.append(struct.pack('<ii', 11, 88) + b'xyz')
    raw = b''.join(elements)
    return (array(raw) + b'\0' * 36 + struct.pack('<4I3IBBI', 77, len(raw), len(raw),
            len(elements), 0, checksum(raw), 0, 0, 6, timestamp))


class Backend:
    def __init__(self, *, login_code=1, ack=False, reject=False):
        self.catalog = Catalog()
        self.reader = asyncio.StreamReader()
        self.frames = []
        self.closed = False
        self.login_code, self.ack, self.reject = login_code, ack, reject
        self.token = uuid.uuid4()

    async def connect(self, host, port, **kwargs):
        assert kwargs['ssl'].check_hostname
        return self.reader, self

    def feed(self, name, payload=b''):
        self.reader.feed_data(self.catalog.frame(name, payload))

    def game_state(self, payload=None):
        self.feed('GsMessageMessage', self.token.bytes_le + struct.pack('<ii', 11, 77)
                  + array(self.catalog.frame('GamePlayStatusMessage', payload or state_packet())))

    def write(self, frame):
        if self.closed:
            raise ConnectionError('Closed')
        op, payload = self.catalog.parse(frame)
        if op == 16:
            c = Cursor(payload)
            message_id, inner = c.take(4), c.blob()
            self.feed('AckMessageRespMessage', message_id)
            op, payload = self.catalog.parse(inner)
        self.frames.append((op, payload))
        if op == 1:
            self.feed('FlsUserLoginRespMessage', login_packet(self.login_code, 0x1000000 if self.ack else 0))
            self.feed('HeartbeatReqMessage', struct.pack('<I', 99))
        elif op == 82:
            p = struct.pack('<ii', 1, 99) + string('Synthetic', True) + string('1?CLEGACY?0?', True)
            p += string('', False) + struct.pack('<i', 1) + DeckItem(1, 0, 0, 60, False).encode()
            self.feed('FlsNetDeckListGetRespMessage', p + struct.pack('<5i', 0, 0, 0, 0, 5))
        elif op == 982:
            self.feed('FlsPlayerEventCurrentEventListMessage', struct.pack('<i', 1) + match_packet(self.token))
        elif op == 929:
            self.token = uuid.UUID(bytes_le=payload[:16])
            if self.reject:
                self.feed('FlsGameInvalidDeckMessage', self.token.bytes_le)
            else:
                self.feed('FlsGameCreatedMessage', self.token.bytes_le + match_packet(self.token))
        elif op == 1024:
            self.game_state()

    async def drain(self):
        await asyncio.sleep(0)

    def close(self):
        if not self.closed:
            self.closed = True
            self.reader.feed_eof()

    async def wait_closed(self):
        pass


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def login(self, backend=None, **options):
        backend = backend or Backend()
        client = MTGOClient(connector=backend.connect, timeout=1, **options)
        self.addAsyncCleanup(client.close)
        await client.login('synthetic', 'synthetic')
        return client, backend

    async def test_public_workflow_and_game_envelope(self):
        client, backend = await self.login()
        decks = await client.list_decks()
        self.assertEqual(decks.decks[0].format, 'CLEGACY')
        self.assertEqual(decks.response_code, 5)
        game = await client.create_solitaire(decks.decks[0])
        self.assertEqual(game.id, 77)
        events = game.events()
        async def close_events():
            await events.aclose()
        self.addAsyncCleanup(close_events)
        obs = await anext(events)
        self.assertIsInstance(obs, Observation)
        self.assertEqual(obs.actions[0].label, 'Cast')
        receipt = await game.submit(obs.decision_id, PromptChoice(13))
        self.assertEqual(receipt.status, 'sent')
        c = Cursor(backend.frames[-1][1])
        self.assertEqual(c.take(16), backend.token.bytes_le)
        self.assertEqual((c.integer(), c.integer()), (11, 77))
        op, payload = client.catalog.parse(c.blob())
        self.assertEqual((op, struct.unpack('<III', payload)), (4638, (13, 0, 1)))
        self.assertGreaterEqual(client.heartbeats_answered, 1)

    async def test_ack_negotiation_and_existing_game(self):
        client, backend = await self.login(Backend(ack=True))
        matches = await client.current_matches()
        game = await client.connect_game(matches[0])
        self.assertIs(await client.connect_game(matches[0]), game)
        await game.submit(game.observation.decision_id, CardChoice(123, REVERSE['ACTION_LIST0'], (0,)))
        c = Cursor(backend.frames[-1][1]); c.take(24)
        op, payload = client.catalog.parse(c.blob())
        self.assertEqual(op, 4645)
        self.assertEqual(payload, card_payload(123, REVERSE['ACTION_LIST0'], 1, (0,)))

    async def test_stale_same_timestamp_duplicate_and_other_session(self):
        client, backend = await self.login()
        game = await client.connect_game((await client.current_matches())[0])
        old = game.observation
        backend.game_state(state_packet(text='Different', unknown=True))
        obs = await game.wait_for_update(old.revision, timeout=1)
        self.assertEqual(old.decision_id.timestamp, obs.decision_id.timestamp)
        self.assertEqual(obs.unknown_elements, ((88, b'xyz'),))
        for decision in (old.decision_id, replace(obs.decision_id, session=uuid.uuid4())):
            with self.assertRaises(StaleDecision):
                await game.submit(decision, PromptChoice(13))
        await game.submit(obs.decision_id, PromptChoice(13))
        with self.assertRaises(StaleDecision):
            await game.submit(obs.decision_id, PromptChoice(13))
        with self.assertRaises(TypeError):
            obs.cards[123]['properties']['THINGNUMBER'] = 42

    async def test_unadvertised_choices_are_not_sent(self):
        client, backend = await self.login()
        game = await client.connect_game((await client.current_matches())[0])
        before = len(backend.frames)
        for choice in (PromptChoice(999), CardChoice(999, REVERSE['ACTION_LIST0'])):
            with self.assertRaises(UnsupportedInteraction):
                await game.submit(game.observation.decision_id, choice)
        self.assertEqual(len(backend.frames), before)

    async def test_auth_refusal_closes_socket_and_does_not_retry(self):
        backend = Backend(login_code=101)
        client = MTGOClient(connector=backend.connect)
        with self.assertRaises(AuthenticationError) as error:
            await client.login('synthetic', 'synthetic')
        self.assertEqual(error.exception.code, 101)
        self.assertTrue(backend.closed)
        self.assertFalse(client.connected)
        self.assertEqual(sum(op == 1 for op, _ in backend.frames), 1)

    async def test_deck_rejection_is_typed(self):
        client, _ = await self.login(Backend(reject=True))
        with self.assertRaises(RequestRejected) as error:
            await client.create_solitaire((DeckItem(1, 0, 0, 1, False),), format='CFREEFORM')
        self.assertEqual(error.exception.opcode, 945)

    async def test_disconnect_wakes_game_and_request_waiters(self):
        client, backend = await self.login()
        game = await client.connect_game((await client.current_matches())[0])
        task = asyncio.create_task(game.wait_for_update(game.revision))
        backend.close()
        with self.assertRaises(Disconnected):
            await asyncio.wait_for(task, 1)
        self.assertFalse(client.connected)

    async def test_slow_consumer_fails_instead_of_dropping_state(self):
        client, backend = await self.login(queue_size=1)
        game = await client.connect_game((await client.current_matches())[0])
        backend.game_state(state_packet(text='Next'))
        with self.assertRaises(Disconnected):
            await game.wait_for_update(99, timeout=1)
        self.assertIsInstance(client._failure, EventOverflow)

    async def test_request_timeout_invalidates_session(self):
        client, backend = await self.login()
        client.timeout = 0.01
        with self.assertRaises(TimeoutError):
            await client._request('FlsPlayerEventCurrentEventListReqMessage', b'', lambda op, _: op == 60000)
        self.assertFalse(client.connected)
        self.assertTrue(backend.closed)

    async def test_close_logs_off_and_ends_reader(self):
        client, backend = await self.login()
        await client.close()
        self.assertTrue(client._reader_task.done())
        self.assertFalse(client.connected)
        self.assertIn(client.catalog.by_name['UserLogoffMessage']['opcode'], [op for op, _ in backend.frames])

    async def test_cancelled_request_invalidates_session(self):
        client, backend = await self.login()
        task = asyncio.create_task(client._request('FlsPlayerEventCurrentEventListReqMessage',
            b'', lambda op, _: op == 60000))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(client.connected)
        self.assertTrue(backend.closed)

    async def test_heartbeat_while_game_consumer_is_idle(self):
        client, backend = await self.login()
        game = await client.connect_game((await client.current_matches())[0])
        before = client.heartbeats_answered
        backend.feed('HeartbeatReqMessage', struct.pack('<I', 678))
        for _ in range(20):
            if client.heartbeats_answered > before:
                break
            await asyncio.sleep(0)
        self.assertGreater(client.heartbeats_answered, before)
        self.assertIn((23, struct.pack('<I', 678)), backend.frames)
        self.assertEqual(game.revision, 1)

    async def test_bad_checksum_is_reported_without_new_observation(self):
        client, backend = await self.login()
        game = await client.connect_game((await client.current_matches())[0])
        payload = bytearray(state_packet())
        payload[-14] ^= 1  # Corrupt the expected checksum, not an early byte
        backend.game_state(bytes(payload))
        with self.assertRaises(ProtocolError):
            await game.wait_for_update(2, timeout=1)
        self.assertEqual(game.revision, 1)

    async def test_game_over_and_results_invalidate_decision(self):
        client, backend = await self.login()
        game = await client.connect_game((await client.current_matches())[0])
        await game._receive(backend.token.bytes_le, 4485, struct.pack('<IiBiBI',77,0,1,1,2,100))
        with self.assertRaises(StaleDecision):
            await game.submit(game.observation.decision_id, PromptChoice(13))

    async def test_write_failure_does_not_retry_action(self):
        client, backend = await self.login()
        game = await client.connect_game((await client.current_matches())[0])
        before = sum(op == 1153 for op, _ in backend.frames)
        async def fail_drain():
            raise ConnectionError('Synthetic partial write')
        backend.drain = fail_drain
        with self.assertRaises(ConnectionError):
            await game.submit(game.observation.decision_id, PromptChoice(13))
        self.assertFalse(client.connected)
        self.assertEqual(sum(op == 1153 for op, _ in backend.frames), before + 1)

    async def test_direct_challenge_acceptance_player_order_and_concession(self):
        class ChallengeBackend(Backend):
            def write(self, frame):
                op, payload = self.catalog.parse(frame)
                super().write(frame)
                if op == 118:
                    self.token = uuid.UUID(bytes_le=payload[:16])
                    self.feed('FlsGameCreatedMessage', self.token.bytes_le + match_packet(self.token))
                    self.feed('FlsEventSyncMessage', match_packet(self.token))
                elif op == 946:
                    self.feed('FlsMatchChallengeAcceptedMessage', self.token.bytes_le + struct.pack('<ii', 11, 123) + string('synthetic', True))
                    self.feed('FlsMatchGameStartedMessage', self.token.bytes_le + struct.pack('<ii', 11, 77))
                elif op == 1024:
                    self.feed('GsPlayerOrderMessage', self.token.bytes_le + struct.pack('<5i', 11, 77, 2, 123, 42))
                elif op == 972:
                    self.feed('FlsMatchConcededMessage', self.token.bytes_le + struct.pack('<i', 11))
        client, backend = await self.login(ChallengeBackend())
        cards = (DeckItem(1, 0, 0, 60, False),)
        match = await client.challenge(123, cards, format='CFREEFORM')
        self.assertEqual((await client.wait_for_match(match.token)).id, 11)
        game = await client.accept_challenge(match, cards)
        for _ in range(20):
            if game.player_index is not None:
                break
            await asyncio.sleep(0)
        self.assertEqual(game.player_index, 1)
        self.assertEqual(game.player_ids, (123, 42))
        await client.concede_match(match)
        self.assertTrue(client.connected)
        self.assertIn((990, backend.token.bytes_le + struct.pack('<4i', 11, 2, 42, 123)), backend.frames)
        self.assertIn((111, backend.token.bytes_le + struct.pack('<i', 11)), backend.frames)

    async def test_ready_ack_for_a_recovered_match(self):
        client, backend = await self.login()
        match = (await client.current_matches())[0]
        payload = match.token.bytes_le + struct.pack('<i', match.id)
        await client._dispatch(933, payload)
        self.assertIn((950, payload), backend.frames)

    async def test_owned_product_opening_returns_collection_deltas(self):
        class ProductBackend(Backend):
            def write(self, frame):
                op, payload = self.catalog.parse(frame)
                super().write(frame)
                if op == 300:
                    self.feed('DosCollectionListMessage', struct.pack('<5i', 42, 1, 8, 129, 1))
                elif op == 2422:
                    self.token = uuid.UUID(bytes_le=payload[:16])
                    self.feed('FlsEscrowVerifyMessage', struct.pack('<i', 10) + self.token.bytes_le + struct.pack('<ii', 1, 42))
                elif op == 2423:
                    c = Cursor(payload)
                    self.assert_update = (c.take(16), c.integer(), c.integer(), c.take(9), c.integer(), c.take(12))
                    c.end()
                    self.feed('FlsEscrowCompleteMessage', self.token.bytes_le + struct.pack('<i8i',
                        2, 8, 129, -1, 42, 123, 642, 4, 42))
        client, backend = await self.login(ProductBackend())
        result = await client.open_product(CollectionItem(8, 129, 1))
        self.assertEqual(result, (CollectionItem(8, 129, -1), CollectionItem(123, 642, 4)))
        self.assertEqual(backend.assert_update[1:], (10, 42, b'\0' * 9, 1, struct.pack('<iii', 8, 129, 1)))
        with self.assertRaises(ValueError):
            await client.open_product(CollectionItem(99, 129, 1))
