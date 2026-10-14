"""Async TLS session and game handles, independent of the official client."""
import asyncio
from collections import OrderedDict
from contextlib import suppress
from dataclasses import asdict
import secrets
import ssl
import struct
import uuid

from .actions import CardChoice, PromptChoice, action_entries, card_payload
from .errors import (AuthenticationError, Disconnected, EventOverflow,
                     RequestRejected, StaleDecision, UnsupportedInteraction)
from .game import DeckItem, count, decks_response, event_sync, solitaire_payload
from .models import (CardAction, DecisionID, Deck, DeckList, GameMessage, Match,
                     Observation, Submission, freeze)
from .protocol import (Catalog, Cursor, ProtocolError, array, current_time,
                       login_payload, login_response, read_frame)
from .state import State


class MTGOClient:
    """One account connection. Create a new instance after disconnection.

    The reader handles heartbeat/ACK traffic even while callers think. No
    automatic login retries or action retries are performed.
    """

    def __init__(self, *, host='mtgologin1.mtgo.com', port=7770,
                 timeout=30.0, queue_size=256, strings=(), catalog=None,
                 connector=None, device_id=None):
        if queue_size < 1 or timeout <= 0:
            raise ValueError('Queue size and timeout must be positive')
        self.host, self.port, self.timeout = host, port, timeout
        self.queue_size, self.strings = queue_size, tuple(strings)
        self.catalog = catalog or Catalog()
        self._connector = connector or asyncio.open_connection
        self.device_id = device_id or secrets.token_hex(32).upper()
        self.session_id = uuid.uuid4()
        self.user_id = None
        self._writer = None
        self._reader_task = None
        self._write_lock = asyncio.Lock()
        self._decks_lock = asyncio.Lock()
        self._events_lock = asyncio.Lock()
        self._waiters = []
        self._pending = OrderedDict()
        self._next_ack = secrets.randbits(31) or 1
        self._ack_mode = False
        self._games = {}
        self._starts = {}
        self._failure = None
        self._closed = False
        self._authenticated = False
        self.heartbeats_answered = 0

    @property
    def connected(self):
        return self._authenticated and not self._closed and self._failure is None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def _check(self):
        if isinstance(self._failure, (ProtocolError, EventOverflow)):
            raise self._failure
        if self._closed or self._failure is not None or self._writer is None:
            raise Disconnected('Session is not connected') from self._failure

    def _waiter(self, predicate):
        self._check()
        future = asyncio.get_running_loop().create_future()
        entry = (predicate, future)
        self._waiters.append(entry)
        future.add_done_callback(lambda _: self._waiters.remove(entry) if entry in self._waiters else None)
        return future

    async def _request(self, name, payload, predicate):
        future = self._waiter(predicate)
        try:
            await self._send(name, payload)
            return await asyncio.wait_for(future, self.timeout)
        except (TimeoutError, asyncio.CancelledError) as error:
            # Uncorrelated responses (e.g. deck lists) could otherwise satisfy
            # a later request after cancellation or timeout.
            self._fail(error)
            raise
        finally:
            future.cancel()
            if future.done() and not future.cancelled():
                future.exception()  # Consume failures if sending failed first.

    async def login(self, username: str, password: str, *, challenge=''):
        """Authenticate; passwords and credential files are not retained."""
        if self._writer is not None or self._closed:
            raise RuntimeError('A client instance can only be logged in once')
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        try:
            reader, self._writer = await asyncio.wait_for(self._connector(
                self.host, self.port, ssl=context, server_hostname=self.host), self.timeout)
            self._reader_task = asyncio.create_task(self._read_loop(reader), name='mtgo-reader')
            _, payload = await self._request('FlsUserLoginReqMessage', login_payload(
                username, password, self.device_id, str(self.session_id), challenge), lambda op, _: op == 512)
            response = login_response(payload)
            if response.response_code != 1:
                raise AuthenticationError(response.response_code, response.authorization_type)
            # ACK negotiation is applied by the reader before it reads the next
            # packet, including any heartbeat immediately following login.
            self.user_id = response.user_id
            await self._send('ServerTimeSyncRespMessage', response.server_time + current_time())
            await self._send('DosUserroleGetprivilegesMessage', struct.pack('<ii', self.user_id, self.user_id))
            self._authenticated = True
        except BaseException:
            await self.close()
            raise

    async def _write(self, frame):
        self._check()
        self._writer.write(frame)
        await asyncio.wait_for(self._writer.drain(), self.timeout)

    async def _send(self, name, payload=b''):
        async with self._write_lock:
            frame = self.catalog.frame(name, payload)
            if self._ack_mode:
                if len(self._pending) >= 4096:
                    raise ProtocolError('Acknowledgement queue exhausted')
                message_id = self._next_ack
                self._next_ack = ((message_id + 1) & 0xffffffff) or 1
                self._pending[message_id] = None
                frame = self.catalog.frame('AckMessageReqMessage', struct.pack('<I', message_id) + array(frame))
            try:
                await self._write(frame)
            except BaseException as error:
                # A partial write has unknown outcome. Never retry automatically.
                self._fail(error)
                raise

    async def _unwrap(self, frame):
        for _ in range(8):
            opcode, payload = self.catalog.parse(frame)
            if opcode == 16:
                c = Cursor(payload)
                message_id = c.take(4)
                frame = c.blob()
                c.end()
                async with self._write_lock:
                    await self._write(self.catalog.frame('AckMessageRespMessage', message_id))
            elif opcode == 17:
                if len(payload) != 4:
                    raise ProtocolError('Invalid acknowledgement')
                message_id = struct.unpack('<I', payload)[0]
                if message_id in self._pending:
                    while self._pending:
                        key, _ = self._pending.popitem(last=False)
                        if key == message_id:
                            break
                return None
            elif opcode == 18:
                c = Cursor(payload)
                frame = c.blob()
                c.end()
            else:
                return opcode, payload
        raise ProtocolError('Envelope nesting limit exceeded')

    async def _read_loop(self, reader):
        try:
            while True:
                item = await self._unwrap(await asyncio.wait_for(read_frame(reader), 120))
                if item is not None:
                    await self._dispatch(*item)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._fail(error)
            if self._writer is not None:
                self._writer.close()

    def _fail(self, error):
        if self._failure is not None:
            return
        self._failure = error
        self._authenticated = False
        if self._writer is not None:
            self._writer.close()
        for _, future in tuple(self._waiters):
            if not future.done():
                future.set_exception(error if isinstance(error, (ProtocolError, EventOverflow))
                                     else Disconnected('Session ended'))
        for game in self._games.values():
            game._failure = error
            game._changed.set()
            # Failure is checked before consuming queued messages.
            with suppress(asyncio.QueueFull):
                game._queue.put_nowait(None)

    async def _dispatch(self, opcode, payload):
        if opcode == 512:
            response = login_response(payload)
            if response.response_code == 1:
                self._ack_mode = bool(response.client_flags & 0x1000000)
        elif opcode == 528:
            if len(payload) != 4:
                raise ProtocolError('Invalid heartbeat')
            await self._send('HeartbeatAckMessage', payload)
            self.heartbeats_answered += 1
        elif opcode == 3861:
            if len(payload) != 9:
                raise ProtocolError('Invalid time sync')
            await self._send('ServerTimeSyncRespMessage', payload + current_time())
        elif opcode in (self.catalog.by_name['UserKickedMessage']['opcode'],
                        self.catalog.by_name['ServerDisconnectMessage']['opcode']):
            raise Disconnected('Server ended the session')
        elif opcode == 2463:
            c = Cursor(payload)
            token, match_id, game_id = uuid.UUID(bytes_le=c.take(16)), c.integer(), c.integer()
            c.end()
            if len(self._starts) >= 1024:
                raise ProtocolError('Too many pending game starts')
            self._starts[token] = (match_id, game_id)
        elif opcode == 1153:
            c = Cursor(payload)
            route, match_id, game_id, inner = c.take(16), c.integer(), c.integer(), c.blob()
            c.end()
            game = self._games.get((match_id, game_id))
            if game is not None:
                inner_opcode, inner_payload = self.catalog.parse(inner)
                await game._receive(route, inner_opcode, inner_payload)
        for predicate, future in tuple(self._waiters):
            if not future.done() and predicate(opcode, payload):
                future.set_result((opcode, payload))
                break

    def _require_login(self):
        self._check()
        if not self.connected:
            raise Disconnected('Login is required')

    async def list_decks(self) -> DeckList:
        self._require_login()
        async with self._decks_lock:
            _, payload = await self._request('FlsNetDeckListGetReqMessage', struct.pack('<i', 0), lambda op, _: op == 94)
            code, decks = decks_response(payload)
            return DeckList(code, tuple(Deck(d['deck_id'], d['name'], d['description'],
                tuple(DeckItem(**card) for card in d['cards'])) for d in decks))

    async def current_matches(self) -> tuple[Match, ...]:
        self._require_login()
        async with self._events_lock:
            _, payload = await self._request('FlsPlayerEventCurrentEventListReqMessage', b'', lambda op, _: op == 983)
            c = Cursor(payload)
            matches = tuple(Match.from_wire(event_sync(c)) for _ in range(count(c)))
            c.end()
            return matches

    async def create_solitaire(self, deck: Deck | tuple[DeckItem, ...], *, format=None) -> 'Game':
        self._require_login()
        if isinstance(deck, Deck):
            format = format or deck.format
            cards = deck.cards
        else:
            cards = tuple(deck)
        if not format:
            raise ValueError('Format identifier is required')
        token = uuid.uuid4()
        opcode, payload = await self._request('FlsGameCreateAndJoinMessage',
            solitaire_payload(token, [asdict(card) for card in cards], format),
            lambda op, data: op in (960, 962, 944, 945, 107) and data[:16] == token.bytes_le)
        if opcode != 960:
            reason = struct.unpack_from('<i', payload, 16)[0] if len(payload) >= 20 else None
            raise RequestRejected(opcode, reason)
        c = Cursor(payload[16:])
        match = Match.from_wire(event_sync(c))
        c.end()
        if match.games:
            return await self.connect_game(match)
        # Register before sending host-start to cover immediately arriving start.
        future = self._waiter(lambda op, data: op == 2463 and data[:16] == match.token.bytes_le)
        try:
            started = self._starts.pop(match.token, None)
            if started is None:
                if match.state == 3:
                    await self._send('FlsMatchHostStartMessage', match.token.bytes_le + struct.pack('<i', match.id))
                await asyncio.wait_for(future, self.timeout)
                started = self._starts.pop(match.token)
            return await self.connect_game(match, game_id=started[1])
        finally:
            future.cancel()

    async def connect_game(self, match: Match, *, game_id=None) -> 'Game':
        """Attach to an existing ordinary match; also used after explicit re-login."""
        self._require_login()
        if game_id is None:
            if not match.games:
                raise ValueError('Match has no game')
            game_id = match.games[-1]
        key = (match.id, game_id)
        if key in self._games:
            return self._games[key]
        game = Game(self, match.id, game_id)
        self._games[key] = game
        try:
            await self._send('GsConnectMessage', match.token.bytes_le + struct.pack('<ii', *key))
            await game.wait_for_update(0, timeout=self.timeout)
            return game
        except BaseException:
            self._games.pop(key, None)
            raise

    async def close(self):
        if self._closed:
            return
        if self.connected:
            with suppress(Exception):
                await asyncio.wait_for(self._send('UserLogoffMessage'), min(3, self.timeout))
        self._closed = True
        self._fail(Disconnected('Client closed'))
        if self._reader_task is not None:
            self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)
        if self._writer is not None:
            self._writer.close()
            with suppress(Exception):
                await asyncio.wait_for(self._writer.wait_closed(), 3)


class Game:
    """A game stream with explicit decisions selected by the caller."""

    def __init__(self, client, match_id, game_id):
        self.client, self.match_id, self.id = client, match_id, game_id
        self._state = State(client.strings)
        self._route = None
        self._lock = asyncio.Lock()
        self._queue = asyncio.Queue(client.queue_size)
        self._changed = asyncio.Event()
        self._failure = None
        self._submitted = None
        self._streaming = False
        self._ended = False
        self.observation = None
        self.revision = 0

    def _check(self):
        self.client._check()
        if self._failure is not None:
            raise Disconnected('Game stream ended') from self._failure

    def _emit(self, event):
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            raise EventOverflow('Game consumer fell behind') from None

    async def _receive(self, route, opcode, payload):
        async with self._lock:
            self._route = route
            if opcode == 4652:
                if not self._state.update(payload):
                    return
                if self._state.status['game_id'] != self.id:
                    raise ProtocolError('Game state ID mismatch')
                self.revision += 1
                prompt = self._state.prompt
                decision = (DecisionID(self.client.session_id, self.id, self.revision, prompt['timestamp'])
                            if prompt is not None and not self._ended else None)
                actions = tuple(CardAction(card['id'], code,
                    props.get('ACTION_NAME_OVERRIDE_STRING') or props.get('ACTION_NAME_STRING') or '', freeze(props))
                    for card in self._state.cards.values() for code, props in action_entries(card['properties']))
                self.observation = Observation(decision, self.revision, freeze(prompt),
                    freeze(self._state.cards), freeze(self._state.players), actions,
                    freeze(self._state.status), self._state.unknown_elements, self._state.raw)
                self._changed.set()
                self._emit(self.observation)
            else:
                if opcode == 4632:
                    self._ended = True
                    self._submitted = self.observation.decision_id if self.observation else None
                name = self.client.catalog.by_opcode.get(opcode, {}).get('name', f'opcode_{opcode}')
                self._emit(GameMessage(opcode, name, payload))

    async def events(self):
        """Single-consumer stream of Observation and unmodeled GameMessage."""
        if self._streaming:
            raise RuntimeError('Only one event consumer is supported per game')
        self._streaming = True
        try:
            while True:
                self._check()
                item = await self._queue.get()
                self._check()
                yield item
        finally:
            self._streaming = False

    async def wait_for_update(self, after_revision: int, *, timeout=None) -> Observation:
        async def wait():
            while True:
                self._check()
                if self.revision > after_revision:
                    return self.observation
                self._changed.clear()
                await self._changed.wait()
        return await asyncio.wait_for(wait(), timeout)

    async def submit(self, decision_id: DecisionID, choice: PromptChoice | CardChoice) -> Submission:
        """Validate freshness and advertised IDs, then send once.

        Targets/modes are caller-selected wire values. This library does not
        implement a local rules engine or certify target legality.
        """
        async with self._lock:
            self._check()
            obs = self.observation
            if (self._ended or obs is None or obs.decision_id is None
                    or obs.decision_id != decision_id or self._submitted == decision_id):
                raise StaleDecision('Decision is outdated or already submitted')
            if isinstance(choice, PromptChoice):
                prompt = obs.prompt
                allowed = any(b['enabled'] and b['code'] == choice.code for b in prompt['buttons'])
                allowed |= any(m['flags'] & choice.code == choice.code and choice.code != 0 for m in prompt['mana'])
                allowed |= bool(prompt.get('flags', 0) & choice.code == choice.code and choice.code != 0)
                if not allowed:
                    raise UnsupportedInteraction('Prompt code is not advertised; use send_raw for unmodeled interactions')
                name, payload = 'PromptActionMessage', struct.pack('<III', choice.code, choice.parameter, decision_id.timestamp)
            elif isinstance(choice, CardChoice):
                if not any(a.card_id == choice.card_id and a.action_id == choice.action_id for a in obs.actions):
                    raise UnsupportedInteraction('Card action is not advertised')
                name = 'NewCardActionMessage'
                payload = card_payload(choice.card_id, choice.action_id, decision_id.timestamp,
                    choice.targets, modes=choice.modes, extra_mana=choice.extra_mana, flags=choice.flags)
            else:
                raise TypeError('Expected PromptChoice or CardChoice')
            self._submitted = decision_id
            await self._send(name, payload)
            return Submission(decision_id)

    async def _send(self, name, payload):
        if self._route is None:
            raise Disconnected('No game route')
        await self.client._send('GsMessageMessage', self._route + struct.pack('<ii', self.match_id, self.id)
                                + array(self.client.catalog.frame(name, payload)))

    async def send_raw(self, name: str, payload: bytes):
        """Expert escape hatch; bypasses decision validation. Never retried."""
        async with self._lock:
            self._check()
            await self._send(name, payload)
