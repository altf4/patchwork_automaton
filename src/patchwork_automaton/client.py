"""Async TLS session and game handles, independent of the official client."""
import asyncio
from collections import OrderedDict
from contextlib import suppress
from dataclasses import asdict, replace
import secrets
import ssl
import struct
import uuid

from .actions import (CardChoice, CardSelection, DamageChoice, ManaChoice, OrderingChoice,
                      PromptChoice, SelectorChoice, action_entries, encode_choice)
from .errors import (AuthenticationError, Disconnected, EventOverflow, GameEnded,
                     RequestRejected, StaleDecision, UnsupportedInteraction)
from .game import (DeckItem, accept_challenge_payload, challenge_payload, count,
                   decks_response, event_sync, solitaire_payload)
from .models import (CardAction, CollectionItem, DecisionID, Deck, DeckList, GameMessage, Match,
                     Observation, GameResult, PlayerResult, SideboardResult, ValidationError, Submission, freeze)
from .protocol import (Catalog, Cursor, ProtocolError, array, current_time,
                       login_payload, login_response, read_frame)
from .state import State
from .interactions import GameObject, mode_spec, prompt_selectors, target_groups
from .choice_catalog import ChoiceCatalog
from .properties import indexed
from .controls import ControlReceipt, YieldControl, stop_payload


class MTGOClient:
    """One account connection. Create a new instance after disconnection.

    The reader handles heartbeat/ACK traffic even while callers think. No
    automatic login retries or action retries are performed.
    """

    def __init__(self, *, host='mtgologin1.mtgo.com', port=7770,
                 timeout=30.0, queue_size=256, strings=(), catalog=None,
                 connector=None, device_id=None, choice_catalog=None):
        if queue_size < 1 or timeout <= 0:
            raise ValueError('Queue size and timeout must be positive')
        self.host, self.port, self.timeout = host, port, timeout
        self.queue_size, self.strings = queue_size, tuple(strings)
        self.catalog = catalog or Catalog()
        self.choice_catalog = choice_catalog or ChoiceCatalog()
        self._connector = connector or asyncio.open_connection
        self.device_id = device_id or secrets.token_hex(32).upper()
        self.session_id = uuid.uuid4()
        self.user_id = None
        self._writer = None
        self._reader_task = None
        self._write_lock = asyncio.Lock()
        self._decks_lock = asyncio.Lock()
        self._events_lock = asyncio.Lock()
        self._collection_lock = asyncio.Lock()
        self._open_lock = asyncio.Lock()
        self._sideboard_lock = asyncio.Lock()
        self._waiters = []
        self._pending = OrderedDict()
        self._next_ack = secrets.randbits(31) or 1
        self._ack_mode = False
        self._games = {}
        self._starts = {}
        self._matches = {}
        self._matches_changed = asyncio.Event()
        self._player_orders = {}
        self._outgoing_challenges = {}
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
        self._matches_changed.set()

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
            self._matches_changed.set()
        elif opcode in (959, 960):
            c = Cursor(payload[16:] if opcode == 960 else payload)
            match = Match.from_wire(event_sync(c))
            c.end()
            self._matches[match.token] = match
            self._matches_changed.set()
        elif opcode == 123:
            c = Cursor(payload)
            token, match_id, receiver = uuid.UUID(bytes_le=c.take(16)), c.integer(), c.integer()
            c.string(); c.end()
            opponent = self._outgoing_challenges.pop(token, None)
            if opponent is not None:
                if receiver != opponent:
                    raise ProtocolError('Challenge receiver mismatch')
                # MatchChallengeMadeByCurrentUserState does both operations
                # immediately after FlsMatchChallengeAcceptedMessage.
                await self._send('FlsMatchPlayerOrderMessage', token.bytes_le
                    + struct.pack('<4i', match_id, 2, self.user_id, receiver))
                await self._send('FlsMatchHostStartMessage', token.bytes_le + struct.pack('<i', match_id))
        elif opcode == 933:
            if len(payload) != 20:
                raise ProtocolError('Invalid game-ready message')
            token = uuid.UUID(bytes_le=payload[:16])
            if token in self._matches:
                await self._send('FlsGameReadyToStartAckMessage', payload)
        elif opcode == 1155:
            c = Cursor(payload)
            c.take(16)
            key = (c.integer(), c.integer())
            order = tuple(c.integer() for _ in range(count(c)))
            c.end()
            self._player_orders[key] = order
            if key in self._games:
                await self._games[key]._set_player_order(order)
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
            for match in matches:
                self._matches[match.token] = match
            return matches

    async def collection(self) -> tuple[CollectionItem, ...]:
        """Read this account's own collection, including unopened products."""
        self._require_login()
        async with self._collection_lock:
            _, payload = await self._request('GetCollectionMessage', struct.pack('<i', self.user_id),
                lambda op, data: op in (848, 2583) and len(data) >= 4
                and struct.unpack_from('<i', data)[0] == self.user_id)
            c = Cursor(payload)
            c.integer()
            items = tuple(CollectionItem(*struct.unpack('<iii', c.take(12))) for _ in range(count(c)))
            c.end()
            return items

    async def open_product(self, item: CollectionItem) -> tuple[CollectionItem, ...]:
        """Explicitly consume an owned sealed product; return collection deltas.

        This changes inventory and is never automatically retried. Catalog
        product identity and the caller's intent must be established first.
        """
        self._require_login()
        if item.quantity <= 0:
            raise ValueError('Quantity must be positive')
        async with self._open_lock:
            owned = await self.collection()
            if sum(i.quantity for i in owned if (i.catalog_id, i.permission) ==
                   (item.catalog_id, item.permission)) < item.quantity:
                raise ValueError('Requested product quantity is not owned')
            token = uuid.uuid4()
            opcode, payload = await self._request('DosEscrowOpenpackSetupReqMessage',
                token.bytes_le + struct.pack('<i', self.user_id),
                lambda op, data: (op == 182 and data[4:20] == token.bytes_le)
                or (op in (819, 821) and data[:16] == token.bytes_le))
            if opcode != 182:
                reason = struct.unpack_from('<i', payload, 20)[0] if len(payload) >= 24 else None
                raise RequestRejected(opcode, reason)
            c = Cursor(payload)
            escrow_id = c.integer()
            c.take(16); c.blob(4); c.end()
            # Native open-pack DepositList sets both participant IDs and
            # ClearFirst to zero; only Items is populated.
            deposit = b'\0' * 9 + struct.pack('<i', 1) + item.encode()
            opcode, payload = await self._request('DosEscrowOpenpackUpdatelistReqMessage',
                token.bytes_le + struct.pack('<ii', escrow_id, self.user_id) + deposit,
                lambda op, data: op in (823, 819, 821, 188) and data[:16] == token.bytes_le)
            if opcode != 823:
                reason = struct.unpack_from('<i', payload, 20)[0] if len(payload) >= 24 else None
                raise RequestRejected(opcode, reason)
            c = Cursor(payload)
            c.take(16)
            deltas = []
            for _ in range(count(c)):
                catalog_id, permission, quantity, receiver = struct.unpack('<4i', c.take(16))
                if receiver == self.user_id:
                    deltas.append(CollectionItem(catalog_id, permission, quantity))
            c.end()
            return tuple(deltas)

    async def challenge(self, opponent_id: int, deck: Deck | tuple[DeckItem, ...], *, format=None, best_of=1) -> Match:
        """Send an ordinary direct challenge to a logged-in user."""
        self._require_login()
        cards = deck.cards if isinstance(deck, Deck) else tuple(deck)
        format = format or (deck.format if isinstance(deck, Deck) else None)
        if not format:
            raise ValueError('Format identifier is required')
        token = uuid.uuid4()
        request = challenge_payload(token, cards, opponent_id, format, best_of=best_of)
        self._outgoing_challenges[token] = opponent_id
        opcode, payload = await self._request('FlsMatchChallengeRequestMessage',
            request,
            lambda op, data: op in (960, 125, 107, 944, 945) and data[:16] == token.bytes_le)
        if opcode != 960:
            self._outgoing_challenges.pop(token, None)
            offset = 20 if opcode == 125 else 16
            reason = struct.unpack_from('<i', payload, offset)[0] if len(payload) >= offset + 4 else None
            raise RequestRejected(opcode, reason)
        return self._matches[token]

    async def start_match(self, match: Match, *, player_ids: tuple[int, ...]) -> 'Game':
        """Host an accepted ordinary match, including one recovered after login."""
        self._require_login()
        await self._send('FlsMatchPlayerOrderMessage', match.token.bytes_le
            + struct.pack('<ii', match.id, len(player_ids))
            + b''.join(struct.pack('<i', user_id) for user_id in player_ids))
        await self._send('FlsMatchHostStartMessage', match.token.bytes_le + struct.pack('<i', match.id))
        return await self.wait_for_game(match)

    async def wait_for_match(self, token: uuid.UUID, *, timeout=None) -> Match:
        """Wait for an incoming match sync, including a direct challenge."""
        async def wait():
            while True:
                self._require_login()
                if token in self._matches:
                    return self._matches[token]
                self._matches_changed.clear()
                await self._matches_changed.wait()
        return await asyncio.wait_for(wait(), self.timeout if timeout is None else timeout)

    async def accept_challenge(self, match: Match, deck: Deck | tuple[DeckItem, ...]) -> 'Game':
        self._require_login()
        cards = deck.cards if isinstance(deck, Deck) else tuple(deck)
        opcode, payload = await self._request('FlsPlayerEventJoinReqMessage',
            accept_challenge_payload(match.token, cards),
            lambda op, data: op in (123, 125, 980, 944, 945) and data[:16] == match.token.bytes_le)
        if opcode != 123:
            offset = 20 if opcode == 125 else 16
            reason = struct.unpack_from('<i', payload, offset)[0] if len(payload) >= offset + 4 else None
            raise RequestRejected(opcode, reason)
        return await self.wait_for_game(match)

    async def wait_for_game(self, match: Match, *, after_game_id=None, timeout=None) -> 'Game':
        """Wait for a game, or the next game after an explicitly supplied ID.

        Match syncs and game-start notifications can arrive in either order.
        Refresh the match from the session cache while waiting.
        """
        async def wait():
            while True:
                self._require_login()
                started = self._starts.pop(match.token, None)
                if started and started[0] != match.id:
                    raise ProtocolError('Game start match ID mismatch')
                if started and started[1] != after_game_id:
                    existing = self._games.get((match.id, started[1]))
                    if after_game_id is None or existing is None or not existing._ended:
                        return await self.connect_game(match, game_id=started[1])
                latest = self._matches.get(match.token, match)
                candidates = latest.games
                if after_game_id is not None:
                    candidates = candidates[candidates.index(after_game_id)+1:] if after_game_id in candidates else ()
                if candidates:
                    return await self.connect_game(latest, game_id=candidates[-1])
                self._matches_changed.clear()
                await self._matches_changed.wait()
        return await asyncio.wait_for(wait(), self.timeout if timeout is None else timeout)

    async def submit_sideboard(self, match: Match, deck: Deck | tuple[DeckItem, ...]) -> SideboardResult:
        """Submit the caller's complete revised deck; never retry automatically.

        The server validates sideboarding and decides when to start the next
        game. This does not join a queue or tournament. Inventory is unchanged.
        """
        self._require_login()
        known = self._matches.get(match.token)
        if known is None or known.id != match.id or self.user_id not in known.metadata.get('player_ids', ()):
            raise ValueError('Match is not an ordinary match owned by this session')
        cards = deck.cards if isinstance(deck, Deck) else tuple(deck)
        if not cards or any(not isinstance(c, DeckItem) or c.quantity <= 0 for c in cards):
            raise ValueError('Supply a nonempty complete deck with positive quantities')
        async with self._sideboard_lock:
            games = [g for (m,_),g in tuple(self._games.items()) if m == match.id]
            if any(not game._ended for game in games):
                raise ValueError('Finish the current game before submitting a sideboard')
            for game in games:
                await game.detach()
            opcode, payload = await self._request('FlsGameSubmitSideboardMessage',
                match.token.bytes_le + struct.pack('<ii', match.id, len(cards))
                + b''.join(card.encode() for card in cards),
                lambda op,data: op in (969,970,971) and data[:20] == match.token.bytes_le + struct.pack('<i',match.id))
            c = Cursor(payload)
            c.take(20)
            errors = tuple(ValidationError(*struct.unpack('<iIii?', c.take(17)))
                           for _ in range(count(c))) if opcode == 970 else ()
            c.end()
            return SideboardResult(match.id, opcode == 969, errors, opcode == 971)

    async def concede_match(self, match: Match):
        """Explicitly concede this ordinary match; wait for server confirmation."""
        self._require_login()
        await self._request('FlsMatchConcedeMessage', match.token.bytes_le + struct.pack('<i', match.id),
            lambda op, data: op == 108 and data[:16] == match.token.bytes_le)

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
        self._submitted_parts = set()
        self._streaming = False
        self._ended = False
        self.result = None
        self.player_ids = client._player_orders.get((match_id, game_id), ())
        self.observation = None
        self.revision = 0

    @property
    def ended(self):
        return self._ended

    @property
    def player_index(self):
        """This account's index in the game server's player order, if known."""
        try:
            return self.player_ids.index(self.client.user_id)
        except ValueError:
            return None

    def _check(self):
        self.client._check()
        if self._failure is not None:
            raise Disconnected('Game stream ended') from self._failure

    def _emit(self, event):
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            raise EventOverflow('Game consumer fell behind') from None

    async def _set_player_order(self, order):
        async with self._lock:
            self.player_ids = order
            if self.observation is not None:
                self.revision += 1
                old = self.observation
                decision = replace(old.decision_id, revision=self.revision) if old.decision_id else None
                self.observation = replace(old, revision=self.revision, decision_id=decision,
                    selectors=prompt_selectors(old.prompt or {}, order, self.client.choice_catalog))
                self._submitted_parts.clear()
                self._changed.set()
                self._emit(self.observation)

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
                decision = (DecisionID(self.client.session_id, self.id, self.revision,
                            prompt['timestamp'] if prompt else self._state.status['timestamp'])
                            if (prompt is not None or self._state.damage or self._state.ordering) and not self._ended else None)
                objects = {}
                for card in self._state.cards.values():
                    props = self.client.choice_catalog.state_properties(card['properties'])
                    controller = props.get('CONTROLLER')
                    if controller is None and card['id'] < len(self.player_ids):
                        controller = card['id']
                    objects[card['id']] = GameObject(card['id'], props.get('ZONE'), controller, props.get('OWNER'),
                        props.get('CARDTEXTURE_NUMBER'), freeze(props.get('COUNTERS_LIST', {})), freeze(props),
                        tuple(value for _, value in indexed(props, 605240576)))
                actions = tuple(CardAction(card['id'], code,
                    props.get('ACTION_NAME_OVERRIDE_STRING') or card['properties'].get('ACTION_NAME_STRING')
                    or props.get('ACTION_NAME_STRING') or '', freeze(props),
                    target_groups(props, card['properties']), mode_spec(props, card['properties']),
                    card['properties'].get('ACTION_CAST_OPTIONS_STRING') or props.get('ACTION_CAST_OPTIONS_STRING', ''),
                    bool((card['properties'].get('ACTION_FLAGS') or props.get('ACTION_FLAGS', 0)) & 32),
                    bool(props.get('ACTION_STACK_THIS')), props.get('ATTACK_VICTIM_ID'),
                    bool(props.get('ACTION_IS_NON_GAME_ACTION')),
                    bool(card['properties'].get('ALTACTION_DONT_RECALCULATE') or props.get('ALTACTION_DONT_RECALCULATE')),
                    bool(card['properties'].get('ALTACTION_TAP_LOCALLY') or props.get('ALTACTION_TAP_LOCALLY')))
                    for card in self._state.cards.values() for code, props in action_entries(card['properties']))
                self.observation = Observation(decision, self.revision, freeze(prompt),
                    freeze(self._state.cards), freeze(self._state.players), actions,
                    freeze(self._state.status), self._state.unknown_elements, self._state.raw,
                    prompt_selectors(prompt or {}, self.player_ids, self.client.choice_catalog), freeze(objects),
                    self._state.mana, self._state.damage, self._state.ordering,
                    self._state.commanders, self._state.dungeons, self._state.auto_yields)
                self._submitted_parts.clear()
                self._changed.set()
                self._emit(self.observation)
            else:
                result = None
                if opcode == 4485:
                    c = Cursor(payload)
                    game_id,moved_first,status = struct.unpack('<IiB',c.take(9))
                    number = count(c)
                    if game_id != self.id or number > 16:
                        raise ProtocolError('Game result ID or player count mismatch')
                    result = GameResult(game_id,moved_first,status,
                        tuple(PlayerResult(*struct.unpack('<BI',c.take(5))) for _ in range(number)))
                    c.end()
                    self.result = result
                if opcode in (4632, 4485):
                    self._ended = True
                    self._changed.set()
                    self._submitted = self.observation.decision_id if self.observation else None
                name = self.client.catalog.by_opcode.get(opcode, {}).get('name', f'opcode_{opcode}')
                self._emit(GameMessage(opcode, name, payload, result))

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
                if self._ended:
                    raise GameEnded(self.id, self.result)
                self._changed.clear()
                await self._changed.wait()
        return await asyncio.wait_for(wait(), timeout)

    async def submit(self, decision_id: DecisionID, choice) -> Submission:
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
            if obs.prompt and self.player_index is not None and obs.prompt['player'] not in (self.player_index, 255):
                raise UnsupportedInteraction('This account is not the prompted player')
            name, payload, part = self._encode_choice(obs, choice, decision_id)
            if part is not None:
                if part in self._submitted_parts:
                    raise StaleDecision('This part of the decision was already submitted')
                self._submitted_parts.add(part)
            else:
                self._submitted = decision_id
            await self._send(name, payload)
            return Submission(decision_id)

    def _encode_choice(self, obs, choice, decision_id):
        if isinstance(choice, PromptChoice):
            prompt = obs.prompt
            if prompt is None:
                raise UnsupportedInteraction('No prompt is advertised')
            allowed = any(b['enabled'] and b['code'] == choice.code for b in prompt['buttons'])
            allowed |= any(m['flags'] & choice.code == choice.code and choice.code != 0 for m in prompt['mana'])
            allowed |= bool(prompt.get('flags', 0) & choice.code == choice.code and choice.code != 0)
            if not allowed:
                raise UnsupportedInteraction('Prompt code is not advertised; use send_raw for unmodeled interactions')
            name = 'ExtraPromptActionMessage' if choice.extra else 'PromptActionMessage'
            return name, struct.pack('<III', choice.code, choice.parameter, decision_id.timestamp), None
        return encode_choice(obs, choice, player_index=self.player_index)

    async def submit_batch(self, decision_id: DecisionID, choices) -> tuple[Submission, ...]:
        """Validate a complete locally staged sequence before writing any part.

        All except an optional final choice must be advertised local actions,
        mana payments, damage assignments, or orderings. No server update may
        be needed between them. Atomic validation does not mean atomic delivery:
        a failed/partial transport write ends the session and is never retried.
        """
        choices = tuple(choices)
        if not choices or len(choices) > 512:
            raise ValueError('Supply between 1 and 512 staged choices')
        async with self._lock:
            self._check()
            obs = self.observation
            if (self._ended or obs is None or obs.decision_id != decision_id
                    or decision_id is None or self._submitted == decision_id):
                raise StaleDecision('Decision is outdated or already submitted')
            if obs.prompt and self.player_index is not None and obs.prompt['player'] not in (self.player_index,255):
                raise UnsupportedInteraction('This account is not the prompted player')
            encoded = [self._encode_choice(obs, choice, decision_id) for choice in choices]
            parts = set(self._submitted_parts)
            for i, (_, _, part) in enumerate(encoded):
                if part is None and i != len(encoded)-1:
                    raise ValueError('Only the last choice may finish a decision')
                if part is not None:
                    if part in parts:
                        raise StaleDecision('This part of the decision was already submitted')
                    parts.add(part)
            for name,payload,part in encoded:
                if part is None:
                    self._submitted = decision_id
                else:
                    self._submitted_parts.add(part)
                await self._send(name,payload)
            return tuple(Submission(decision_id) for _ in encoded)

    async def _control(self, name, payload=b'', *, expected_revision=None, invalidate=False):
        async with self._lock:
            self._check()
            if self._ended:
                raise StaleDecision('Game is over')
            if expected_revision is not None and expected_revision != self.revision:
                raise StaleDecision('Game revision changed')
            if self.player_index is None:
                raise UnsupportedInteraction('The account is not in the known player order')
            if invalidate and self.observation is not None:
                self._submitted = self.observation.decision_id
            await self._send(name,payload)
            return ControlReceipt(name,self.id,self.revision)

    async def undo(self, *, expected_revision=None):
        """Request undo; the server decides whether the prior action is undoable."""
        return await self._control('UndoRequestMessage', expected_revision=expected_revision, invalidate=True)

    async def yield_control(self, control: YieldControl, *, expected_revision=None):
        """Send a native toggle, or DISABLE_ALL. This is not an idempotent setter."""
        key = YieldControl(control)
        return await self._control('FunctionKeyClickedMessage',bytes([key]),expected_revision=expected_revision)

    async def remove_yield(self, yield_id: int, *, expected_revision=None):
        entry = next((y for y in self.observation.auto_yields if y.id == yield_id),None) if self.observation else None
        if entry is None:
            raise UnsupportedInteraction('Yield is not advertised in the current observation')
        return await self._control('RemoveYieldFromObjectMessage',struct.pack('<ii',entry.id,entry.player_mask),
                                   expected_revision=expected_revision if expected_revision is not None else self.revision)

    async def update_stops(self, *, checksum, stops=(), yields=(), preferences=0, expected_revision=None):
        stops, yields = tuple(stops),tuple(yields)
        if any(not 0 <= entry.player_id < len(self.player_ids) for entry in (*stops,*yields)):
            raise ValueError('Stop preferences reference a player outside this game')
        return await self._control('UpdateYieldStopsMessage',stop_payload(checksum,stops,yields,preferences),
                                   expected_revision=expected_revision)

    async def concede(self, *, expected_revision=None):
        """Concede this game; observe subsequent server results for confirmation."""
        return await self._control('GameConcedeMessage',b'\0',expected_revision=expected_revision,invalidate=True)

    async def detach(self):
        """Close this game route while retaining the authenticated account session."""
        async with self._lock:
            self._check()
            match = next((m for m in self.client._matches.values() if m.id == self.match_id),None)
            if match is None:
                raise ValueError('Match token is not known')
            await self.client._send('GsCloseMatchMessage',match.token.bytes_le + struct.pack('<ii',self.match_id,self.id))
            self.client._games.pop((self.match_id,self.id),None)
            self._failure = Disconnected('Game route detached')
            self._changed.set()
            with suppress(asyncio.QueueFull):
                self._queue.put_nowait(None)

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
