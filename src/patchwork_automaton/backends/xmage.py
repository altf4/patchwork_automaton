"""Headless XMage adapter through the optional Nexus protocol-v1 Java gateway.

The Java gateway connects to an unmodified XMage server. No Java object
serialization, GUI, game rules, or policy is implemented in this module.
"""
import asyncio
from collections import deque
from collections.abc import Mapping
import json
import uuid
import zlib

from ..core import (Capabilities, Decision, DecisionToken, DeckSpec, Handle,
                    ObjectView, Observation, Option, Outcome, PlayerView, Response)
from ..errors import Disconnected, EventOverflow, GameEnded, StaleDecision, UnsupportedInteraction
from ..properties import freeze
from ..protocol import ProtocolError

MAX_MESSAGE = 16 * 1024 * 1024
PROMPTS = frozenset(('GAME_ASK','GAME_SELECT','GAME_TARGET','GAME_CHOOSE_ABILITY',
    'GAME_CHOOSE_PILE','GAME_CHOOSE_CHOICE','GAME_PLAY_MANA','GAME_PLAY_XMANA',
    'GAME_GET_AMOUNT','GAME_GET_MULTI_AMOUNT'))


class XMageError(Exception):
    def __init__(self, action, code, detail):
        self.action, self.code = action, code
        super().__init__(f'{action}: {code}: {detail}')


def decode_message(raw):
    if isinstance(raw, bytes) and raw[:2] == b'\x1f\x8b':
        decompressor = zlib.decompressobj(16+zlib.MAX_WBITS)
        raw = decompressor.decompress(raw,MAX_MESSAGE+1)
        if len(raw)>MAX_MESSAGE or not decompressor.eof or decompressor.unused_data:
            raise ProtocolError('Oversized or incomplete gateway frame')
    if len(raw)>MAX_MESSAGE:
        raise ProtocolError('Gateway frame exceeds limit')
    def bad_constant(value):
        raise ProtocolError('Nonfinite JSON number')
    try:
        item = json.loads(raw,parse_constant=bad_constant)
    except (ValueError,UnicodeError) as error:
        raise ProtocolError('Invalid gateway JSON') from error
    if not isinstance(item,dict) or item.get('type') not in ('event','result','lobby','info','error','connected','disconnected','serverLink'):
        raise ProtocolError('Unknown gateway envelope')
    return item


class XMageClient:
    capabilities = Capabilities('xmage',frozenset(('tables','sideboarding','concede','private_views','sequential_choices')))

    def __init__(self, *, proxy_url='ws://127.0.0.1:8787', host='127.0.0.1', port=17171,
                 timeout=30, connector=None, max_games=256, on_observation=None):
        if timeout<=0 or max_games<1:
            raise ValueError('Timeout and maximum games must be positive')
        self.proxy_url,self.host,self.port,self.timeout = proxy_url,host,port,timeout
        self._connector = connector
        self._ws = None
        self._reader = None
        self._pending = {}
        self._games = {}
        self._tables = {}
        self._starts = {}
        self._changed = asyncio.Event()
        self._failure = None
        self._closed = False
        self._logged_in = False
        self._counter = 0
        self._max_games = max_games
        # Optional fast synchronous listener, called with this seat's view only.
        self._on_observation = on_observation
        self.session_id = str(uuid.uuid4())
        self.server_info = None
        self.sideboards = {}
        self.user_requests = {}
        self.diagnostics = deque(maxlen=64)

    @property
    def connected(self):
        return self._logged_in and not self._closed and self._failure is None

    async def __aenter__(self):
        return self

    async def __aexit__(self,*exc):
        await self.close()

    def _check(self):
        if self._closed or self._failure is not None or self._ws is None:
            raise Disconnected('XMage gateway session ended') from self._failure

    def _fail(self,error):
        if self._failure is not None:
            return
        self._failure = error
        self._logged_in = False
        for future in self._pending.values():
            if not future.done():
                future.set_exception(Disconnected('XMage gateway session ended'))
        for game in self._games.values():
            game._changed.set()
        self._changed.set()

    async def login(self, username, password='', *, allow_attach=False):
        if self._ws is not None or self._closed:
            raise RuntimeError('Create a new client for a new session')
        connector = self._connector
        if connector is None:
            try:
                from websockets.asyncio.client import connect
            except ImportError as error:
                raise ImportError('Install patchwork-automaton[xmage] for WebSocket support') from error
            connector = connect
        try:
            self._ws = await connector(self.proxy_url,origin='http://localhost',max_size=MAX_MESSAGE,
                                       max_queue=64,open_timeout=self.timeout)
            self._reader = asyncio.create_task(self._read_loop(),name='xmage-reader')
            result = await self.request('connect',host=self.host,port=self.port,username=username,password=password)
            if result.get('attached') and not allow_attach:
                # Closing only this socket must not log another controller out.
                raise UnsupportedInteraction('Existing proxy session requires explicit allow_attach=True')
            self._logged_in = True
            self.server_info = freeze(await self.request('getServerInfo'))
            if (self.server_info.get('protocolVersion') != '1'
                    or not self.server_info.get('version','').startswith('1.4.62-V1')):
                raise UnsupportedInteraction('This adapter is pinned to XMage 1.4.62-V1 and gateway protocol 1')
            return self.server_info
        except BaseException:
            await self.close()
            raise

    async def request(self, command, **args):
        """Explicit gateway escape hatch; callers own argument semantics.

        Native sendPlayer* calls bypass decision validation: use game.submit()
        for policy responses. No request or game response is automatically retried.
        """
        self._check()
        if len(self._pending)>=256:
            raise EventOverflow('Too many pending gateway requests')
        self._counter+=1
        rid = str(self._counter)
        future = asyncio.get_running_loop().create_future()
        self._pending[rid]=future
        try:
            raw=json.dumps({'requestId':rid,'action':command,'args':args},allow_nan=False)
            if len(raw.encode())>MAX_MESSAGE:
                raise ProtocolError('Outgoing gateway frame exceeds limit')
            try:
                await self._ws.send(raw)
            except BaseException as error:
                self._fail(error)
                raise
            try:
                result=await asyncio.wait_for(future,self.timeout)
            except (TimeoutError,asyncio.CancelledError) as error:
                self._fail(error)
                raise
            if result.get('action') != command:
                error=ProtocolError('Gateway result action does not match request')
                self._fail(error)
                raise error
            if result.get('ok') is not True:
                raise XMageError(command,result.get('errorCode','FAILED'),result.get('error','Request rejected'))
            return result.get('data')
        finally:
            self._pending.pop(rid,None)
            future.cancel()
            if future.done() and not future.cancelled():
                future.exception()

    async def _read_loop(self):
        try:
            async for raw in self._ws:
                await self._dispatch(decode_message(raw))
            if not self._closed:
                self._fail(Disconnected('Gateway closed its stream'))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._fail(error)
            await self._ws.close()

    async def _dispatch(self,item):
        kind=item['type']
        if kind=='result':
            future=self._pending.get(str(item.get('requestId')))
            if future and not future.done():
                future.set_result(item)
        elif kind=='disconnected' or kind=='serverLink' and item.get('state')!='restored':
            # The upstream proxy may relink automatically. Do not let a relink
            # continue an RL episode with stale response identities.
            raise Disconnected('XMage server link changed; start a new episode')
        elif kind=='event':
            method=item.get('method')
            data=item.get('data')
            if method=='START_GAME' and isinstance(data,dict):
                game_id=data.get('gameId') or item.get('objectId')
                game=self.game(game_id)
                game.player_id=data.get('playerId') or game.player_id
                self._starts[data.get('currentTableId')]=game_id
                self._changed.set()
            elif method=='USER_REQUEST_DIALOG' and isinstance(data,dict):
                if len(self.user_requests)>=64:
                    raise EventOverflow('Too many unanswered user requests')
                self.user_requests[item['messageId']]=freeze(data)
                self._changed.set()
            elif method=='SIDEBOARD' and isinstance(data,dict):
                self.sideboards[data.get('currentTableId')]=freeze(data)
                self._changed.set()
            elif method in PROMPTS or isinstance(method,str) and (method.startswith('GAME_') or method=='END_GAME_INFO'):
                game_id=item.get('objectId')
                if game_id:
                    game = self.game(game_id)
                    game._receive(item)
                    if self._on_observation is not None and game.player_id is not None:
                        self._on_observation(await game.snapshot())
            else:
                self.diagnostics.append(freeze(item))
        elif kind=='error':
            self.diagnostics.append(freeze(item))

    def game(self,game_id):
        game_id=str(uuid.UUID(str(game_id)))
        if game_id not in self._games:
            if len(self._games)>=self._max_games:
                raise EventOverflow('Release completed game handles before opening more games')
            self._games[game_id]=XMageGame(self,game_id)
        return self._games[game_id]

    async def resolve_deck(self, deck: DeckSpec):
        resolved=await self.request('resolvePrintings',names=list(dict.fromkeys(c.name for c in deck.cards)))
        if not resolved or not resolved.get('ready'):
            raise UnsupportedInteraction('Proxy card database is not ready')
        by_name={entry['name']:entry for entry in resolved['results']}
        entries={'name':deck.name,'cards':[],'sideboard':[]}
        for card in deck.cards:
            match=by_name.get(card.name,{})
            if not match.get('found'):
                raise ValueError(f'Card is not implemented in this XMage catalog: {card.name}')
            entries['sideboard' if card.sideboard else 'cards'].append(dict(cardName=match['cardName'],
                setCode=card.set_code or match['setCode'],cardNumber=card.collector_number or match['cardNumber'],amount=card.quantity))
        return entries

    async def create_table(self, *, name='Patchwork local game', deck_type='Constructed - Modern',
                           game_type='Two Player Duel', wins_needed=1, **options):
        if wins_needed not in (1,2):
            raise ValueError('wins_needed must be 1 or 2')
        table=await self.request('createTable',name=name,gameType=game_type,deckType=deck_type,winsNeeded=wins_needed,
            playerTypes=['HUMAN','HUMAN'],rated=False,spectatorsAllowed=False,timeLimit='NONE',**options)
        self._tables[table['tableId']]=freeze(table)
        return freeze(table)

    async def join_table(self,table_id,deck: DeckSpec,*,player_name='Patchwork'):
        return await self.request('joinTable',tableId=table_id,playerName=player_name,playerType='HUMAN',deck=await self.resolve_deck(deck))

    async def start_match(self,table_id):
        return await self.request('startMatch',tableId=table_id)

    async def wait_for_game(self,table_id,*,after_game_id=None,timeout=None):
        async def wait():
            while True:
                self._check()
                game_id=self._starts.get(table_id)
                if game_id and game_id!=after_game_id:
                    game=self.game(game_id)
                    if not game.joined:
                        await self.request('joinGame',gameId=game_id)
                        game.joined=True
                    return game
                self._changed.clear()
                await self._changed.wait()
        return await asyncio.wait_for(wait(),self.timeout if timeout is None else timeout)

    async def submit_sideboard(self,table_id,deck: DeckSpec):
        if table_id not in self.sideboards:
            raise UnsupportedInteraction('No sideboard request is pending for this table')
        result=await self.request('submitDeck',tableId=table_id,deck=await self.resolve_deck(deck))
        self.sideboards.pop(table_id,None)
        return result

    async def respond_user_request(self,message_id,button):
        """Answer an explicitly retrieved permission dialog, separately from play."""
        self._check()
        request=self.user_requests.get(message_id)
        if request is None or button not in (1,2,3):
            raise StaleDecision('User request is absent or button invalid')
        if not request.get(f'button{button}Text'):
            raise UnsupportedInteraction('User-request button is not offered')
        action=request.get(f'button{button}Action')
        result=None
        if action:
            result=await self.request('sendPlayerAction',gameId=request['gameId'],action=action,data=request.get('relatedUserId'))
        self.user_requests.pop(message_id,None)
        return result

    async def remove_table(self,table_id):
        result=await self.request('removeTable',tableId=table_id)
        self._tables.pop(table_id,None)
        return result

    def release_game(self,game_id):
        game=self._games.get(game_id)
        if game and not game.terminal:
            raise ValueError('Finish the game before releasing it')
        self._games.pop(game_id,None)

    async def close(self):
        if self._closed:
            return
        if self.connected:
            try:
                await self.request('disconnect')
            except Exception:
                pass
        self._closed=True
        self._fail(Disconnected('Client closed'))
        if self._ws:
            await self._ws.close()
        if self._reader:
            self._reader.cancel()
            await asyncio.gather(self._reader,return_exceptions=True)


class XMageGame:
    capabilities=XMageClient.capabilities

    def __init__(self,client,game_id):
        self.client,self.id=client,game_id
        self.handle=Handle('xmage','game',game_id)
        self.player_id=None
        self.joined=False
        self.revision=0
        self.generation=0
        self.terminal=False
        self.outcome=None
        self._view={}
        self._decision=None
        self._bindings={}
        self._last_view_id=-1
        self._last_prompt_id=-1
        self._seen=deque(maxlen=256)
        self._changed=asyncio.Event()
        self._lock=asyncio.Lock()

    def _receive(self,event):
        method=event['method'];data=event.get('data') or {};message_id=event.get('messageId')
        if type(message_id) is not int:
            raise ProtocolError('Callback messageId must be an integer')
        key=(method,message_id)
        if key in self._seen:
            return
        self._seen.append(key)
        if method=='GAME_ERROR':
            self.client._fail(ProtocolError('XMage reported a game error'))
            return
        if method not in PROMPTS | {'GAME_INIT','GAME_UPDATE','GAME_UPDATE_AND_INFORM','GAME_INFORM_PERSONAL','GAME_OVER','END_GAME_INFO','GAME_REDRAW_GUI'}:
            raise UnsupportedInteraction('Unknown XMage callback: '+method)
        if not isinstance(data,dict):
            return
        view=data.get('gameView')
        if view is None and 'myHand' in data:
            view=data
        if isinstance(view,dict) and message_id>=self._last_view_id:
            self._last_view_id=message_id
            self._view=view
            self.player_id=view.get('myPlayerId') or self.player_id
            self.revision+=1
        if method in ('GAME_OVER','END_GAME_INFO'):
            self.terminal=True
            self._decision=None
            if method=='END_GAME_INFO':
                self.outcome=Outcome(data.get('won') if type(data.get('won')) is bool else None,
                                     data.get('draw') if type(data.get('draw')) is bool else None,freeze(data))
        elif method in PROMPTS and not self.terminal and message_id>self._last_prompt_id:
            self._last_prompt_id=message_id
            self._offer(method,data)
        self._changed.set()

    def _offer(self,method,data):
        from .xmage_choices import choices
        self.generation+=1
        options,self._bindings=choices(method,data,self._view)
        actor=Handle('xmage','player',str(self.player_id or 'pending'))
        token=DecisionToken('xmage',self.client.session_id,self.handle,actor,self.generation)
        self._decision=Decision(token,method,data.get('message') or '',tuple(options),freeze(data))

    async def snapshot(self):
        self.client._check()
        from .xmage_choices import object_views
        actor=Handle('xmage','player',str(self.player_id or 'pending'))
        players=tuple(PlayerView(Handle('xmage','player',p['playerId']),p.get('name',''),p.get('life'),
            p.get('handCount'),p.get('libraryCount'),freeze({c.get('name',str(i)):c.get('count') for i,c in enumerate(p.get('counters',()))}),freeze(p))
            for p in self._view.get('players',()))
        active=self._view.get('activePlayerId')
        priority=next((p.get('playerId') for p in self._view.get('players',()) if p.get('hasPriority')),None)
        return Observation('xmage',self.handle,actor,self.revision,players,
            freeze(object_views(self._view)),self._decision,self.terminal,self.outcome,freeze(self._view),
            turn=self._view.get('turn'),phase=(self._view.get('step') or '').lower() or None,
            active_player=Handle('xmage','player',active) if active else None,
            priority_player=Handle('xmage','player',priority) if priority else None)

    async def wait_for_decision(self,*,timeout=None):
        async def wait():
            while True:
                self.client._check()
                if self._decision is not None or self.terminal:
                    return await self.snapshot()
                self._changed.clear()
                await self._changed.wait()
        return await asyncio.wait_for(wait(),timeout)

    async def submit(self,token,response: Response):
        async with self._lock:
            self.client._check()
            decision=self._decision
            if self.terminal or decision is None or decision.token!=token:
                raise StaleDecision('XMage decision is outdated or already submitted')
            if response.option not in self._bindings:
                raise UnsupportedInteraction('Option is not offered in this decision')
            from .xmage_choices import encode_response
            command,args=encode_response(self._bindings[response.option],response.value)
            self._decision=None
            try:
                return await self.client.request(command,gameId=self.id,**args)
            except XMageError:
                # A definite rejection is observable, but a retry needs a new
                # token. Preserve any newer callback that arrived meanwhile.
                if self._decision is None and not self.terminal:
                    self._offer(decision.kind,dict(decision.native))
                    self._changed.set()
                raise

    async def wait_for_result(self,*,timeout=None):
        async def wait():
            while True:
                self.client._check()
                if self.outcome is not None:return await self.snapshot()
                self._changed.clear()
                await self._changed.wait()
        return await asyncio.wait_for(wait(),timeout)

    async def concede(self):
        self.client._check()
        if self.terminal:
            raise GameEnded(self.id,self.outcome)
        self._decision=None
        return await self.client.request('sendPlayerAction',gameId=self.id,action='CONCEDE',data=None)
