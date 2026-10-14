"""Opt-in common gameplay adapter. The existing MTGO public API is unchanged."""
import asyncio

from .. import actions as native
from ..client import MTGOClient
from ..core import (Capabilities, Decision, DecisionToken, DeckSpec, Handle,
                    ObjectView, Observation, Option, Outcome, PlayerView, Response)
from ..errors import GameEnded, StaleDecision, UnsupportedInteraction
from ..game import DeckItem
from ..properties import freeze


def resolve_deck(deck: DeckSpec):
    if any(c.mtgo_catalog_id is None for c in deck.cards):
        raise ValueError('MTGO decks need explicit catalog IDs and collection permissions')
    return tuple(DeckItem(c.mtgo_catalog_id,c.mtgo_annotation,c.mtgo_permission,c.quantity,c.sideboard) for c in deck.cards)


class MTGOBackend:
    capabilities=Capabilities('mtgo',frozenset(('challenge','solitaire','sideboarding','native_local_batches','concede','private_views')))

    def __init__(self,client=None,**options):
        self.client=client or MTGOClient(**options)

    async def login(self,username,password):
        return await self.client.login(username,password)

    async def close(self):
        await self.client.close()

    async def __aenter__(self):return self
    async def __aexit__(self,*args):await self.close()

    def game(self,native_game):
        if native_game.client is not self.client:
            raise ValueError('Game belongs to another account session')
        return MTGOGame(native_game)

    async def create_solitaire(self,deck: DeckSpec,*,format):
        return self.game(await self.client.create_solitaire(resolve_deck(deck),format=format))

    async def create_match(self,opponent,first_deck,second_deck,*,format,best_of=1):
        match=await self.client.challenge(opponent.client.user_id,resolve_deck(first_deck),format=format,best_of=best_of)
        incoming=await opponent.client.wait_for_match(match.token)
        first,second=await asyncio.gather(self.client.wait_for_game(match),opponent.client.accept_challenge(incoming,resolve_deck(second_deck)))
        return self.game(first),opponent.game(second)


class MTGOGame:
    capabilities=MTGOBackend.capabilities

    def __init__(self,game):
        self.native_game=game
        self.handle=Handle('mtgo','game',str(game.id))
        self._generation=0
        self._revision=-1
        self._token=None
        self._bindings={}
        self._lock=asyncio.Lock()

    def _decision(self,obs,actor):
        game=self.native_game
        if (obs.decision_id is None or game.ended or game.player_index is None
                or obs.prompt and obs.prompt['player'] not in (game.player_index,255)
                or game._submitted==obs.decision_id):
            return None
        if self._revision!=obs.revision or self._token is None:
            self._generation+=1
            self._revision=obs.revision
            self._token=DecisionToken('mtgo',str(game.client.session_id),self.handle,actor,self._generation)
        options=[];self._bindings={}
        def add(key,kind,label,binding,**constraints):
            options.append(Option(key,kind,label,freeze(constraints)));self._bindings[key]=binding
        for button in (obs.prompt or {}).get('buttons',()):
            if button['enabled']:
                add('button:'+str(button['code']),'boolean',button['label'],('fixed',native.PromptChoice(button['code'])))
        for action in obs.actions:
            if action.non_game or ('card',action.card_id,action.action_id) in game._submitted_parts:
                continue
            add(f'card:{action.card_id}:{action.action_id}','card',action.label,('card',action),
                object=str(action.card_id),targets=[{'minimum':g.minimum,'maximum':g.maximum,
                    'candidates':[str(t) for t in g.legal_targets],'deferred':g.deferred,'total':g.distributed_total} for g in action.target_groups],
                modes={'minimum':action.modes.minimum,'maximum':action.modes.maximum,
                       'options':[{'index':o.index,'label':o.label,'maximum':o.maximum,'weight':o.weight} for o in action.modes.options]}
                       if action.modes else None)
        for selector in obs.selectors:
            if selector.kind=='unknown':continue
            if selector.options:
                for option in selector.options:
                    add(f'selector:{selector.id}:{option.value}','string',option.label,('fixed',native.SelectorChoice(selector.id,option.value)))
            else:
                add(f'selector:{selector.id}','integer',selector.kind,('selector',selector),minimum=selector.minimum,maximum=selector.maximum,
                    catalog_required=selector.catalog_required)
        for mana in obs.mana:
            if mana.player==game.player_index and mana.amount>0 and mana.color_flag and ('mana',int(mana.color_flag),mana.id) not in game._submitted_parts:
                add(f'mana:{mana.id}','mana',str(mana.color_flag),('fixed',native.ManaChoice(mana.color_flag,mana.id)),
                    source=mana.source_id,snow=mana.snow,restriction=mana.restriction)
        for damage in obs.damage:
            if ('damage',damage.source_id) not in game._submitted_parts:
                add('damage:'+str(damage.source_id),'distribution','Assign damage',('damage',damage),
                    total=damage.total,recipients=[{'id':str(t.id),'minimum':t.minimum,'maximum':t.maximum} for t in damage.targets])
        for order in obs.ordering:
            if ('ordering',order.source_id) not in game._submitted_parts:
                add('order:'+str(order.source_id),'ordering','Order objects',('ordering',order),candidates=[str(t) for t in order.targets])
        return Decision(self._token,'decision',(obs.prompt or {}).get('text',''),tuple(options))

    async def snapshot(self):
        game=self.native_game;obs=game.observation
        if obs is None:raise UnsupportedInteraction('Game has no observation yet')
        actor=Handle('mtgo','player',str(game.player_index) if game.player_index is not None else 'pending')
        objects={}
        shared={1000961:'battlefield',1001728:'stack',1247745:'command',1475255:'mutate'}
        private={1000192:'graveyard',1000448:'hand',1000704:'library',1001472:'sideboard',1011456:'exile'}
        for obj in obs.objects.values():
            handle=Handle('mtgo','object',str(obj.id));zone=shared.get(obj.zone,private.get((obj.zone or 0)&0xffffff00,'unknown'))
            objects[handle]=ObjectView(handle,None if obj.properties.get('face_down') else obj.properties.get('CARDNAME_STRING'),zone,
                Handle('mtgo','player',str(obj.controller)) if obj.controller is not None else None,obj.properties)
        outcome=None
        if game.result and game.player_index is not None:
            winners=game.result.winner_indices
            outcome=Outcome(game.player_index in winners if winners else None,None,freeze({'status':game.result.status,
                'results':[{'code':p.code,'playing_time':p.playing_time} for p in game.result.players]}))
        summary=obs.players[0] if obs.players else {}
        def count(key,index):
            values=summary.get(key,())
            return values[index] if index<len(values) else None
        players=tuple(PlayerView(Handle('mtgo','player',str(i)),str(user_id),count('life',i),count('hand',i),count('library',i),
            obs.objects[i].counters if i in obs.objects else freeze({})) for i,user_id in enumerate(game.player_ids))
        phase_number=(obs.prompt or {}).get('phase')
        phases={1:'untap',2:'upkeep',3:'draw',4:'precombat_main',5:'begin_combat',6:'declare_attackers',
            7:'declare_blockers',8:'combat_damage',9:'end_combat',10:'postcombat_main',11:'end_turn',12:'cleanup'}
        active=summary.get('active_player');priority=obs.status.get('priority_player')
        return Observation('mtgo',self.handle,actor,obs.revision,players,freeze(objects),self._decision(obs,actor),
            game.ended,outcome,freeze({'status':obs.status,'prompt':obs.prompt}),
            turn=(obs.prompt or {}).get('turn'),phase=phases.get(phase_number),
            active_player=Handle('mtgo','player',str(active)) if active is not None else None,
            priority_player=Handle('mtgo','player',str(priority)) if priority is not None and priority>=0 else None)


    async def wait_for_decision(self,*,timeout=None):
        async def wait():
            while True:
                obs=await self.snapshot()
                if obs.decision or obs.terminal:return obs
                try:await self.native_game.wait_for_update(obs.revision)
                except GameEnded:return await self.snapshot()
        return await asyncio.wait_for(wait(),timeout)

    async def submit(self,token,response: Response):
        async with self._lock:
            obs=await self.snapshot()
            if not obs.decision or obs.decision.token!=token:
                raise StaleDecision('MTGO common decision is stale or belongs to another session')
            if response.option not in self._bindings:raise UnsupportedInteraction('Option is not offered')
            kind,value=self._bindings[response.option]
            if kind=='fixed':
                if response.value is not None:raise ValueError('Finite options do not take a value')
                choice=value
            elif kind=='selector':choice=native.SelectorChoice(value.id,response.value)
            elif kind=='card':
                payload=response.value or {}
                if not hasattr(payload,'get'):raise ValueError('Card selection takes a mapping')
                choice=native.CardSelection(value.card_id,value.action_id,
                    tuple(tuple(int(t) for t in g) for g in payload.get('targets',())),
                    tuple(payload['modes']) if payload.get('modes') is not None else None,
                    tuple(tuple(g) for g in payload.get('distributions',())),bool(payload.get('retain_priority',False)))
            elif kind=='damage':choice=native.DamageChoice(value.source_id,tuple((int(t),n) for t,n in response.value))
            elif kind=='ordering':choice=native.OrderingChoice(value.source_id,tuple(int(t) for t in response.value))
            else:raise UnsupportedInteraction(kind)
            result=await self.native_game.submit(self.native_game.observation.decision_id,choice)
            self._token=None  # Partial/local decisions receive a new common token.
            return result

    async def wait_for_result(self,*,timeout=None):
        async def wait():
            while True:
                self.native_game._check()
                if self.native_game.result is not None:return await self.snapshot()
                self.native_game._changed.clear()
                await self.native_game._changed.wait()
        return await asyncio.wait_for(wait(),timeout)

    async def concede(self):return await self.native_game.concede()
