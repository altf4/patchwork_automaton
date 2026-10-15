"""Gateway protocol and actor/decision correctness without sockets or a JVM."""
import asyncio
from dataclasses import replace
import gzip
import json
import unittest
import uuid

from patchwork_automaton.backends.xmage import XMageClient, XMageError, decode_message
from patchwork_automaton.backends.xmage_choices import choices,encode_response,object_views
from patchwork_automaton.core import CardSpec,DeckSpec,Handle,Option,Response
from patchwork_automaton.errors import Disconnected,StaleDecision,UnsupportedInteraction
from patchwork_automaton.protocol import ProtocolError

GAME='00000000-0000-0000-0000-000000000001'
ME='00000000-0000-0000-0000-000000000002'
OTHER='00000000-0000-0000-0000-000000000003'
CARD='00000000-0000-0000-0000-000000000004'
TABLE='00000000-0000-0000-0000-000000000005'


def view():
    return {'myPlayerId':ME,'myHand':{CARD:{'id':CARD,'name':'Synthetic land'}},'stack':{},
        'players':[{'playerId':ME,'name':'Alice','life':20,'handCount':1,'battlefield':{},'graveyard':{}},
                   {'playerId':OTHER,'name':'Bob','life':20,'handCount':7,'battlefield':{},'graveyard':{}}],
        'canPlayObjects':{'objects':{CARD:{'basicPlayAbilities':[{'id':CARD,'value':'Play'}]}}}}


def event(method='GAME_ASK',message_id=1,**data):
    return {'type':'event','method':method,'messageId':message_id,'objectId':GAME,
            'data':{'gameView':view(),'message':'Synthetic choice',**data}}


class Socket:
    def __init__(self):self.incoming=asyncio.Queue();self.sent=[];self.closed=False;self.reject=False;self.no_reply=False;self.wrong=False
    def __aiter__(self):return self
    async def __anext__(self):
        item=await self.incoming.get()
        if item is None:raise StopAsyncIteration
        return item
    async def send(self,raw):
        req=json.loads(raw);self.sent.append(req)
        if self.no_reply:return
        data=None
        if req['action']=='connect':data={'attached':False,'streamId':'synthetic'}
        elif req['action']=='getServerInfo':data={'protocolVersion':'1','version':'1.4.62-V1 (synthetic)'}
        response={'type':'result','requestId':req['requestId'],'action':'wrong' if self.wrong else req['action'],
                  'ok':not self.reject,'data':data,'errorCode':'REJECTED','error':'synthetic rejection'}
        await self.incoming.put(json.dumps(response))
    async def close(self):
        if not self.closed:self.closed=True;await self.incoming.put(None)


class CallbackTests(unittest.TestCase):
    def test_json_and_gzip_and_bad_frames(self):
        raw=b'{"type":"info","message":"hello"}'
        self.assertEqual(decode_message(raw),decode_message(gzip.compress(raw)))
        for raw in ('null','[]','{"type":"future"}','{"type":"info","bad":NaN}',b'\xff'):
            with self.subTest(raw=raw),self.assertRaises(ProtocolError):decode_message(raw)
        with self.assertRaises(ProtocolError):decode_message(gzip.compress(b'x'*17_000_000))

    def test_boolean_piles_and_amount_bounds(self):
        for method in ('GAME_ASK','GAME_CHOOSE_PILE'):
            options,bindings=choices(method,{},view())
            self.assertEqual([o.kind for o in options],['boolean','boolean'])
            self.assertEqual(encode_response(bindings['yes'],None),('sendPlayerBoolean',{'value':True}))
        options,bindings=choices('GAME_GET_AMOUNT',{'min':2,'max':5},view())
        self.assertEqual(encode_response(bindings['amount'],3),('sendPlayerInteger',{'value':3}))
        for value in (1,6,True,3.0):
            with self.assertRaises(ValueError):encode_response(bindings['amount'],value)

    def test_multi_amount_native_space_separated_response(self):
        data={'min':4,'max':4,'messages':[{'min':0,'max':3},{'min':1,'max':4}]}
        _,bindings=choices('GAME_GET_MULTI_AMOUNT',data,view())
        self.assertEqual(encode_response(bindings['amounts'],(1,3)),('sendPlayerString',{'value':'1 3'}))
        for value in ((1,), (4,0),(1,1),(True,3)):
            with self.assertRaises(ValueError):encode_response(bindings['amounts'],value)

    def test_target_sets_optional_done_and_ability_uuid(self):
        options,bindings=choices('GAME_TARGET',{'targets':[OTHER],'flag':True},view())
        self.assertEqual(len(options),1)
        self.assertEqual(encode_response(bindings[options[0].id],None),('sendPlayerUUID',{'value':OTHER}))
        options,_=choices('GAME_TARGET',{'options':{'possibleTargets':[CARD],'UI.right.btn.text':'Done'},'flag':True},view())
        self.assertEqual([o.kind for o in options],['uuid','boolean'])
        options,bindings=choices('GAME_CHOOSE_ABILITY',{'choices':{CARD:'Activate ability'}},view())
        self.assertEqual(encode_response(bindings[options[0].id],None)[0],'sendPlayerUUID')

    def test_trigger_order_uses_offered_ability_ids_not_source_card_ids(self):
        ability='00000000-0000-0000-0000-000000000099'
        data={'targets':None,'flag':True,'options':{'queryType':'PICK_ABILITY'},
              'cardsView1':{ability:{'sourceName':'Eidolon of the Great Revel',
                                    'sourceCard':{'id':CARD}}}}
        options,bindings=choices('GAME_TARGET',data,view())
        self.assertEqual(len(options),1)
        self.assertEqual(options[0].constraints['source'],CARD)
        self.assertEqual(options[0].constraints['selection_kind'],'trigger_order')
        self.assertEqual(encode_response(bindings[options[0].id],None),
                         ('sendPlayerUUID',{'value':ability}))

    def test_script_target_preferences_do_not_remove_self_targets_from_backend(self):
        options,bindings=choices('GAME_TARGET',{'targets':[ME,CARD],'flag':True},view())
        self.assertEqual({o.id for o in options},{'object:'+ME,'object:'+CARD})
        self.assertEqual(encode_response(bindings['object:'+ME],None)[1]['value'],ME)

    def test_key_and_string_choices_use_values_not_labels(self):
        _,bindings=choices('GAME_CHOOSE_CHOICE',{'choice':{'keyChoices':{'k':'Pretty label'},'required':True}},view())
        self.assertEqual(encode_response(bindings['choice:0'],None),('sendPlayerString',{'value':'k'}))
        _,bindings=choices('GAME_CHOOSE_CHOICE',{'choice':{'choices':['a','b'],'required':False}},view())
        self.assertIn('no',bindings)
        self.assertEqual(encode_response(bindings['choice:1'],None)[1],{'value':'b'})

    def test_priority_uses_advertised_playable_objects_and_hidden_information(self):
        state=view();state['players'][1]['sideboard']={OTHER:{'name':'Hidden opponent card'}}
        options,_=choices('GAME_SELECT',{},state)
        self.assertEqual(options[0].label,'Synthetic land')
        objects=object_views(state)
        self.assertEqual(len(objects),1)
        self.assertEqual(objects[Handle('xmage','object',CARD)].zone,'hand')
        state['myHand'][CARD]['faceDown']=True
        self.assertIsNone(object_views(state)[Handle('xmage','object',CARD)].name)

    def test_models_freeze_nested_constraints_and_response_values(self):
        original={'x':[{'y':1}]};option=Option('a','integer','A',original)
        original['x'][0]['y']=2
        self.assertEqual(option.constraints['x'][0]['y'],1)
        with self.assertRaises(TypeError):option.constraints['x'][0]['y']=3
        self.assertEqual(Response('a',[1,2]).value,(1,2))

    def test_card_identity_namespaces_do_not_mix(self):
        with self.assertRaises(ValueError):CardSpec('Mountain',1,set_code='M11')
        with self.assertRaises(ValueError):CardSpec('Mountain',0)
        with self.assertRaises(ValueError):DeckSpec(())
        from patchwork_automaton.backends.mtgo import resolve_deck
        with self.assertRaises(ValueError):resolve_deck(DeckSpec((CardSpec('Mountain',60),)))


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_clicked_source_binds_next_picker_without_guessing_between_lands(self):
        _,_,game=await self.setup_client()
        native=view()
        native['myHand'][CARD]['name']='Sacred Foundry'
        native['myHand'][TABLE]={'id':TABLE,'name':'Inspiring Vantage'}
        game._receive(event('GAME_SELECT',gameView=native,targets=[CARD]))
        obs=await game.snapshot()
        await game.submit(obs.decision.token,Response('object:'+CARD))
        game._receive(event('GAME_CHOOSE_ABILITY',message_id=2,gameView=native,
                            choices={OTHER:'1. {T}: Add {R}.'}))
        selected=(await game.snapshot()).decision.options[0]
        self.assertEqual(selected.constraints['source'],CARD)
        self.assertEqual(selected.constraints['semantic_card'],16)
        # Context is consumed; a later unrelated picker must not reuse it.
        game._receive(event('GAME_CHOOSE_ABILITY',message_id=3,gameView=native,
                            choices={OTHER:'1. {T}: Add {R}.'}))
        self.assertNotIn('semantic_ability',(await game.snapshot()).decision.options[0].constraints)

    async def test_snapshot_cache_tracks_views_and_decisions(self):
        client,_,game=await self.setup_client()
        game._receive(event())
        first=await game.snapshot()
        self.assertIs(first,await game.snapshot())
        game._receive(event('GAME_UPDATE',message_id=2))
        second=await game.snapshot()
        self.assertIsNot(first,second)
        game._receive(event('GAME_GET_AMOUNT',message_id=3,gameView=None,min=1,max=4))
        third=await game.snapshot()
        self.assertIsNot(second,third)
        self.assertEqual(third.decision.kind,'GAME_GET_AMOUNT')
        await client.close()
        with self.assertRaises(Disconnected):await game.snapshot()

    async def setup_client(self):
        socket=Socket()
        async def connector(*args,**kwargs):return socket
        client=XMageClient(connector=connector,timeout=.1)
        self.addAsyncCleanup(client.close)
        await client.login('synthetic')
        return client,socket,client.game(GAME)

    async def test_game_response_identity_and_duplicate_submission(self):
        client,socket,game=await self.setup_client()
        game._receive(event())
        obs=await game.wait_for_decision()
        with self.assertRaises(StaleDecision):await game.submit(replace(obs.decision.token,session='wrong'),Response('yes'))
        await game.submit(obs.decision.token,Response('yes'))
        self.assertEqual(socket.sent[-1]['action'],'sendPlayerBoolean')
        self.assertTrue(socket.sent[-1]['args']['value'])
        with self.assertRaises(StaleDecision):await game.submit(obs.decision.token,Response('yes'))

    async def test_updates_do_not_invalidate_pending_prompt_and_late_critical_survives(self):
        _,_,game=await self.setup_client()
        game._receive(event(message_id=4))
        old=(await game.snapshot()).decision.token
        game._receive(event('GAME_UPDATE',message_id=9))
        self.assertEqual((await game.snapshot()).decision.token,old)
        game._receive(event('GAME_GET_AMOUNT',message_id=6,min=1,max=4))
        self.assertEqual((await game.snapshot()).decision.kind,'GAME_GET_AMOUNT')
        with self.assertRaises(StaleDecision):await game.submit(old,Response('yes'))
        game._receive(event(message_id=4))
        self.assertEqual((await game.snapshot()).decision.kind,'GAME_GET_AMOUNT')

    async def test_definite_rejection_mints_new_token_without_retry(self):
        _,socket,game=await self.setup_client();game._receive(event())
        obs=await game.snapshot();socket.reject=True;before=len(socket.sent)
        with self.assertRaises(XMageError):await game.submit(obs.decision.token,Response('yes'))
        self.assertEqual(len(socket.sent),before+1)
        self.assertNotEqual((await game.snapshot()).decision.token,obs.decision.token)

    async def test_timeout_ends_session_and_does_not_retry(self):
        client,socket,game=await self.setup_client();socket.no_reply=True
        before=len(socket.sent)
        with self.assertRaises(TimeoutError):await client.request('ping')
        self.assertEqual(len(socket.sent),before+1)
        self.assertFalse(client.connected)
        with self.assertRaises(Disconnected):await game.snapshot()

    async def test_action_argument_is_not_confused_with_envelope_command(self):
        client,socket,game=await self.setup_client()
        await game.concede()
        self.assertEqual(socket.sent[-1]['action'],'sendPlayerAction')
        self.assertEqual(socket.sent[-1]['args']['action'],'CONCEDE')

    async def test_results_and_terminal_wait_are_separate_from_game_over(self):
        _,_,game=await self.setup_client();game._receive(event())
        waiting=asyncio.create_task(game.wait_for_result(timeout=1))
        await asyncio.sleep(0)
        game._receive(event('GAME_OVER',message_id=2))
        self.assertTrue((await game.wait_for_decision()).terminal)
        self.assertFalse(waiting.done())
        game._receive(event('END_GAME_INFO',message_id=3,won=True))
        result=await waiting
        self.assertTrue(result.outcome.won)

    async def test_server_relink_fails_closed_instead_of_resuming_old_decisions(self):
        client,_,game=await self.setup_client();game._receive(event())
        with self.assertRaises(Disconnected):await client._dispatch({'type':'serverLink','state':'lost'})

    async def test_unknown_callbacks_are_explicit(self):
        _,_,game=await self.setup_client()
        with self.assertRaises(UnsupportedInteraction):game._receive(event('GAME_NEW_REQUIRED_CHOICE'))

    async def test_display_dialog_does_not_replace_game_decision(self):
        client,_,game=await self.setup_client();game._receive(event())
        token=(await game.snapshot()).decision.token
        await client._dispatch(event('USER_REQUEST_DIALOG',message_id=2,gameId=GAME,button1Text='Deny',button1Action=None))
        self.assertEqual((await game.snapshot()).decision.token,token)
        await client.respond_user_request(2,1)
        self.assertFalse(client.user_requests)

    async def test_observer_receives_intermediate_private_views(self):
        client, _, game = await self.setup_client()
        snapshots = []
        client._on_observation = snapshots.append
        await client._dispatch(event('GAME_UPDATE', message_id=1))
        await client._dispatch(event('GAME_ASK', message_id=2))
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0].actor, Handle('xmage', 'player', ME))
        self.assertIsNone(snapshots[0].decision)
        self.assertEqual(snapshots[1].decision.kind, 'GAME_ASK')
        self.assertNotIn(Handle('xmage', 'object', OTHER), snapshots[0].objects)

    async def test_correlated_wrong_action_result_is_protocol_failure(self):
        client,socket,_=await self.setup_client();socket.wrong=True
        with self.assertRaises(ProtocolError):await client.request('ping')
        self.assertFalse(client.connected)

    async def test_start_notification_routes_to_correct_table_and_game(self):
        client,_,_=await self.setup_client()
        await client._dispatch({'type':'event','method':'START_GAME','messageId':1,'objectId':GAME,
            'data':{'gameId':GAME,'currentTableId':TABLE,'playerId':ME}})
        game=await client.wait_for_game(TABLE)
        self.assertEqual((game.id,game.player_id),(GAME,ME))
        self.assertTrue(game.joined)
