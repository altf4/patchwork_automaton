"""Additional native interaction families, using synthetic account/server data."""
import asyncio
from dataclasses import replace
import struct
import unittest

from patchwork_automaton import (CardSelection, ChoiceCatalog, DeckItem, Disconnected, GameEnded,
    ManaChoice, MTGOClient, PlayerStops, PromptChoice, SelectorChoice, StaleDecision,
    Step, UnsupportedInteraction, YieldControl, YieldPreference)
from patchwork_automaton.controls import stop_payload
from patchwork_automaton.game import challenge_payload
from patchwork_automaton.protocol import Cursor, ProtocolError, array
from test_client import Backend, state_packet
import test_interactions as fixtures
from test_interactions import ACTION, attrs, element, packet, prompt, targeted, thing, ui


# Reuse the setup helpers, without inheriting the existing test cases twice.
class MechanicsTests(unittest.IsolatedAsyncioTestCase):
    make_game = fixtures.InteractionAPITests.make_game
    last_message = fixtures.InteractionAPITests.last_message

    async def test_commander_damage_matrix_and_partner_owners(self):
        data = bytes([2, 3]) + struct.pack('<6I', 0, 2, 9, 0, 21, 4)
        data += struct.pack('<BIIBII', 0, 200, 5000, 1, 201, 5001)
        game, _ = await self.make_game(prompt(), element(10, data))
        first,second = game.observation.commanders
        self.assertEqual((first.object_id,first.damage), (200,(0,9,21)))
        self.assertEqual((second.owner,second.texture,second.damage), (1,5001,(2,0,4)))
        self.assertFalse(game.observation.unknown_elements)
        with self.assertRaises(AttributeError):
            first.owner = 2

    async def test_dungeon_completion_counts_and_indices(self):
        data = bytes([2])+struct.pack('<2iI',500,501,4)+bytes([0,0,1,1,0,0,0,1])
        game, _ = await self.make_game(prompt(), element(12,data))
        first,second = game.observation.dungeons
        self.assertEqual((first.texture,dict(first.completions)),(500,{0:2,1:1}))
        self.assertEqual(dict(second.completions),{1:1})
        with self.assertRaises(TypeError):
            first.completions[0]=3

    async def test_auto_yield_state_and_explicit_removal(self):
        data = struct.pack('<iiB',1,701,3)+b'Ability\0Draw\0'
        game, backend = await self.make_game(prompt(), element(11,data))
        self.assertEqual(game.observation.auto_yields[0].description,'Ability')
        await game.remove_yield(701)
        self.assertEqual(self.last_message(game,backend),(4696,struct.pack('<ii',701,3)))
        with self.assertRaises(UnsupportedInteraction):
            await game.remove_yield(702)

    async def test_undo_invalidates_prompt_and_yield_toggles_do_not(self):
        game,backend = await self.make_game(prompt())
        obs = game.observation
        await game.undo(expected_revision=obs.revision)
        self.assertEqual(self.last_message(game,backend),(4644,b''))
        with self.assertRaises(StaleDecision):
            await game.submit(obs.decision_id,PromptChoice(1))
        backend.game_state(packet(prompt(timestamp=44)))
        obs=await game.wait_for_update(obs.revision,timeout=1)
        for control in YieldControl:
            await game.yield_control(control)
            self.assertEqual(self.last_message(game,backend),(4646,bytes([control])))
        await game.submit(obs.decision_id,PromptChoice(1))
        self.assertEqual(self.last_message(game,backend)[0],4638)

    async def test_controls_reject_stale_spectator_ended_and_invalid_keys(self):
        game,backend = await self.make_game(prompt())
        with self.assertRaises(StaleDecision):
            await game.undo(expected_revision=0)
        with self.assertRaises(ValueError):
            await game.yield_control(255)
        await game._set_player_order((100,101))
        with self.assertRaises(UnsupportedInteraction):
            await game.undo()
        await game._set_player_order((42,123))
        await game.concede()
        self.assertEqual(self.last_message(game,backend),(4637,b'\0'))
        await game._receive(backend.token.bytes_le,4632,b'')
        with self.assertRaises(StaleDecision):
            await game.undo()

    async def test_stop_preferences_use_server_indices_and_phase_minus_two(self):
        game,backend = await self.make_game(prompt())
        await game.update_stops(checksum='a'*32,stops=(PlayerStops(0,Step.UPKEEP|Step.END),),
                                yields=(PlayerStops(1,Step.COMBAT_DAMAGE),),preferences=YieldPreference.OWN_MADNESS)
        op,data = self.last_message(game,backend)
        c=Cursor(data)
        self.assertEqual(op,4640)
        self.assertEqual(c.string(False),'a'*32)
        self.assertEqual(struct.unpack('<iIIiIII',c.take(28)),(1,0,513,1,1,64,64))
        c.end()
        with self.assertRaises(ValueError):
            await game.update_stops(checksum='a'*32,stops=(PlayerStops(42,1),))
        with self.assertRaises(ValueError):
            stop_payload('',(),(),0)
        with self.assertRaises(ValueError):
            stop_payload('a'*32,(PlayerStops(0,1),PlayerStops(0,2)),(),0)

    async def test_staged_three_piles_validate_everything_before_send(self):
        raw=(struct.pack('<iiIII6B',1,3,43,4097,0,0,0,0,0,0,0)+b'\0'*128+b'Arrange\0')
        local={'ACTION_NAME_STRING':'Move to pile 1','ALTACTION_DONT_RECALCULATE':1}
        game,backend=await self.make_game(element(3,raw),thing(200,action=local),thing(201,action=local),thing(202,action=local))
        obs=game.observation
        self.assertTrue(obs.prompt['arranging_three_piles'])
        before=len(backend.frames)
        with self.assertRaises(UnsupportedInteraction):
            await game.submit_batch(obs.decision_id,(CardSelection(200,ACTION),CardSelection(999,ACTION),PromptChoice(1)))
        self.assertEqual(len(backend.frames),before)
        result=await game.submit_batch(obs.decision_id,tuple(CardSelection(i,ACTION) for i in (200,201,202))+(PromptChoice(1),))
        self.assertEqual(len(result),4)
        self.assertEqual(self.last_message(game,backend)[0],4638)
        with self.assertRaises(StaleDecision):
            await game.submit_batch(obs.decision_id,(CardSelection(200,ACTION),))

    async def test_batch_does_not_guess_states_between_ordinary_actions(self):
        game,backend=await self.make_game(prompt(),thing(200,action={'ACTION_NAME_STRING':'Cast'}))
        before=len(backend.frames)
        with self.assertRaises(ValueError):
            await game.submit_batch(game.observation.decision_id,(CardSelection(200,ACTION),PromptChoice(1)))
        self.assertEqual(len(backend.frames),before)

    async def test_batch_duplicate_local_action_is_rejected_before_send(self):
        game,backend=await self.make_game(prompt(),thing(200,action={'ALTACTION_DONT_RECALCULATE':1}))
        before=len(backend.frames)
        with self.assertRaises(StaleDecision):
            await game.submit_batch(game.observation.decision_id,(CardSelection(200,ACTION),CardSelection(200,ACTION)))
        self.assertEqual(len(backend.frames),before)

    async def test_batch_transport_failure_closes_session_without_retry(self):
        game,backend=await self.make_game(prompt(),thing(200,action={'ALTACTION_DONT_RECALCULATE':1}))
        before=len(backend.frames)
        def fail(frame):
            raise ConnectionError('Synthetic write failure')
        backend.write=fail
        with self.assertRaises(ConnectionError):
            await game.submit_batch(game.observation.decision_id,(CardSelection(200,ACTION),PromptChoice(1)))
        self.assertFalse(game.client.connected)
        self.assertEqual(len(backend.frames),before)

    async def test_advertised_non_game_object_control_requires_opt_in(self):
        game,backend=await self.make_game(prompt(),thing(200,action={'ACTION_NAME_STRING':'Always yield to this','ACTION_IS_NON_GAME_ACTION':1}))
        await game.submit(game.observation.decision_id,CardSelection(200,ACTION,allow_non_game=True))
        self.assertEqual(self.last_message(game,backend)[0],4645)
        await game.submit(game.observation.decision_id,PromptChoice(1))
        self.assertEqual(self.last_message(game,backend)[0],4638)

    async def test_deferred_divided_targets_can_wait_for_resolution(self):
        action=targeted(ALTACTION_DONT_CHOOSE_TARGETS_NOW=1,ACTION_DAMAGEAMOUNT=3)
        action['ACTION_TARGET_INFO_LIST0']['ACTION_DIVIDE_EFFECT_UP']=1
        game,backend=await self.make_game(prompt(),thing(200,action=action))
        await game.submit(game.observation.decision_id,CardSelection(200,ACTION,((),)))
        payload=self.last_message(game,backend)[1]
        self.assertEqual(payload[16],0)
        self.assertEqual(payload[18],3)

    async def test_changeling_and_lost_changeling_follow_server_flags(self):
        action=targeted(ALL_TARGETS_MUST_SHARE_CREATURE_TYPE=1)
        action['ACTION_TARGET_INFO_LIST0'].update(MIN_TARGETS=2,MAX_TARGETS=2)
        catalog=ChoiceCatalog(definitions={1000:{'creature_types':('Human',)},1001:{'creature_types':(), 'CHANGELING':1}})
        game,_=await self.make_game(prompt(),thing(123,action=action),thing(200,texture=1000),thing(201,texture=1001),catalog=catalog)
        obs=game.observation
        self.assertTrue(obs.objects[201].properties['changeling'])
        await game.submit(obs.decision_id,CardSelection(123,ACTION,((200,201),)))
        game,_=await self.make_game(prompt(),thing(123,action=action),thing(200,texture=1000),thing(201,texture=1001,REMOVE_ALL_TEXT=1),catalog=catalog)
        with self.assertRaises(ValueError):
            await game.submit(game.observation.decision_id,CardSelection(123,ACTION,((200,201),)))

    async def test_ranged_dynamic_creature_type_ids_replace_static(self):
        catalog=ChoiceCatalog(definitions={1000:{'creature_types':('Human',)}})
        elf=next(o for o in catalog.options(32768) if o.label.lower()=='elf')
        game,_=await self.make_game(prompt(),thing(200,CREATURE_TYPEID0=elf.value),catalog=catalog)
        self.assertEqual(game.observation.objects[200].properties['creature_types'],('Elf',))

    async def test_room_names_follow_unlocked_doors_and_different_name_constraint(self):
        catalog=ChoiceCatalog(definitions={1000:{'IS_ROOM':1, 'CARDNAME_STRING':'Left // Right',
            'LEFT_HALF_ATTRIBUTES_LIST':{'CARDNAME_STRING':'Left'},'RIGHT_HALF_ATTRIBUTES_LIST':{'CARDNAME_STRING':'Right'}}})
        action=targeted(TARGETS_MUST_HAVE_DIFFERENT_NAMES=1)
        action['ACTION_TARGET_INFO_LIST0'].update(MIN_TARGETS=2,MAX_TARGETS=2)
        game,_=await self.make_game(prompt(),thing(123,action=action),thing(200,LEFT_UNLOCKED=1),thing(201,LEFT_UNLOCKED=1,RIGHT_UNLOCKED=1),catalog=catalog)
        self.assertEqual(game.observation.objects[201].properties['card_names'],('Left','Right'))
        with self.assertRaises(ValueError):
            await game.submit(game.observation.decision_id,CardSelection(123,ACTION,((200,201),)))
        game,_=await self.make_game(prompt(),thing(123,action=action),thing(200,LEFT_UNLOCKED=1),thing(201,RIGHT_UNLOCKED=1),catalog=catalog)
        await game.submit(game.observation.decision_id,CardSelection(123,ACTION,((200,201),)))

    async def test_face_down_objects_are_nameless_for_distinct_name_choices(self):
        action=targeted(TARGETS_MUST_HAVE_DIFFERENT_NAMES=1)
        action['ACTION_TARGET_INFO_LIST0'].update(MIN_TARGETS=2,MAX_TARGETS=2)
        game,_=await self.make_game(prompt(),thing(123,action=action),thing(200,FACE_DOWN=1),thing(201,MANIFEST_FACE_DOWN=1))
        self.assertEqual(game.observation.objects[200].properties['card_names'],())
        with self.assertRaises(ValueError):
            await game.submit(game.observation.decision_id,CardSelection(123,ACTION,((200,201),)))

    async def test_mana_source_snow_and_restriction_are_visible(self):
        data=struct.pack('<4i',0,700,1,4)+attrs({'MANA_POOL_FROM_SOURCE_ID':200,'MANA_POOL_SNOW':1,
            'MANA_RIDER_SHORT_DESCRIPTION_STRING':'Only on creatures'})
        game,_=await self.make_game(prompt(),element(4,data))
        mana=game.observation.mana[0]
        self.assertEqual((mana.source_id,mana.snow,mana.restriction),(200,True,'Only on creatures'))

    async def test_full_player_status_yield_bits_and_active_player(self):
        data=struct.pack('<64h',*([20]*64))+bytes([1,0,1,0,1])+b'\0'*208
        game,_=await self.make_game(prompt(),element(2,data))
        self.assertEqual(game.observation.players[0]['active_player'],1)
        self.assertEqual(game.observation.players[0]['auto_stack'],1)
        self.assertFalse(game.observation.unknown_elements)

    async def test_wildcard_creature_subtype_is_universal(self):
        from patchwork_automaton.properties import PROPERTY_IDS
        catalog=ChoiceCatalog(definitions={1000:{'creature_types':('Elf',)}})
        game,_=await self.make_game(prompt(),thing(200,CREATURE_TYPEID0=PROPERTY_IDS['GROUPMATCH_WILDCARD']),catalog=catalog)
        self.assertEqual(game.observation.objects[200].properties['creature_types'],('*',))
        self.assertTrue(game.observation.objects[200].properties['all_creature_types'])

    async def test_unknown_dynamic_subtype_does_not_inherit_static_type(self):
        catalog=ChoiceCatalog(definitions={1000:{'creature_types':('Human',)}})
        game,_=await self.make_game(prompt(),thing(200,CREATURE_TYPEID0=99999999),catalog=catalog)
        self.assertNotIn('creature_types',game.observation.objects[200].properties)

    async def test_hidden_or_original_texture_does_not_replace_current_definition(self):
        catalog=ChoiceCatalog(definitions={1000:{'CARDNAME_STRING':'Current'},1001:{'CARDNAME_STRING':'Other'}})
        game,_=await self.make_game(prompt(),thing(200,MORPHED_CARD_TEXTURE_NUMBER=1001),catalog=catalog)
        self.assertEqual(game.observation.objects[200].properties['card_names'],('Current',))

    async def test_room_on_stack_has_only_chosen_half_name(self):
        catalog=ChoiceCatalog(definitions={1000:{'IS_ROOM':1,
            'LEFT_HALF_ATTRIBUTES_LIST':{'CARDNAME_STRING':'Left'},'RIGHT_HALF_ATTRIBUTES_LIST':{'CARDNAME_STRING':'Right'}}})
        game,_=await self.make_game(prompt(),thing(200,ZONE=1001728,CHOSE_RIGHT=1),catalog=catalog)
        self.assertEqual(game.observation.objects[200].properties['card_names'],('Right',))

    async def test_face_down_does_not_inherit_printed_subtypes(self):
        catalog=ChoiceCatalog(definitions={1000:{'creature_types':('Human',)}})
        game,_=await self.make_game(prompt(),thing(200,FACE_DOWN=1),catalog=catalog)
        self.assertEqual(game.observation.objects[200].properties['creature_types'],())

    async def test_wait_for_update_wakes_on_game_end_without_new_state(self):
        game,backend=await self.make_game(prompt())
        waiting=asyncio.create_task(game.wait_for_update(game.revision,timeout=1))
        await asyncio.sleep(0)
        raw=struct.pack('<IiBiBIBI',77,0,1,2,2,100,3,200)
        await game._receive(backend.token.bytes_le,4485,raw)
        with self.assertRaises(GameEnded) as error:
            await waiting
        self.assertIs(error.exception.result,game.result)
        self.assertEqual(error.exception.game_id,77)

    async def test_game_results_are_typed_and_check_route_identity(self):
        game,backend=await self.make_game(prompt())
        raw=struct.pack('<IiBiBIBI',77,1,3,2,2,100,3,200)
        await game._receive(backend.token.bytes_le,4485,raw)
        self.assertTrue(game.ended)
        self.assertEqual(game.result.winner_indices,(0,))
        self.assertEqual(game.result.players[1].playing_time,200)
        self.assertEqual(game.result.moved_first,1)
        game,_=await self.make_game(prompt())
        with self.assertRaises(ProtocolError):
            await game._receive(backend.token.bytes_le,4485,struct.pack('<IiBi',999,0,1,0))
        with self.assertRaises(ProtocolError):
            await game._receive(backend.token.bytes_le,4485,b'')

    async def test_special_step_metadata_is_preserved(self):
        raw=bytearray(prompt()[8:]);raw[13:15]=bytes([1,8])
        game,_=await self.make_game(element(9,raw))
        self.assertEqual((game.observation.prompt['special_player'],game.observation.prompt['special_type']),(1,8))


class MalformedStateTests(unittest.TestCase):
    def test_new_state_families_reject_truncated_or_invalid_arrays(self):
        from patchwork_automaton.state import State
        for kind,data in ((10,b'\1\2'),(10,bytes([1,1])+struct.pack('<IBII',0,3,1,1)),
                          (11,struct.pack('<i',-1)),(11,struct.pack('<iiB',1,4,1)+b'unterminated'),
                          (12,b'\0'+struct.pack('<I',1)+b'\0\0'),(12,b'\0'+struct.pack('<I',0xffffffff))):
            with self.subTest(kind=kind,data=data),self.assertRaises(ProtocolError):
                State().update(packet(element(kind,data)))
