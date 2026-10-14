"""Synthetic wire fixtures based on client 3.4.159.4704 serializers.

Names and IDs here are fictional; no private account captures are committed.
"""
import asyncio
from dataclasses import replace
import struct
import unittest

from patchwork_automaton import (CardSelection, ChoiceCatalog, DamageChoice,
    ManaChoice, MTGOClient, OrderingChoice, PromptChoice, PromptCode,
    SelectorChoice, SelectionSource, StaleDecision, UnsupportedInteraction)
from patchwork_automaton.actions import selected_groups
from patchwork_automaton.interactions import mode_spec, target_groups
from patchwork_automaton.properties import PROPERTY_IDS, property_id
from patchwork_automaton.protocol import Cursor, ProtocolError, array
from patchwork_automaton.state import checksum, interaction
from test_client import Backend


ACTION = PROPERTY_IDS['ACTION_LIST0']


def attrs(properties):
    data = b''
    for key, value in properties.items():
        code = property_id(key)
        if code < 0:
            raise ValueError(key)
        kind = code & 0x78000000
        if kind == 0x08000000:
            data += struct.pack('<I', code) + attrs(value)
        elif kind == 0x40000000:
            raw = value.encode('latin-1')
            data += struct.pack('<IH', code, len(raw)) + raw
        elif kind == 0x10000000:
            if value:
                data += struct.pack('<I', code)
        elif kind == 0x20000000:
            data += struct.pack('<Ii', code + 0x08000000, value)
        else:
            raise ValueError(key)
    return data + b'\0' * 4


def element(kind, payload):
    return struct.pack('<ii', len(payload) + 8, kind) + payload


def thing(number, *, action=None, texture=1000, **properties):
    props = dict(THINGNUMBER=number, CONTROLLER=0, CARDTEXTURE_NUMBER=texture, ZONE=1000961)
    props.update(properties)
    if action is not None:
        props['ACTION_LIST0'] = action
    return element(5, struct.pack('<i', 0) + attrs(props))


def ui(kind, payload=b''):
    return struct.pack('<iB', len(payload) + 5, kind) + payload


def prompt(*entries, timestamp=43):
    button = ui(1, struct.pack('<iBB', 1, 0, 1) + b'OK\0')
    body = struct.pack('<iiI3B', 2, 3, timestamp, 0, 0, 0) + b'Choose\0'
    return element(9, body + bytes([len(entries) + 1]) + button + b''.join(entries))


def packet(*parts, timestamp=43):
    raw = b''.join(parts)
    return array(raw) + b'\0' * 36 + struct.pack('<4I3IBBI', 77, len(raw), len(raw), len(parts),
                                               0, checksum(raw), 0, 0, 6, timestamp)


def target_info(legal=(200, 201), minimum=1, maximum=1, **extra):
    return dict(MIN_TARGETS=minimum, MAX_TARGETS=maximum, TARGET_NAME_STRING='Target', **extra)


def legal_targets(ids):
    return {f'attribute_{604983296+i}' if i else 'ACTION_TARGETID0': value for i, value in enumerate(ids)}


def targeted(**extra):
    return dict(ACTION_NAME_STRING='Synthetic action', ACTION_TARGET_INFO_LIST0=target_info(),
                ACTION_LEGAL_TARGET_LIST0=legal_targets((200, 201)), **extra)


class SelectorDecoderTests(unittest.TestCase):
    def test_all_native_descriptor_types(self):
        parts = [ui(3, bytes([1, 9, 3])), ui(4), ui(5, struct.pack('<iBBii', 32768, 0, 2, 1, 2)),
                 ui(6, struct.pack('<I', 128 | 256)), ui(7, struct.pack('<BiBiB', 2, 500, 1, 501, 0)), ui(8)]
        raw = prompt(*parts)[8:]
        parsed = interaction(raw)
        self.assertEqual([s['kind'] for s in parsed['selectors']], ['number','player','list','color','card','wish'])
        self.assertEqual(parsed['selectors'][0]['maximum'], 9)
        self.assertEqual(parsed['selectors'][2]['except_values'], [1, 2])
        self.assertEqual(parsed['selectors'][4]['options'], [(500, 'Card texture 500')])

    def test_bad_known_descriptor_is_rejected(self):
        with self.assertRaises(ProtocolError):
            interaction(prompt(ui(3, b'\1'))[8:])
        with self.assertRaises(ProtocolError):
            interaction(prompt(ui(7, b'\2'))[8:])

    def test_sparse_indexed_target_groups_and_card_fallback(self):
        action = {'ACTION_TARGET_INFO_LIST0':target_info(), 'attribute_202482434':{'TARGET_NAME_STRING':'Second'}}
        card = {'ACTION_LEGAL_TARGET_LIST0':legal_targets((200,)), 'attribute_202553858':legal_targets((201,))}
        groups = target_groups(action, card)
        self.assertEqual([g.index for g in groups], [0, 2])
        self.assertEqual([g.legal_targets for g in groups], [(200,), (201,)])
        self.assertEqual((groups[1].minimum, groups[1].maximum), (1, 1))

    def test_native_target_minimum_and_distribution_sentinel(self):
        action = targeted(ACTION_DAMAGEAMOUNT=-1)
        action['ACTION_TARGET_INFO_LIST0'].update(MIN_TARGETS=3, MAX_TARGETS=2, ACTION_DIVIDE_EFFECT_UP=1)
        group, = target_groups(action, {})
        self.assertEqual((group.minimum, group.maximum), (2, 2))
        self.assertIsNone(group.distributed_total)

    def test_legacy_damage_confirmation(self):
        raw = (struct.pack('<iiIII6B', 1, 3, 43, int(PromptCode.DAMAGE_OK), 0, 0, 0, 0, 0, 0, 0)
               + b'\0' * 128 + b'Assign damage\0')
        self.assertEqual(interaction(raw, legacy=True)['buttons'],
                         [dict(code=int(PromptCode.DAMAGE_OK), label='OK', enabled=True)])

    def test_static_catalog_token_and_texture_namespaces(self):
        catalog = ChoiceCatalog(definitions={1000:{'CARDNAME_TOKEN':5000,'CARDNAME_STRING':'Sample','IS_CREATURE':1},
                                            1001:{'CARDNAME_TOKEN':5000,'CARDNAME_STRING':'Sample','IS_CREATURE':1},
                                            1002:{'CARDNAME_TOKEN':5002,'CARDNAME_STRING':'Other','IS_LAND':1}},
                                allowed_name_tokens=(5000,))
        self.assertEqual([o.value for o in catalog.options(SelectionSource.CARD_NAME)], [5000])
        self.assertEqual([o.value for o in catalog.wish_options()], [1000, 1001])
        self.assertTrue(catalog.options(SelectionSource.COUNTER_TYPE))
        self.assertTrue(catalog.options(SelectionSource.CREATURE_TYPE))
        self.assertEqual(catalog.options(3 | 16)[0].value, PROPERTY_IDS['IS_CREATURE'])

    def test_fixed_mode_combination_is_preserved(self):
        props = {'ACTION_STRING0':'First', 'attribute_1142185473':'Second',
                 'ACTION_MODAL_MAX_CHOICES':2, 'ACTION_MODE_AVAILABLE0':1,'attribute_605465857':1}
        # Exact string-property key name is intentionally resolved numerically.
        props['attribute_1142185472'] = props.pop('ACTION_STRING0')
        spec = mode_spec(props, {'ACTION_GROUP_HEADER_STRING':'Cast'})
        self.assertEqual(spec.fixed_mapping, '01')


class InteractionAPITests(unittest.IsolatedAsyncioTestCase):
    async def make_game(self, *parts, catalog=None):
        backend = Backend()
        client = MTGOClient(connector=backend.connect, timeout=1, choice_catalog=catalog)
        self.addAsyncCleanup(client.close)
        await client.login('synthetic', 'synthetic')
        game = await client.connect_game((await client.current_matches())[0])
        await game._set_player_order((42, 123))
        backend.game_state(packet(*parts))
        await game.wait_for_update(game.revision, timeout=1)
        return game, backend

    def last_message(self, game, backend):
        c = Cursor(backend.frames[-1][1])
        c.take(24)
        op, data = game.client.catalog.parse(c.blob())
        c.end()
        return op, data

    async def test_targeted_action_and_stack_counterspell(self):
        game, backend = await self.make_game(prompt(), thing(123, action=targeted()),
                                             thing(200, ZONE=1001728), thing(201))
        obs = game.observation
        self.assertEqual([obj.id for obj in obs.stack], [200])
        self.assertEqual(obs.actions[0].target_groups[0].legal_targets, (200, 201))
        await game.submit(obs.decision_id, CardSelection(123, ACTION, ((200,),)))
        opcode, raw = self.last_message(game, backend)
        self.assertEqual(opcode, 4645)
        self.assertEqual(struct.unpack('<iiiIBBBB', raw[:20]), (123, ACTION, 0, 43, 1, 0, 1, 0))
        self.assertEqual(raw[20:], struct.pack('<iI', 1, 200))

    async def test_target_constraints_are_checked_before_sending(self):
        game, backend = await self.make_game(prompt(), thing(123, action=targeted()), thing(200), thing(201))
        count = len(backend.frames)
        for selection in ((), (999,), (200, 201), (200, 200)):
            with self.assertRaises(ValueError):
                await game.submit(game.observation.decision_id, CardSelection(123, ACTION, (selection,)))
        self.assertEqual(len(backend.frames), count)

    async def test_divided_effect_target_count_excludes_distribution_values(self):
        action = targeted(ACTION_DAMAGEAMOUNT=3)
        action['ACTION_TARGET_INFO_LIST0'].update(MAX_TARGETS=2, ACTION_DIVIDE_EFFECT_UP=1)
        action['attribute_202482433'] = target_info()
        action['attribute_202553857'] = legal_targets((201,))
        game, backend = await self.make_game(prompt(), thing(123, action=action), thing(200), thing(201))
        await game.submit(game.observation.decision_id, CardSelection(123, ACTION, ((200,), (201,)),
                                                                       distributions=((3,), ())))
        _, raw = self.last_message(game, backend)
        self.assertEqual(struct.unpack('<IBBBB', raw[12:20]), (43, 3, 0, 3, 0))
        self.assertEqual(struct.unpack('<i6I', raw[20:]), (6, 200, 0xfffffffe, 201, 3, 0xfffffffe, 0))

    async def test_modal_selection_filters_and_repeats_target_groups(self):
        action = dict(ACTION_NAME_STRING='Modal', ACTION_MODAL_MAX_CHOICES=2, ACTION_MODAL_MIN_CHOICES=2,
                      ACTION_MODAL_REPEAT_CHOICES=1, ACTION_MODE_AVAILABLE0=1,
                      attribute_605465857=1, attribute_1142185472='First', attribute_1142185473='Second',
                      ACTION_MODAL_EFFECT_LIST0={'ACTION_TARGET_INFO_LIST0':target_info(),
                                                'ACTION_LEGAL_TARGET_LIST0':legal_targets((200,))},
                      attribute_202812161={'ACTION_TARGET_INFO_LIST0':target_info(),
                                          'ACTION_LEGAL_TARGET_LIST0':legal_targets((201,))})
        game, backend = await self.make_game(prompt(), thing(123, action=action), thing(200), thing(201))
        selected = CardSelection(123, ACTION, ((200,), (200,)), modes=(0, 0))
        await game.submit(game.observation.decision_id, selected)
        _, raw = self.last_message(game, backend)
        self.assertEqual(raw[8:14], struct.pack('<i', 2) + b'00')
        self.assertEqual(struct.unpack('<i3I', raw[22:]), (3, 200, 0xfffffffe, 200))

    async def test_modal_invalid_count_and_unavailable_modes(self):
        action = dict(ACTION_NAME_STRING='Modal', ACTION_MODAL_MAX_CHOICES=1, ACTION_MODAL_MIN_CHOICES=1,
                      ACTION_MODE_AVAILABLE0=1, attribute_1142185472='First')
        game, _ = await self.make_game(prompt(), thing(123, action=action))
        for modes in ((), (1,), (0, 0)):
            with self.assertRaises(ValueError):
                await game.submit(game.observation.decision_id, CardSelection(123, ACTION, modes=modes))

    async def test_weighted_modes_and_retain_priority(self):
        action = dict(ACTION_NAME_STRING='Weighted', ACTION_MODAL_MAX_CHOICES=2, ACTION_MODAL_MIN_CHOICES=2,
                      ACTION_MODAL_REPEAT_CHOICES=1, ACTION_MODE_AVAILABLE0=1,
                      ACTION_MODAL_CHOICE_WEIGHT0=2, attribute_1142185472='First')
        game, backend = await self.make_game(prompt(), thing(123, action=action))
        await game.submit(game.observation.decision_id, CardSelection(123, ACTION, modes=(0,), retain_priority=True))
        _, raw = self.last_message(game, backend)
        self.assertEqual(raw[8:13], struct.pack('<i', 1) + b'0')
        self.assertEqual(raw[19], 0)

    async def test_counter_observation_and_counter_type_selection(self):
        game, backend = await self.make_game(prompt(ui(5, struct.pack('<iBB', 32, 0, 0))),
                                             thing(200, COUNTERS_LIST={'PLUS_ONE_PLUS_ONE_COUNTERS':3}))
        obs = game.observation
        self.assertEqual(obs.objects[200].counters['PLUS_ONE_PLUS_ONE_COUNTERS'], 3)
        with self.assertRaises(TypeError):
            obs.objects[200].counters['PLUS_ONE_PLUS_ONE_COUNTERS'] = 4
        choice = PROPERTY_IDS['PLUS_ONE_PLUS_ONE_COUNTERS']
        self.assertIn(choice, [o.value for o in obs.selectors[0].options])
        await game.submit(obs.decision_id, SelectorChoice(obs.selectors[0].id, choice))
        self.assertEqual(self.last_message(game, backend), (4638, struct.pack('<III', 2097152, choice, 43)))

    async def test_numeric_cost_choice_bounds(self):
        game, backend = await self.make_game(prompt(ui(3, b'\1\5\2')))
        selector = game.observation.selectors[0]
        for value in (0, 6):
            with self.assertRaises(ValueError):
                await game.submit(game.observation.decision_id, SelectorChoice(selector.id, value))
        await game.submit(game.observation.decision_id, SelectorChoice(selector.id, 4))
        self.assertEqual(self.last_message(game, backend), (4638, struct.pack('<III', 16384, 4, 43)))

    async def test_player_and_color_selectors(self):
        for entry, value, expected in ((ui(4), 1, (524288, 1, 43)),
                                       (ui(6, struct.pack('<I', 128)), 128, (128, 0, 0))):
            game, backend = await self.make_game(prompt(entry))
            selector = game.observation.selectors[0]
            await game.submit(game.observation.decision_id, SelectorChoice(selector.id, value))
            self.assertEqual(self.last_message(game, backend), (4638, struct.pack('<III', *expected)))

    async def test_name_token_filters_and_inversion(self):
        catalog = ChoiceCatalog(definitions={1000:{'CARDNAME_TOKEN':6000,'CARDNAME_STRING':'A'},
                                            1001:{'CARDNAME_TOKEN':6001,'CARDNAME_STRING':'B'}})
        for invert, expected in ((0, 6001), (1, 6000)):
            game, backend = await self.make_game(prompt(ui(5, struct.pack('<iBBi', 2097152, invert, 1, 6000))), catalog=catalog)
            selector = game.observation.selectors[0]
            self.assertEqual([o.value for o in selector.options], [expected])
            await game.submit(game.observation.decision_id, SelectorChoice(selector.id, expected))
            self.assertEqual(self.last_message(game, backend)[1], struct.pack('<III', 2097152, expected, 43))

    async def test_card_name_and_wish_use_texture_numbers(self):
        catalog = ChoiceCatalog(definitions={1000:{'CARDNAME_TOKEN':6000,'CARDNAME_STRING':'A'}})
        for entry, expected_code in ((ui(7, struct.pack('<BiB', 1, 1000, 1)), 2097152), (ui(8), 16777216)):
            game, backend = await self.make_game(prompt(entry), catalog=catalog)
            selector = game.observation.selectors[0]
            await game.submit(game.observation.decision_id, SelectorChoice(selector.id, 1000))
            self.assertEqual(self.last_message(game, backend)[1], struct.pack('<III', expected_code, 1000, 43))

    async def test_disabled_card_selector_is_not_selectable(self):
        game, _ = await self.make_game(prompt(ui(7, struct.pack('<BiB', 1, 1000, 0))))
        with self.assertRaises(ValueError):
            await game.submit(game.observation.decision_id, SelectorChoice(game.observation.selectors[0].id, 1000))

    async def test_mana_pool_restrictions_and_zero_timestamp(self):
        mana = struct.pack('<4i', 0, 72, 1, 4) + attrs({'MANA_POOL_FROM_SOURCE_ID':200, 'MANA_POOL_SNOW':1})
        game, backend = await self.make_game(prompt(ui(2, struct.pack('<BI', 0, 128))), element(4, mana))
        obs = game.observation
        self.assertEqual(obs.mana[0].color_flag, PromptCode.RED)
        self.assertEqual(obs.mana[0].attributes['MANA_POOL_SNOW'], 1)
        await game.submit(obs.decision_id, ManaChoice(PromptCode.RED, 72))
        self.assertEqual(self.last_message(game, backend), (4638, struct.pack('<III', 128, 72, 0)))
        with self.assertRaises(StaleDecision):
            await game.submit(obs.decision_id, ManaChoice(PromptCode.RED, 72))

    async def test_mana_not_offered_and_wrong_bucket_are_rejected(self):
        game, _ = await self.make_game(prompt(ui(2, struct.pack('<BI', 0, 128))))
        for choice in (ManaChoice(PromptCode.BLUE), ManaChoice(PromptCode.RED, 99)):
            with self.assertRaises((ValueError, UnsupportedInteraction)):
                await game.submit(game.observation.decision_id, choice)

    async def test_combat_damage_and_independent_assigners_then_confirm(self):
        parts = [element(7, struct.pack('<4i4i', source, 0, 3, 1, 200, 3, 0, 3)) for source in (123, 124)]
        game, backend = await self.make_game(prompt(), *parts)
        obs = game.observation
        await game.submit(obs.decision_id, DamageChoice(123, ((200, 3),)))
        self.assertEqual(self.last_message(game, backend), (4648, struct.pack('<iHHiII', 123, 1, 1, 2, 200, 3)))
        await game.submit(obs.decision_id, DamageChoice(124, ((200, 3),)))
        with self.assertRaises(StaleDecision):
            await game.submit(obs.decision_id, DamageChoice(123, ((200, 3),)))
        await game.submit(obs.decision_id, PromptChoice(1))

    async def test_damage_bounds_and_total(self):
        game, _ = await self.make_game(prompt(), element(7, struct.pack('<4i8i', 123, 0, 3, 2,
                                             200, 1, 1, 2, 201, 2, 1, 2)))
        for assignments in (((200, 3), (201, 0)), ((200, 1), (201, 1)), ((200, 3),), ((200, 1), (200, 2))):
            with self.assertRaises(ValueError):
                await game.submit(game.observation.decision_id, DamageChoice(123, assignments))

    async def test_combat_ordering_permutation_and_rank_encoding(self):
        game, backend = await self.make_game(prompt(), element(8, struct.pack('<4i', 123, 2, 200, 201)))
        obs = game.observation
        with self.assertRaises(ValueError):
            await game.submit(obs.decision_id, OrderingChoice(123, (200, 200)))
        await game.submit(obs.decision_id, OrderingChoice(123, (201, 200)))
        self.assertEqual(self.last_message(game, backend), (4648, struct.pack('<iHHi4I', 123, 2, 0, 4, 201, 1, 200, 2)))

    async def test_sacrifice_exile_discard_use_advertised_target_groups(self):
        for label, zone in (('Sacrifice',1000961), ('Exile',1000192), ('Discard',1000448)):
            action = targeted(ACTION_NAME_OVERRIDE_STRING=label)
            game, backend = await self.make_game(prompt(), thing(123, action=action), thing(200, ZONE=zone))
            await game.submit(game.observation.decision_id, CardSelection(123, ACTION, ((200,),)))
            self.assertEqual(self.last_message(game, backend)[0], 4645)

    async def test_attacker_and_blocker_choices_use_card_action_ids(self):
        game, backend = await self.make_game(prompt(), thing(123, action={'ACTION_NAME_STRING':'Attack', 'ATTACK_VICTIM_ID':1}))
        self.assertEqual(game.observation.actions[0].attack_victim_id, 1)
        await game.submit(game.observation.decision_id, CardSelection(123, ACTION))
        self.assertEqual(self.last_message(game, backend)[0], 4645)
        game, backend = await self.make_game(prompt(), thing(123, action=targeted(ACTION_NAME_OVERRIDE_STRING='Block')), thing(200))
        await game.submit(game.observation.decision_id, CardSelection(123, ACTION, ((200,),)))
        self.assertEqual(self.last_message(game, backend)[1][-4:], struct.pack('<I', 200))

    async def test_target_controller_and_sum_constraints(self):
        for constraint in ('TARGETS_MUST_HAVE_DIFFERENT_CONTROLLERS', 'TARGETTING_UPDATE_SUM_VALUES'):
            action = targeted(**{constraint:1 if 'CONTROLLER' in constraint else PROPERTY_IDS['POWER']}, TARGETTING_MAX_SUM_VALUE=3)
            action['ACTION_TARGET_INFO_LIST0'].update(MIN_TARGETS=2, MAX_TARGETS=2)
            game, _ = await self.make_game(prompt(), thing(123, action=action), thing(200, POWER=2), thing(201, POWER=2))
            with self.assertRaises(ValueError):
                await game.submit(game.observation.decision_id, CardSelection(123, ACTION, ((200,201),)))

    async def test_previous_targets_and_optional_deferred_targets(self):
        action = targeted(ACTION_TARGET_MUST_NOT_BE_PREVIOUS_TARGETS=1)
        action['attribute_202482433'] = target_info()
        action['attribute_202553857'] = legal_targets((200, 201))
        game, _ = await self.make_game(prompt(), thing(123, action=action), thing(200), thing(201))
        with self.assertRaises(ValueError):
            await game.submit(game.observation.decision_id, CardSelection(123, ACTION, ((200,), (200,))))
        action = targeted(ALTACTION_DONT_CHOOSE_TARGETS_NOW=1)
        game, backend = await self.make_game(prompt(), thing(123, action=action))
        await game.submit(game.observation.decision_id, CardSelection(123, ACTION, ((),)))
        self.assertEqual(self.last_message(game, backend)[1][16], 0)

    async def test_non_game_menu_entries_cannot_be_sent(self):
        game, _ = await self.make_game(prompt(), thing(123, action={'ACTION_NAME_STRING':'Inspect', 'ACTION_IS_NON_GAME_ACTION':1}))
        with self.assertRaises(UnsupportedInteraction):
            await game.submit(game.observation.decision_id, CardSelection(123, ACTION))

    async def test_unknown_selector_is_preserved_and_not_guessed(self):
        game, _ = await self.make_game(prompt(ui(99, b'xyz')))
        self.assertEqual(game.observation.selectors[0].raw, b'xyz')
        with self.assertRaises(UnsupportedInteraction):
            await game.submit(game.observation.decision_id, SelectorChoice(game.observation.selectors[0].id, 1))

    async def test_extra_prompt_wire_family(self):
        game, backend = await self.make_game(prompt())
        await game.submit(game.observation.decision_id, PromptChoice(1, extra=True))
        self.assertEqual(self.last_message(game, backend), (4650, struct.pack('<III', 1, 0, 43)))

    async def test_same_controller_toughness_and_different_value_constraints(self):
        cases = [('TARGETS_MUST_HAVE_SAME_CONTROLLER', {'CONTROLLER':0}, {'CONTROLLER':1}),
                 ('TARGETS_MUST_HAVE_EQUAL_TOUGHNESS', {'TOUGHNESS':2}, {'TOUGHNESS':3}),
                 ('TARGETTING_UPDATE_DIFFERING_VALUES', {'POWER':2}, {'POWER':2})]
        for requirement, first, second in cases:
            action = targeted(**{requirement:PROPERTY_IDS['POWER'] if 'VALUES' in requirement else 1})
            action['ACTION_TARGET_INFO_LIST0'].update(MIN_TARGETS=2, MAX_TARGETS=2)
            game, _ = await self.make_game(prompt(), thing(123, action=action), thing(200, **first), thing(201, **second))
            with self.assertRaises(ValueError):
                await game.submit(game.observation.decision_id, CardSelection(123, ACTION, ((200,201),)))

    async def test_catalog_type_and_name_constraints(self):
        for requirement, definitions in (
            ('TARGETS_MUST_HAVE_DIFFERENT_NAMES', {2000:{'CARDNAME_STRING':'Same'},2001:{'CARDNAME_STRING':'Same'}}),
            ('ALL_TARGETS_MUST_SHARE_CREATURE_TYPE', {2000:{'creature_types':['Human']},2001:{'creature_types':['Elf']}}),
            ('ACTION_TARGETS_MUST_SHARE_CARD_TYPE', {2000:{'card_types':['Creature']},2001:{'card_types':['Land']}}),
            ('ACTION_TARGETS_MUST_SHARE_PERMANENT_TYPE', {2000:{'permanent_types':['Creature']},2001:{'permanent_types':['Land']}})):
            action = targeted(**{requirement:1})
            action['ACTION_TARGET_INFO_LIST0'].update(MIN_TARGETS=2, MAX_TARGETS=2)
            game, _ = await self.make_game(prompt(), thing(123, action=action), thing(200, texture=2000),
                                           thing(201, texture=2001), catalog=ChoiceCatalog(definitions=definitions))
            with self.assertRaises(ValueError):
                await game.submit(game.observation.decision_id, CardSelection(123, ACTION, ((200,201),)))

    async def test_missing_card_metadata_does_not_guess_shared_types(self):
        action = targeted(ALL_TARGETS_MUST_SHARE_CREATURE_TYPE=1)
        action['ACTION_TARGET_INFO_LIST0'].update(MIN_TARGETS=2, MAX_TARGETS=2)
        game, _ = await self.make_game(prompt(), thing(123, action=action), thing(200), thing(201))
        with self.assertRaises(UnsupportedInteraction):
            await game.submit(game.observation.decision_id, CardSelection(123, ACTION, ((200,201),)))
        with self.assertRaises(UnsupportedInteraction):
            game.observation.actions[0].legal_targets(game.observation, 0, chosen=((200,),))

    async def test_replacement_or_trigger_cross_group_value_constraint(self):
        action = targeted(TARGETING_UPDATE_SAME_VALUES=PROPERTY_IDS['POWER'], TARGETING_UPDATE_TARGETLIST_BITMAP=1)
        action['attribute_202482433'] = target_info()
        action['attribute_202553857'] = legal_targets((201,))
        game, _ = await self.make_game(prompt(), thing(123, action=action), thing(200, POWER=2), thing(201, POWER=3))
        with self.assertRaises(ValueError):
            await game.submit(game.observation.decision_id, CardSelection(123, ACTION, ((200,), (201,))))

    async def test_partial_choices_are_invalidated_by_same_timestamp_update(self):
        raw = element(8, struct.pack('<4i', 123, 2, 200, 201))
        game, backend = await self.make_game(prompt(), raw)
        old = game.observation
        await game.submit(old.decision_id, OrderingChoice(123, (201, 200)))
        backend.game_state(packet(prompt(), raw))
        new = await game.wait_for_update(old.revision, timeout=1)
        self.assertEqual(old.decision_id.timestamp, new.decision_id.timestamp)
        with self.assertRaises(StaleDecision):
            await game.submit(old.decision_id, OrderingChoice(123, (201, 200)))
        await game.submit(new.decision_id, OrderingChoice(123, (200, 201)))

    async def test_cannot_answer_an_opponents_prompt(self):
        game, _ = await self.make_game(prompt())
        await game._set_player_order((123,42))
        with self.assertRaises(UnsupportedInteraction):
            await game.submit(game.observation.decision_id, PromptChoice(1))

    async def test_legacy_number_selector_and_pile_choices(self):
        raw = (struct.pack('<iiIII6B',1,3,43,16384,0,1,5,2,0,0,0)
               + b'\0'*128 + b'Choose number\0')
        game, backend = await self.make_game(element(3, raw))
        self.assertEqual(game.observation.selectors[0].kind, 'number')
        await game.submit(game.observation.decision_id, SelectorChoice(0, 3))
        self.assertEqual(self.last_message(game, backend)[1], struct.pack('<III', 16384, 3, 43))
        raw = (struct.pack('<iiIII6B',1,3,43,int(PromptCode.PILE),0,0,0,0,0,0,0)
               + b'\0'*128 + b'Choose pile\0')
        game, backend = await self.make_game(element(3, raw))
        self.assertEqual(game.observation.selectors[0].kind, 'pile')
        await game.submit(game.observation.decision_id, SelectorChoice(0, 2))
        self.assertEqual(self.last_message(game, backend)[1], struct.pack('<III', int(PromptCode.PILE), 2, 43))

    async def test_named_sources_use_types_and_counter_ids(self):
        for source in (4096,8192,32768,8388608):
            game, backend = await self.make_game(prompt(ui(5, struct.pack('<iBB', source, 0, 0))))
            selector = game.observation.selectors[0]
            self.assertTrue(selector.options)
            value = selector.options[0].value
            await game.submit(game.observation.decision_id, SelectorChoice(selector.id, value))
            self.assertEqual(self.last_message(game, backend)[1], struct.pack('<III', 2097152, value, 43))

    async def test_split_and_modal_faces_preserve_child_object_ids(self):
        game, backend = await self.make_game(prompt(), thing(123, GS_SPLITCARD_ID0=200),
                                             thing(200, action={'ACTION_NAME_STRING':'Cast face'}))
        actions = game.observation.actions_for(123)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].card_id, 200)
        await game.submit(game.observation.decision_id, CardSelection(200, ACTION))
        self.assertEqual(struct.unpack_from('<i', self.last_message(game, backend)[1])[0], 200)

    async def test_server_type_changes_override_local_static_definition(self):
        catalog = ChoiceCatalog(definitions={2000:{'card_types':['Creature'], 'permanent_types':['Creature']}})
        game, _ = await self.make_game(prompt(), thing(200, texture=2000, IS_LAND=1), catalog=catalog)
        self.assertEqual(game.observation.objects[200].properties['card_types'], ('Land',))

    async def test_available_targets_follow_partial_selection_constraints(self):
        action = targeted(TARGETS_MUST_HAVE_DIFFERENT_CONTROLLERS=1)
        action['ACTION_TARGET_INFO_LIST0'].update(MIN_TARGETS=1, MAX_TARGETS=2)
        action['ACTION_LEGAL_TARGET_LIST0'] = legal_targets((200, 201, 202))
        game, _ = await self.make_game(prompt(), thing(123, action=action), thing(200), thing(201), thing(202, CONTROLLER=1))
        obs = game.observation
        self.assertEqual(obs.actions[0].legal_targets(obs, 0), (200, 201, 202))
        self.assertEqual(obs.actions[0].legal_targets(obs, 0, chosen=((200,),)), (202,))

    async def test_multiple_locally_applied_attackers_before_confirmation(self):
        action = {'ACTION_NAME_STRING':'Attack', 'ATTACK_VICTIM_ID':1, 'ALTACTION_DONT_RECALCULATE':1}
        game, backend = await self.make_game(prompt(), thing(123, action=action), thing(124, action=action))
        obs = game.observation
        self.assertTrue(all(a.can_perform_locally for a in obs.actions))
        await game.submit(obs.decision_id, CardSelection(123, ACTION))
        await game.submit(obs.decision_id, CardSelection(124, ACTION))
        with self.assertRaises(StaleDecision):
            await game.submit(obs.decision_id, CardSelection(123, ACTION))
        await game.submit(obs.decision_id, PromptChoice(1))
        self.assertEqual(self.last_message(game, backend)[0], 4638)
