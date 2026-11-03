"""Public Hartford distinctions survive encoding without running native games."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from patchwork_automaton.backends.rustmage import OBSERVATION_SCHEMA
from patchwork_automaton.core import Handle, Response
from patchwork_automaton.neural.rustmage_observation import (
    PLAYERS, _Object, _Observation, _Player,
)
from patchwork_automaton.neural.standard_features import StandardEncoder


def card(identifier, name, *, zone='Battlefield', **fields):
    return dict(id=identifier, reference=dict(card=identifier, generation=1),
                card=name, name=name, controller='P0', owner='P0', zone=zone,
                counters=dict(plus_one=0, minus_one=0, stun=0), **fields)


def observation(*, objects=(), decision='priority', **native):
    players = tuple(_Player(PLAYERS[f'P{i}'], f'P{i}', 20, 0, 50,
                           dict(life=20, hand_count=0, library_count=50))
                    for i in range(2))
    visible = {}
    for item in objects:
        identifier = Handle('rustmage', 'object', str(item['id']))
        visible[identifier] = _Object(identifier, item['name'], item['zone'].lower(),
                                      PLAYERS[item['controller']], item)
    raw = dict(schema=OBSERVATION_SCHEMA, decision=decision, step='Main1',
               turn=3, active='P0', objects=list(objects), stack=[],
               waiting_triggers=[], announcing=None, resolving=None, **native)
    return _Observation('rustmage', Handle('rustmage', 'game', 'review'),
                        PLAYERS['P0'], 1, players, visible,
                        SimpleNamespace(token='review', kind=decision), False, None,
                        raw, 3, 'main1', PLAYERS['P0'], PLAYERS['P0'])


def encode(obs, *actions):
    choices = tuple((Response(f'fixture:{index}'), action)
                    for index, action in enumerate(actions))
    return StandardEncoder().encode(obs, choices=choices).arrays


class HartfordSchema198ReviewTests(unittest.TestCase):
    def test_perilous_snare_speed_and_turn_use_are_visible_player_state(self):
        obs = observation()

        def with_speed(speed, triggered):
            player = obs.players[0]._replace(native=dict(
                obs.players[0].native, speed=speed,
                speed_triggered_this_turn=triggered))
            return obs._replace(players=(player, obs.players[1]))

        three = encode(with_speed(3, False), dict(kind='Pass'))
        four = encode(with_speed(4, False), dict(kind='Pass'))
        used = encode(with_speed(3, True), dict(kind='Pass'))
        self.assertFalse(np.array_equal(three['entities'], four['entities']))
        self.assertFalse(np.array_equal(three['entities'], used['entities']))

    def test_named_mainboard_counters_remain_public_model_state(self):
        # These counters occur in the actual Hartford mainboards. In particular,
        # Braided Net's remaining uses must not disappear from observations.
        for identity, counter in (
            ('Braided Net // Braided Quipu', 'Net'),
            ('Drake Hatcher', 'Incubation'),
            ('Airbender Ascension', 'Quest'),
            ('Earthbender Ascension', 'Quest'),
            ('Esper Origins // Summon: Esper Maduin', 'Finality'),
            ('Warmaker Gunship', 'Charge'),
        ):
            with self.subTest(identity=identity):
                before = card(1, identity)
                after = deepcopy(before)
                after['counters']['named'] = [[counter, 3]]
                zero = encode(observation(objects=[before]), dict(kind='Pass'))
                three = encode(observation(objects=[after]), dict(kind='Pass'))
                self.assertFalse(np.array_equal(zero['entities'], three['entities']))

    def test_named_counter_payments_distinguish_legal_action_parameters(self):
        obj = card(1, 'Braided Net // Braided Quipu')
        actions = [dict(kind='AnnounceActivate', card=1, ability=0,
                        counter_payment=dict(plus_one=0, minus_one=0, stun=0,
                                             named=[['Net', count]]))
                   for count in (1, 2)]
        arrays = encode(observation(objects=[obj]), *actions)
        self.assertFalse(np.array_equal(*arrays['candidates']))

    def test_private_legal_library_choices_have_distinct_visible_card_features(self):
        for context, field, kind in (
            ('search_choice', 'cards', 'ChooseSearch'),
            ('library_choice', 'remaining', 'OrderLibraryBottom'),
            ('library_selection', 'remaining', 'SelectLibrary'),
        ):
            with self.subTest(context=context):
                mountain = card(1, 'Mountain', zone='Library')
                opt = card(2, 'Opt', zone='Library')
                # A concealed entry must not leak, even if a malformed fixture
                # also lists its identifier among actions. An unselected pool
                # card must not become an entity either.
                concealed = card(3, 'Island', zone='Library', hideInfo=True)
                unselected = card(4, 'Sleight of Hand', zone='Library')
                obs = observation(**{context: {field: [mountain, opt, concealed, unselected]}})
                original = deepcopy(obs.native)
                arrays = encode(obs, *(dict(kind=kind, card=i) for i in (1, 2, 3)))
                self.assertEqual(len(arrays['entities']), 4)  # Two players, two legal visible cards.
                self.assertFalse(np.array_equal(arrays['candidates'][0], arrays['candidates'][1]))
                self.assertEqual(arrays['candidate_refs'][0].sum(), 1)
                self.assertEqual(arrays['candidate_refs'][1].sum(), 1)
                self.assertEqual(arrays['candidate_refs'][2].sum(), 0)
                self.assertTrue(np.all(arrays['card_ids'] == 0))
                self.assertEqual(obs.native, original)
                self.assertEqual(len(obs.objects), 0)

    def test_schema198_stack_source_normalization_preserves_visible_facts(self):
        obj = card(1, 'Opt', zone='Stack')
        flat = dict(id=4, card=1, source_card='Opt', targets=[], x=2,
                    cast=dict(kicker=False))
        nested = dict(flat, source=dict(object=dict(card=1, generation=1), card='Opt'))
        # Conflicting obsolete fields cannot override the authoritative source.
        nested.update(card=99, source_card='Mountain')
        before = observation(objects=[obj])._replace(native=dict(
            observation(objects=[obj]).native, announcing=flat, stack=[flat]))
        after = before._replace(native=dict(before.native, announcing=nested, stack=[nested]))
        first, second = (encode(obs, dict(kind='Pass')) for obs in (before, after))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)

    def test_source_less_rule_cannot_reveal_obsolete_source_identity(self):
        empty = dict(id=4, card=None, source_card=None, targets=[])
        obsolete = dict(empty, source=None, card=99, source_card='Mountain')
        obs = observation()
        first = encode(obs._replace(native=dict(obs.native, announcing=empty)), dict(kind='Pass'))
        second = encode(obs._replace(native=dict(obs.native, announcing=obsolete)), dict(kind='Pass'))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)

    def test_granted_ability_retains_both_source_and_grantor(self):
        objects = [card(1, 'Llanowar Elves'), card(2, 'Forest'), card(3, 'Mountain')]
        actions = [dict(kind='AnnouncePlaneswalkerGrant',
                        source=dict(card=1, generation=1),
                        grantor=dict(card=grantor, generation=1), ability=0)
                   for grantor in (2, 3)]
        arrays = encode(observation(objects=objects), *actions)
        self.assertFalse(np.array_equal(*arrays['candidates']))
        self.assertFalse(np.array_equal(*arrays['candidate_refs']))


if __name__ == '__main__':
    unittest.main()
