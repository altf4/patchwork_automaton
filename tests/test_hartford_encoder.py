"""Pinned action-schema coverage and native/Python numerical parity."""
import ctypes as C
import json
from pathlib import Path
import unittest

import numpy as np

from patchwork_automaton.backends.rustmage import RustMageMatch, OBSERVATION_SCHEMA
from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS, standard_deck
from patchwork_automaton.core import Response
from patchwork_automaton.neural.direct_standard import library, library_path, arrays, check, string
from patchwork_automaton.neural.rustmage_observation import learning_observation
from patchwork_automaton.neural.standard_features import StandardEncoder, FIELDS, CATEGORIES


def action_cases():
    reference = {'card': 0, 'generation': 0}
    values = dict(card=0, ability=0, index=0, amount=3, id=0, mode=0, cost=0,
                  clause=0, value={'BasicLandType': 'Forest'}, grant_timestamp=1,
                  type_mask=1, target={'Player': 'P1'}, player='P1', beneficiary='P1',
                  object=reference, source=reference, sacrifice=reference,
                  grantor=reference, permission=reference, equipment=reference,
                  exiled=reference, graveyard=False, blocker=0, attacker=1,
                  payment=[0] * 6, production=[0] * 6,
                  counter_payment=dict(plus_one=0, minus_one=0, stun=0, loyalty=0, lore=0,
                                       named=[['Incubation', 3], ['Quest', 2], ['Net', 1]]),
                  face='Normal', permanent_type='Creature', pay=True, accept=True,
                  keep=True, library=True, hand=True, intrinsic=True,
                  flash_surcharge=False, collect=True, put_counters=True,
                  pile='FaceDown', effect=0, prevention='ShieldCounter',
                  contribution='Generic', parity='Odd', route='ExileThree',
                  counter='PlusOne', creature_type='Elf', token='Forest',
                  defender={'Player': 'P1'}, destination='Graveyard')
    for kind, fields in FIELDS.items():
        action = dict(kind=kind, **{field: values[field] for field in fields})
        if kind == 'ChooseZoneReplacement':
            action['effect'] = 'Exile'
        if kind == 'ChooseX':
            action['value'] = 3
        if kind in ('AttackPlaneswalker', 'AttackBattle'):
            action['defender'] = reference
        if 'GraveyardBorrowed' in kind:
            action['graveyard'] = reference
        if kind == 'ChooseOutsideBasic':
            action['card'] = 'Forest'
        if kind == 'AnnounceSneak':
            action['permission'] = [reference, 'Creature']
        if kind == 'BeginFaceUp':
            action['cost'] = {'colored': [0] * 6, 'generic': 2}
        yield action
    for value in ({'Color': 'Blue'}, {'CreatureType': 'Elf'}, {'ManaValueParity': 'Even'},
                  {'CardName': 'Forest'}):
        yield dict(kind='ChooseEntryValue', value=value)
    yield dict(kind='ChooseZoneReplacement', effect={'ExileReflexive': {'source': reference, 'ability': 2}})
    yield dict(kind='ChooseSearchDestination', destination={'BattlefieldThenHand': {'battlefield': 2}})
    yield dict(kind='ChooseSearchDestination', destination='Top')
    yield dict(kind='ChooseHandDefender', defender={'Battle': reference})
    yield dict(kind='ChooseMobilizeDefender', defender={'Planeswalker': reference})
    yield dict(kind='AnnounceSneak', card=0)
    for creature_type in CATEGORIES['creature_type']:
        yield dict(kind='ChooseCreatureType', creature_type=creature_type)
    for counter in CATEGORIES['counter']:
        yield dict(kind='SelectTargetCounter', counter=counter)


class HartfordSchemaTests(unittest.TestCase):
    def test_public_actions_have_explicit_distinguishing_fields(self):
        self.assertEqual(len(FIELDS), 200)
        encoder = StandardEncoder()
        self.assertEqual(encoder.manifest()['observation_schema'], OBSERVATION_SCHEMA)
        for action in action_cases():
            with self.subTest(action=action):
                extras = np.asarray(encoder.action_extras(action))
                self.assertTrue(np.isfinite(extras).all())
                self.assertEqual(len(extras), len(encoder.action_extras({})))
        for field in ('pile', 'counter', 'creature_type', 'contribution'):
            first, second = CATEGORIES[field][:2]
            self.assertNotEqual(encoder.action_extras({field: first}),
                                encoder.action_extras({field: second}))


@unittest.skipUnless(library_path().is_file(), 'Requires the guarded schema-198 direct build')
class HartfordNativeParityTests(unittest.TestCase):
    def test_nonzero_named_counters_and_speed_match_reference_tensors(self):
        from patchwork_automaton.core import CardSpec, DeckSpec
        from patchwork_automaton.backends.rustmage import _plain, domain_action
        from patchwork_automaton.baselines.standard import StandardAggro
        lib = library()
        encoder = StandardEncoder()
        for land, card, field in (('Island', 'Braided Net // Braided Quipu', 'counters'),
                                   ('Plains', 'Perilous Snare', 'speed')):
            with self.subTest(field=field):
                decks = [DeckSpec((CardSpec(land, 56), CardSpec(card, 4))),
                         DeckSpec((CardSpec('Forest', 60),))]
                cards = [[c.name for c in d.cards for _ in range(c.quantity)] for d in decks]
                handle = check(lib, lib.ds_new(json.dumps(dict(decks=cards, seed=81,
                                  starting_player='P0', deck_rules='Constructed')).encode()))
                observed = False
                try:
                    with RustMageMatch(*decks, seed=81) as reference:
                        policy = StandardAggro('bw-lifegain')
                        for _ in range(1000):
                            views = [learning_observation(reference, f'P{p}') for p in (0, 1)]
                            if views[0].terminal:
                                break
                            player = next(p for p in (0, 1) if views[p].decision is not None)
                            obs = views[player]
                            choices = encoder.expand_options(obs.decision)
                            self.assertFalse(any(a['kind'] == 'AnnounceBack' for _, a in choices),
                                             'Transforming backs are unavailable as ordinary casts')
                            view = check(lib, lib.ds_observe(handle, player))
                            try:
                                count = C.c_size_t()
                                check(lib, lib.ds_prepare(view, False, C.byref(count)))
                                actual = arrays(lib, lib.ds_encode(view))
                                expected = encoder.encode(obs, choices=choices).arrays
                                for key in expected:
                                    np.testing.assert_array_equal(expected[key], actual[key], err_msg=key)
                                observed = (any(o.get('counters', {}).get('named') for o in obs.native['objects'])
                                            if field == 'counters' else any(p.get('speed') for p in obs.native['players']))
                                if observed:
                                    break
                                response = policy.choose(obs)
                                option = next(o for o in obs.decision.options if o.id == response.option)
                                action = (option.constraints['action'] if option.kind == 'finite'
                                          else domain_action(option, response.value))
                                selected = next(i for i, (_, a) in enumerate(choices) if _plain(a) == _plain(action))
                                check(lib, lib.ds_apply_choice(handle, view, selected))
                                raw = reference._raw(obs.actor.value)
                                updated = reference.native_game.act(raw, _plain(action))
                                reference._after_action(updated, obs.actor.value, raw['revision'])
                            finally:
                                lib.ds_view_free(view)
                finally:
                    lib.ds_free(handle)
                self.assertTrue(observed, field)

    def test_all_public_action_variants_match_reference_tensors(self):
        lib = library()
        lib.ds_encode_debug_action.argtypes = [C.c_void_p, C.c_char_p]
        lib.ds_encode_debug_action.restype = C.c_void_p
        decks = [standard_deck('mono-green-landfall', catalog='hartford')] * 2
        cards = [[c.name for c in deck.cards for _ in range(c.quantity)] for deck in decks]
        handle = check(lib, lib.ds_new(json.dumps(dict(decks=cards, seed=81,
                          starting_player='P0', deck_rules='Constructed')).encode()))
        view = check(lib, lib.ds_observe(handle, 0))
        try:
            with RustMageMatch(*decks, seed=81) as reference:
                observation = learning_observation(reference, 'P0')
                encoder = StandardEncoder()
                for action in action_cases():
                    with self.subTest(action=action):
                        expected = encoder.encode(observation, choices=((Response('debug'), action),)).arrays
                        actual = arrays(lib, lib.ds_encode_debug_action(view, json.dumps(action).encode()))
                        for key in expected:
                            np.testing.assert_array_equal(expected[key], actual[key], err_msg=key)
        finally:
            lib.ds_view_free(view)
            lib.ds_free(handle)

    def test_ten_decks_have_matching_real_private_views_choices_and_tensors(self):
        lib = library()
        encoder = StandardEncoder()
        ids = tuple(HARTFORD_DECKS)
        compared = 0
        for number, deck in enumerate(ids):
            decks = [standard_deck(d, catalog='hartford') for d in (deck, ids[(number + 1) % 10])]
            cards = [[c.name for c in d.cards for _ in range(c.quantity)] for d in decks]
            seed = 9261101000 + number
            handle = check(lib, lib.ds_new(json.dumps(dict(decks=cards, seed=seed,
                              starting_player='P0', deck_rules='Constructed')).encode()))
            rng = np.random.default_rng(seed)
            try:
                with RustMageMatch(*decks, seed=seed) as reference:
                    for step in range(300):
                        views = [learning_observation(reference, f'P{p}') for p in (0, 1)]
                        if views[0].terminal:
                            break
                        player = next(p for p in (0, 1) if views[p].decision is not None)
                        obs = views[player]
                        view = check(lib, lib.ds_observe(handle, player))
                        try:
                            self.assertEqual(json.loads(string(lib, lib.ds_debug_json(view))), obs.native)
                            choices = encoder.expand_options(obs.decision)
                            count = C.c_size_t()
                            check(lib, lib.ds_prepare(view, False, C.byref(count)))
                            self.assertEqual(count.value, len(choices))
                            actual = arrays(lib, lib.ds_encode(view))
                            expected = encoder.encode(obs, choices=choices).arrays
                            for key in expected:
                                np.testing.assert_array_equal(expected[key], actual[key],
                                    err_msg=f'{deck} step={step} {key}')
                            selected = int(rng.integers(len(choices)))
                            action = choices[selected][1]
                            from patchwork_automaton.backends.rustmage import _plain
                            self.assertEqual(json.loads(string(lib, lib.ds_choice_debug_json(view, selected))), _plain(action))
                            check(lib, lib.ds_apply_choice(handle, view, selected))
                            raw = reference._raw(obs.actor.value)
                            updated = reference.native_game.act(raw, _plain(action))
                            reference._after_action(updated, obs.actor.value, raw['revision'])
                            compared += 1
                        finally:
                            lib.ds_view_free(view)
            finally:
                lib.ds_free(handle)
        self.assertGreater(compared, 1000)


if __name__ == '__main__':
    unittest.main()
