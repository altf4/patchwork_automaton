"""Matched A/B identities, continuation windows, and exact opponent quotas."""

from collections import Counter, defaultdict
import random
import unittest

from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS, STANDARD_DECKS
from patchwork_automaton.rl.hartford_ab import (
    EXTERNAL_BASE, HISTORICAL_VERSIONS, PARENT_ROUND, SUPPORTED_SCRIPT_DECKS,
    external_versions, training_schedule)
from patchwork_automaton.rl.hartford_schedule import MAX_GAME_SEED, training_schedule as original_schedule


DECKS = tuple(HARTFORD_DECKS)


class HartfordABScheduleTests(unittest.TestCase):
    def test_current_starts_with_real_predecessors_then_matches_original_recipe(self):
        for number, opponents in ((1, (1099, 1098, 1097)), (2, (0, 1099, 1098)),
                                  (3, (1, 0, 1099)), (4, (2, 1, 0))):
            rows = training_schedule(DECKS, round_id=number, seed=123000, recipe='current')
            self.assertEqual(Counter(g['opponent_version'] for g in rows),
                             {number - 1: 700, **{n: 100 for n in opponents}})
        for number in range(4, 51):
            self.assertEqual(training_schedule(DECKS, round_id=number, seed=123000, recipe='current'),
                             original_schedule(DECKS, round_id=number, seed=123000))
        self.assertEqual(external_versions('current'), (1097, 1098, 1099))
        self.assertEqual(external_versions('diverse'), HISTORICAL_VERSIONS)
        self.assertEqual((EXTERNAL_BASE, PARENT_ROUND), (1000, 100))

    def test_all_fifty_rounds_have_matched_games_balanced_roles_and_exact_quotas(self):
        seen = set()
        role_set = {(seat, f'P{start}') for seat in (0, 1) for start in (0, 1)}
        for number in range(1, 51):
            current = training_schedule(DECKS, round_id=number, seed=123000, recipe='current')
            diverse = training_schedule(DECKS, round_id=number, seed=123000, recipe='diverse')
            self.assertEqual(len(current), 1000)
            self.assertEqual(len(diverse), 1000)
            for a, b in zip(current, diverse):
                self.assertEqual({k: v for k, v in a.items() if k not in ('opponent_kind', 'opponent_version')},
                                 {k: v for k, v in b.items() if k not in ('opponent_kind', 'opponent_version')})
                self.assertNotIn(a['index'], seen)
                seen.add(a['index'])
                self.assertEqual(a['seed'], 123000 + a['index'])
                self.assertEqual(a['behavior_version'], number - 1)
                self.assertEqual(a['decks'][a['learner_seat']], a['learner_deck'])
                self.assertEqual(a['decks'][1 - a['learner_seat']], a['opponent_deck'])
            self.assertEqual(Counter(g['opponent_version'] for g in diverse),
                             {number - 1: 580, None: 120, **{n: 100 for n in HISTORICAL_VERSIONS}})
            for partition, latest, old, scripts in (('training', 464, 80, 96), ('validation', 116, 20, 24)):
                group = [g for g in diverse if g['partition'] == partition]
                self.assertEqual(Counter(g['opponent_version'] for g in group),
                                 {number - 1: latest, None: scripts, **{n: old for n in HISTORICAL_VERSIONS}})
                for opponent in {g['opponent_version'] for g in group}:
                    roles = Counter((g['learner_seat'], g['starting_player']) for g in group
                                    if g['opponent_version'] == opponent)
                    self.assertEqual(set(roles), role_set)
                    self.assertEqual(len(set(roles.values())), 1)
            cells = defaultdict(list)
            for row in diverse:
                cells[row['learner_deck'], row['opponent_deck']].append(row)
                self.assertEqual(row['opponent_kind'], 'scripted' if row['opponent_version'] is None else 'frozen_neural')
                self.assertNotIn('opponent_checkpoint_sha256', row)
                if row['opponent_kind'] == 'scripted':
                    self.assertIn(row['opponent_deck'], SUPPORTED_SCRIPT_DECKS)
            self.assertEqual(len(cells), 100)
            for (_, opponent), rows in cells.items():
                self.assertEqual(Counter(g['partition'] for g in rows), {'training': 8, 'validation': 2})
                expected = {n: 1 for n in HISTORICAL_VERSIONS}
                expected.update({number - 1: 3, None: 4} if opponent in SUPPORTED_SCRIPT_DECKS else {number - 1: 7})
                self.assertEqual(Counter(g['opponent_version'] for g in rows), expected)
                roles = Counter((g['learner_seat'], g['starting_player']) for g in rows if g['partition'] == 'training')
                self.assertEqual(roles, {r: 2 for r in role_set})
        self.assertEqual(seen, set(range(50000)))

    def test_determinism_private_rng_and_script_deck_identity(self):
        before = random.getstate()
        a = training_schedule(DECKS, round_id=11, seed=123456, recipe='diverse')
        self.assertEqual(a, training_schedule(DECKS, round_id=11, seed=123456, recipe='diverse'))
        self.assertEqual(random.getstate(), before)
        self.assertNotEqual([g['index'] for g in a], sorted(g['index'] for g in a))
        for deck in SUPPORTED_SCRIPT_DECKS:
            self.assertEqual(HARTFORD_DECKS[deck]['mainboard'], STANDARD_DECKS[deck]['mainboard'])

    def test_invalid_inputs_and_seed_limit(self):
        for number in (-1, 0, 51, True, 1.5):
            with self.assertRaises(ValueError):
                training_schedule(DECKS, round_id=number, seed=100, recipe='current')
        for seed in (-1, True, 1.5, MAX_GAME_SEED - 998):
            with self.assertRaises(ValueError):
                training_schedule(DECKS, round_id=1, seed=seed, recipe='diverse')
        for recipe in ('', 'other', None):
            with self.assertRaises(ValueError):
                training_schedule(DECKS, round_id=1, seed=100, recipe=recipe)
            with self.assertRaises(ValueError):
                external_versions(recipe)
        with self.assertRaises(ValueError):
            training_schedule(tuple(f'deck-{i}' for i in range(10)), round_id=1, seed=100, recipe='diverse')


if __name__ == '__main__':
    unittest.main()
