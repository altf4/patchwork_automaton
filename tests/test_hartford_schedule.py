"""Exact game, role, opponent, and holdout quotas for Hartford self-play."""

from collections import Counter, defaultdict
from itertools import product
import random
import unittest

from patchwork_automaton.rl.hartford_schedule import (
    MAX_GAME_SEED, evaluation_game_count, evaluation_schedule, evaluation_seed_bounds,
    previous_checkpoints, training_schedule, training_seed_bounds)


DECKS = tuple(f'deck-{i}' for i in range(10))


class HartfordScheduleTests(unittest.TestCase):
    def test_startup_and_previous_checkpoint_window(self):
        self.assertEqual([previous_checkpoints(i) for i in range(5)],
                         [(), (0,), (1, 0), (2, 1, 0), (3, 2, 1)])
        expected = [
            {'training': {0: 800}, 'validation': {0: 200}},
            {'training': {1: 560, 0: 240}, 'validation': {1: 140, 0: 60}},
            {'training': {2: 560, 1: 120, 0: 120}, 'validation': {2: 140, 1: 30, 0: 30}},
            {'training': {3: 560, 2: 80, 1: 80, 0: 80},
             'validation': {3: 140, 2: 20, 1: 20, 0: 20}},
        ]
        for number, counts in enumerate(expected, 1):
            rows = training_schedule(DECKS, round_id=number, seed=1000)
            for partition, expected_counts in counts.items():
                self.assertEqual(Counter(g['opponent_version'] for g in rows
                                         if g['partition'] == partition), expected_counts)

    def test_full_100_round_experiment_has_exact_balanced_cells(self):
        seen_ids, seen_seeds = set(), set()
        holdout_roles = defaultdict(Counter)
        for number in range(1, 101):
            rows = training_schedule(DECKS, round_id=number, seed=100_000)
            self.assertEqual(len(rows), 1000)
            self.assertEqual(Counter(g['partition'] for g in rows),
                             {'training': 800, 'validation': 200})
            for partition in ('training', 'validation'):
                for opponent in {g['opponent_version'] for g in rows}:
                    roles = Counter((g['learner_seat'], g['starting_player']) for g in rows
                                    if g['partition'] == partition and g['opponent_version'] == opponent)
                    counts = [roles[s, f'P{p}'] for s in (0, 1) for p in (0, 1)]
                    self.assertLessEqual(max(counts) - min(counts), 1)
            cells = defaultdict(list)
            for game in rows:
                self.assertNotIn(game['index'], seen_ids)
                self.assertNotIn(game['seed'], seen_seeds)
                seen_ids.add(game['index'])
                seen_seeds.add(game['seed'])
                self.assertEqual(game['behavior_version'], number - 1)
                self.assertEqual(game['opponent_kind'], 'frozen_neural')
                self.assertEqual(game['deck_catalog'], 'hartford')
                self.assertEqual(game['decks'][game['learner_seat']], game['learner_deck'])
                self.assertEqual(game['decks'][1 - game['learner_seat']], game['opponent_deck'])
                self.assertEqual(game['pairing'], sorted(game['decks'], key=DECKS.index))
                cells[game['learner_deck'], game['opponent_deck']].append(game)
            self.assertEqual(set(cells), set(product(DECKS, repeat=2)))
            for cell, games in cells.items():
                training = [g for g in games if g['partition'] == 'training']
                heldout = [g for g in games if g['partition'] == 'validation']
                self.assertEqual(len(training), 8)
                self.assertEqual(len(heldout), 2)
                self.assertEqual(Counter((g['learner_seat'], g['starting_player']) for g in training),
                                 {(s, f'P{p}'): 2 for s in (0, 1) for p in (0, 1)})
                self.assertEqual(Counter(g['learner_seat'] for g in heldout), {0: 1, 1: 1})
                self.assertEqual(Counter(g['starting_player'] == f"P{g['learner_seat']}"
                                         for g in heldout), {True: 1, False: 1})
                holdout_roles[cell].update((g['learner_seat'], g['starting_player']) for g in heldout)
                if number >= 4:
                    self.assertEqual(Counter(g['opponent_version'] for g in games),
                                     {number - 1: 7, number - 2: 1, number - 3: 1, number - 4: 1})
            if number >= 4:
                for partition, latest, old in (('training', 560, 80), ('validation', 140, 20)):
                    self.assertEqual(Counter(g['opponent_version'] for g in rows
                                             if g['partition'] == partition),
                                     {number - 1: latest, number - 2: old,
                                      number - 3: old, number - 4: old})
        self.assertEqual(len(seen_ids), 100_000)
        self.assertEqual(len(seen_seeds), 100_000)
        self.assertEqual(seen_ids, set(range(100_000)))
        self.assertTrue(all(counts == {(s, f'P{p}'): 50 for s in (0, 1) for p in (0, 1)}
                            for counts in holdout_roles.values()))

    def test_evaluation_all_cells_balanced_and_unique_across_windows(self):
        seen_ids, seen_seeds = set(), set()
        for checkpoint in range(1, 6):
            for opponent in previous_checkpoints(checkpoint):
                rows = evaluation_schedule(DECKS, checkpoint_round=checkpoint,
                                           opponent_round=opponent, seed=1_000_000)
                self.assertEqual(len(rows), 4000)
                cells = defaultdict(Counter)
                for game in rows:
                    self.assertNotIn(game['index'], seen_ids)
                    self.assertNotIn(game['seed'], seen_seeds)
                    seen_ids.add(game['index'])
                    seen_seeds.add(game['seed'])
                    self.assertEqual(game['partition'], 'evaluation')
                    self.assertEqual(game['behavior_version'], checkpoint)
                    self.assertEqual(game['opponent_version'], opponent)
                    self.assertEqual(game['opponent_kind'], 'frozen_neural')
                    self.assertEqual(game['deck_catalog'], 'hartford')
                    cells[game['learner_deck'], game['opponent_deck']][
                        game['learner_seat'], game['starting_player']] += 1
                self.assertEqual(set(cells), set(product(DECKS, repeat=2)))
                self.assertTrue(all(c == {(s, f'P{p}'): 10 for s in (0, 1) for p in (0, 1)}
                                    for c in cells.values()))

    def test_deterministic_order_and_no_global_rng_change(self):
        state = random.getstate()
        first = training_schedule(DECKS, round_id=5, seed=123)
        self.assertEqual(first, training_schedule(DECKS, round_id=5, seed=123))
        self.assertNotEqual([g['index'] for g in first], sorted(g['index'] for g in first))
        args = dict(checkpoint_round=5, opponent_round=3, seed=987)
        evaluation = evaluation_schedule(DECKS, **args)
        self.assertEqual(evaluation, evaluation_schedule(DECKS, **args))
        smaller = evaluation_schedule(DECKS, games_per_cell=4, **args)
        by_id = {g['index']: g for g in evaluation}
        self.assertTrue(all(by_id[g['index']] == g for g in smaller))
        self.assertEqual(random.getstate(), state)

    def test_count_and_seed_bounds_cover_actual_schedules(self):
        self.assertEqual(evaluation_game_count(100), 1_188_000)
        for rounds in range(1, 6):
            rows = [g for checkpoint in range(1, rounds + 1)
                    for opponent in previous_checkpoints(checkpoint)
                    for g in evaluation_schedule(DECKS, checkpoint_round=checkpoint,
                                                 opponent_round=opponent,
                                                 games_per_cell=4, seed=1_000_000)]
            self.assertEqual(len(rows), evaluation_game_count(rounds, games_per_cell=4))
            self.assertEqual((min(g['seed'] for g in rows), max(g['seed'] for g in rows)),
                             evaluation_seed_bounds(rounds=rounds, games_per_cell=4, seed=1_000_000))
        self.assertEqual(training_seed_bounds(rounds=100, seed=123), (123, 100_122))
        self.assertEqual(training_seed_bounds(rounds=1, seed=MAX_GAME_SEED - 999),
                         (MAX_GAME_SEED - 999, MAX_GAME_SEED))
        with self.assertRaisesRegex(ValueError, 'player seeds'):
            training_schedule(DECKS, round_id=1, seed=MAX_GAME_SEED - 998)
        with self.assertRaisesRegex(ValueError, 'player seeds'):
            evaluation_schedule(DECKS, checkpoint_round=1, opponent_round=0,
                                games_per_cell=4, seed=MAX_GAME_SEED)
        with self.assertRaisesRegex(ValueError, 'player seeds'):
            evaluation_seed_bounds(rounds=100, seed=MAX_GAME_SEED)

    def test_invalid_inputs(self):
        for decks in (DECKS[:9], DECKS + ('extra',), DECKS[:9] + ('deck-0',), DECKS[:9] + ('',)):
            with self.assertRaises(ValueError):
                training_schedule(decks, round_id=1, seed=1)
        for number in (-1, 0, True, 1.5):
            with self.assertRaises(ValueError):
                training_schedule(DECKS, round_id=number, seed=1)
        for seed in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                training_schedule(DECKS, round_id=1, seed=seed)
        for number in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                previous_checkpoints(number)
        for opponent in (-1, 0, 4, 5):
            with self.assertRaises(ValueError):
                evaluation_schedule(DECKS, checkpoint_round=4, opponent_round=opponent, seed=1)
        for size in (0, 2, 5, True, 4.5):
            with self.assertRaises(ValueError):
                evaluation_schedule(DECKS, checkpoint_round=1, opponent_round=0,
                                    games_per_cell=size, seed=1)


if __name__ == '__main__':
    unittest.main()
