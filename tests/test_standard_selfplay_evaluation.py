"""Read-only evaluation schedules and outcome accounting."""

from collections import Counter
import unittest

from examples.evaluate_standard_selfplay import (TACTICAL_SEED, cells,
    expected_game, scheduled_games, summarize, validate_row)


class StandardSelfplayEvaluationTests(unittest.TestCase):
    def test_nine_ordered_matchups_two_modes_and_four_balanced_roles(self):
        self.assertEqual(len(cells()), 18)
        ids = set()
        for deck, opponent, mode in cells():
            jobs = list(scheduled_games(deck, opponent, mode))
            self.assertEqual(len(jobs), 400)
            self.assertEqual({job['seed'] for job in jobs},
                             set(range(TACTICAL_SEED, TACTICAL_SEED + 400)))
            self.assertFalse(ids & {job['index'] for job in jobs})
            ids.update(job['index'] for job in jobs)
            strata = Counter((job['game']['learner_seat'], job['game']['starting_player'])
                             for job in jobs)
            self.assertEqual(set(strata.values()), {100})
            for job in jobs:
                game = expected_game(job, deck, opponent)
                self.assertEqual(game['decks'][game['learner_seat']], deck)
                self.assertEqual(game['decks'][1 - game['learner_seat']], opponent)
                self.assertEqual(game['partition'], 'evaluation')
        self.assertEqual(len(ids), 7200)

    def test_outcomes_and_sides_are_accounted_without_training_trajectories(self):
        jobs = list(scheduled_games('mono-green-landfall', 'izzet-prowess', 'sampled',
                                    games_per_cell=4))
        rows = []
        for i, job in enumerate(jobs):
            game = job['game']
            draw = i == 2
            winner = None if draw else game['learner_seat'] if i in (0, 3) else 1 - game['learner_seat']
            row = dict(game, behavior_version=0, status='draw' if draw else 'win',
                       winner_seat=winner, draw=draw, learner_won=winner == game['learner_seat'],
                       trajectories=[])
            validate_row(row, job, 'mono-green-landfall', 'izzet-prowess')
            rows.append(row)
        result = summarize(rows)
        self.assertEqual((result['wins'], result['losses'], result['draws']), (2, 1, 1))
        self.assertEqual((result['on_play']['games'], result['on_draw']['games']), (2, 2))
        self.assertEqual(result['score'], .625)
        rows[0]['trajectories'] = [dict(player=0)]
        with self.assertRaises(ValueError):
            validate_row(rows[0], jobs[0], 'mono-green-landfall', 'izzet-prowess')


if __name__ == '__main__':
    unittest.main()
