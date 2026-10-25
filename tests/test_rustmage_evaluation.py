"""Playing-strength metrics keep terminal results and failed attempts distinct."""
import unittest

from patchwork_automaton.neural.rustmage_evaluation import (
    baseline_evaluation_config, match_summary, paired_match_improvement,
)


def row(index, status, **changes):
    result = dict(index=index, seed=50000+index, seat=index % 2,
        starting_player=f'P{(index//2) % 2}', mode='versus_baseline',
        status=status, trajectories=[])
    result.update(changes)
    return result


class RustMageEvaluationTests(unittest.TestCase):
    def test_balanced_fixed_config_has_active_opponent_and_no_recording(self):
        config = baseline_evaluation_config(seed_base=50000)
        options = config['collection']
        self.assertEqual(options['games'], 200)
        self.assertEqual(options['mode'], 'versus_baseline')
        self.assertIsNone(options['output'])
        self.assertTrue(options['greedy'])
        self.assertEqual(config['opponent']['policy'], 'FoundationsAggro')
        self.assertEqual(config['opponent']['behavior'], 'active')
        self.assertEqual(config['cohort']['attempts_per_seat_start_stratum'], 50)
        self.assertFalse(config['cohort']['replace_failed_attempts'])
        for options in (dict(games=199), dict(index_base=1), dict(seed_base=-1),
                        dict(games=True), dict(greedy='yes')):
            with self.subTest(options=options), self.assertRaises(ValueError):
                baseline_evaluation_config(**(dict(seed_base=50000) | options))

    def test_wins_losses_draws_cutoffs_and_errors_keep_attempt_denominator(self):
        rows = [row(0, 'win', own_turns_to_win=3), row(1, 'loss'), row(2, 'draw'),
                row(3, 'cutoff', reason='repeated_visible_state'),
                row(4, 'error', error='failed inference')]
        result = match_summary(rows, expected_games=5)
        self.assertEqual(result['attempted_games'], 5)
        self.assertEqual(result['terminal_games'], 3)
        for field in ('wins', 'losses', 'draws', 'cutoffs', 'errors'):
            self.assertEqual(result[field], 1)
        self.assertEqual(result['unresolved_attempts'], 2)
        self.assertEqual(result['win_rate_all_attempts'], .2)
        self.assertEqual(result['terminal_win_rate'], 1/3)
        self.assertEqual(result['terminal_score_rate'], .5)
        interval = result['win_rate_all_attempts_wilson_95_interval']
        self.assertLess(interval[0], .2)
        self.assertGreater(interval[1], .2)
        self.assertFalse(result['seat_start_balanced'])
        self.assertNotIn('capped_turns_mean', result)
        self.assertNotIn('reward', result)
        self.assertEqual(rows[3]['status'], 'cutoff')
        self.assertNotIn('terminal_rewards', rows[3])

    def test_all_failures_and_empty_evaluation_have_no_terminal_metrics(self):
        failed = match_summary([row(0, 'cutoff'), row(1, 'error')])
        self.assertEqual(failed['win_rate_all_attempts'], 0.)
        self.assertIsNone(failed['terminal_win_rate'])
        self.assertIsNone(failed['terminal_score_rate'])
        self.assertGreater(failed['win_rate_all_attempts_wilson_95_interval'][1], 0.)
        empty = match_summary([], expected_games=0)
        self.assertIsNone(empty['win_rate_all_attempts'])
        self.assertIsNone(empty['win_rate_all_attempts_wilson_95_interval'])
        self.assertFalse(empty['seat_start_balanced'])

    def test_four_seat_start_strata_are_measured(self):
        result = match_summary([row(i, 'win') for i in range(200)], expected_games=200)
        self.assertTrue(result['seat_start_balanced'])
        self.assertEqual({s['attempted_games'] for s in result['seat_start_strata'].values()}, {50})
        self.assertEqual(result['win_rate_all_attempts'], 1.)
        self.assertAlmostEqual(result['win_rate_all_attempts_wilson_95_interval'][1], 1.)

    def test_missing_duplicate_wrong_opponent_and_recorded_games_are_rejected(self):
        invalid = ([row(0, 'win'), row(0, 'loss')],
                   [row(0, 'unknown')], [row(0, 'win', mode='goldfish')],
                   [row(0, 'win', trajectories=[dict(path='training.pkl.gz')])],
                   [row(0, 'win', seat=2)], [row(0, 'win', seed=-1)])
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                match_summary(rows)
        with self.assertRaises(ValueError):
            match_summary([row(0, 'win')], expected_games=2)

    def test_paired_changes_retain_failures_without_inventing_terminal_scores(self):
        initial = [row(0, 'loss'), row(1, 'win'), row(2, 'draw'),
                   row(3, 'cutoff'), row(4, 'error')]
        current = [row(0, 'win'), row(1, 'loss'), row(2, 'draw'),
                   row(3, 'win'), row(4, 'cutoff')]
        result = paired_match_improvement(initial, current, seed=7, resamples=300)
        self.assertEqual(result['attempted_pairs'], 5)
        self.assertEqual(result['terminal_pairs'], 3)
        self.assertEqual(result['unresolved_pairs'], 2)
        self.assertEqual(result['recorded_attempt_win_rate_change'], .2)
        self.assertEqual(result['terminal_score_change_on_terminal_pairs'], 0.)
        self.assertEqual(result['status_transitions']['cutoff->win'], 1)
        self.assertEqual(result['status_transitions']['error->cutoff'], 1)
        self.assertEqual(result['improved_recorded_win_attempts'], 2)
        self.assertEqual(result['worsened_recorded_win_attempts'], 1)
        self.assertEqual(result['tied_recorded_win_attempts'], 2)
        self.assertEqual(result, paired_match_improvement(
            initial[::-1], current[::-1], seed=7, resamples=300))

    def test_unresolved_pairs_never_receive_a_draw_or_loss_score(self):
        result = paired_match_improvement([row(0, 'error')], [row(0, 'cutoff')], resamples=20)
        self.assertEqual(result['recorded_attempt_win_rate_change'], 0.)
        self.assertEqual(result['terminal_pairs'], 0)
        self.assertIsNone(result['terminal_score_change_on_terminal_pairs'])
        self.assertIsNone(result['terminal_score_change_bootstrap_95_interval'])

    def test_pairing_and_bootstrap_arguments_are_validated(self):
        initial = [row(0, 'win')]
        for current in ([row(0, 'win', seed=12)], [row(0, 'win', seat=1)],
                        [row(0, 'win', starting_player='P1')], [row(1, 'win')], []):
            with self.subTest(current=current), self.assertRaises(ValueError):
                paired_match_improvement(initial, current)
        for resamples in (0, -1, True):
            with self.subTest(resamples=resamples), self.assertRaises(ValueError):
                paired_match_improvement(initial, initial, resamples=resamples)
        with self.assertRaises(ValueError):
            paired_match_improvement([], [])


if __name__ == '__main__':
    unittest.main()
