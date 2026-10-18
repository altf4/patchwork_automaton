"""Learning-curve summaries retain failures and missing kill-time estimates."""
import unittest
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE, 'Install neural dependencies with uv')
class CurveTests(unittest.TestCase):
    def test_means_include_only_kills_but_failures_remain_visible(self):
        from examples.plot_rustmage_training import curve_points
        rows = [dict(status='win', own_turns_to_win=6), dict(status='win', own_turns_to_win=10),
                dict(status='loss', own_turns_to_win=None), dict(status='cutoff', own_turns_to_win=None)]
        point = dict(training_games=100, checkpoint='policy.keras', games=rows,
                     summary=dict(capped_turns_mean=17.))
        result = curve_points(dict(evaluations=[point]))[0]
        self.assertEqual(result['mean_kill_turn'], 8.)
        self.assertEqual(result['failures'], 2)
        self.assertEqual(result['completion_rate'], .5)
        self.assertEqual(result['capped_turns_mean'], 17.)
        self.assertLessEqual(result['ci95_low'], 8.)
        self.assertGreaterEqual(result['ci95_high'], 8.)

    def test_no_successes_has_no_fabricated_kill_time(self):
        from examples.plot_rustmage_training import curve_points
        point = dict(training_games=0, checkpoint='policy.keras',
                     games=[dict(status='error', own_turns_to_win=None)], summary=dict(capped_turns_mean=26.))
        result = curve_points(dict(evaluations=[point]))[0]
        self.assertIsNone(result['mean_kill_turn'])
        self.assertIsNone(result['ci95_low'])
        self.assertEqual(result['completion_rate'], 0.)
        self.assertEqual(result['failures'], 1)
