"""Disabling future duels preserves historical configurations and provenance."""
from copy import deepcopy
import unittest

from examples.train_hartford_ppo import (configuration, parser, checkpoint_configuration,
    evaluation_source_pins, required_evaluation_rounds, validate_evaluation_completion)


class SkipEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.old = configuration(parser().parse_args(['--rounds', '200', '--eval-every-games', '10000']))
        self.new = dict(self.old, skip_evaluations=True)
        self.pins = {'source': {'current': 'hash'}}
        self.previous_pins = {'source': {'previous': 'hash'}}
        self.report = dict(config=self.new,
            training_target_amendment=dict(previous_config=dict(self.old, rounds=100),
                new_config=self.old, through_checkpoint=100, previous_pins={'source': {'original': 'hash'}}),
            evaluations={str(n): dict(complete=True, aggregate=dict(games=12000))
                for n in range(10, 131, 10)},
            evaluation_disable_amendment=dict(previous_config=self.old, new_config=self.new,
                through_checkpoint=137, retained_evaluation_rounds=list(range(10, 131, 10)),
                previous_pins=self.previous_pins, new_pins=self.pins))

    def test_fresh_disabled_run_has_no_final_evaluation(self):
        config = configuration(parser().parse_args(['--skip-evaluations']))
        self.assertTrue(config['skip_evaluations'])
        self.assertNotIn('skip_evaluations', self.old)
        self.assertEqual(required_evaluation_rounds(config, {}), set())
        self.assertEqual(validate_evaluation_completion({'evaluations': {}}, config), 0)

    def test_retains_completed_evaluations_and_historical_checkpoint_configs(self):
        self.assertEqual(required_evaluation_rounds(self.new, self.report), set(range(10, 131, 10)))
        self.assertEqual(validate_evaluation_completion(self.report, self.new), 156000)
        self.assertEqual(checkpoint_configuration(self.report, self.new, 0), dict(self.old, rounds=100))
        self.assertEqual(checkpoint_configuration(self.report, self.new, 137), self.old)
        self.assertEqual(checkpoint_configuration(self.report, self.new, 138), self.new)
        self.assertEqual(evaluation_source_pins(self.report, self.pins, 130), self.previous_pins)
        self.assertEqual(evaluation_source_pins(self.report, self.pins, 100), {'source': {'original': 'hash'}})

    def test_rejects_unrelated_recipe_changes_and_missing_history(self):
        report = deepcopy(self.report)
        config = dict(self.new, learning_rate=.9)
        report['evaluation_disable_amendment']['new_config'] = config
        with self.assertRaisesRegex(ValueError, 'training configuration'):
            checkpoint_configuration(report, config, 137)
        for mutate in (lambda r: r['evaluations'].pop('130'),
                       lambda r: r['evaluations']['130'].update(complete=False),
                       lambda r: r['evaluation_disable_amendment']['retained_evaluation_rounds'].append(130)):
            report = deepcopy(self.report)
            mutate(report)
            with self.assertRaises(ValueError):
                required_evaluation_rounds(self.new, report)
        with self.assertRaisesRegex(ValueError, 'audited amendment'):
            required_evaluation_rounds(self.new, {'evaluations': self.report['evaluations']})
        with self.assertRaises(ValueError):
            evaluation_source_pins(self.report, self.pins, 140)


if __name__ == '__main__':
    unittest.main()
