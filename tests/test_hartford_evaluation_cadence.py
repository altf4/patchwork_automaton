"""Cadence changes preserve checkpoint history and keep completion exact."""
from copy import deepcopy
import unittest

from examples.train_hartford_ppo import (configuration, parser, required_evaluation_rounds,
    restore_history, validate_evaluation_completion, checkpoint_hashes, atomic_json)
import test_hartford_driver as fixtures


class EvaluationCadenceTests(unittest.TestCase):
    def setUp(self):
        self.config = configuration(parser().parse_args(['--eval-every-games', '10000']))

    def amendment(self, previous=None, boundary=10, retained=range(1, 10)):
        return dict(previous_config=previous or configuration(parser().parse_args([])),
                    new_config=self.config, through_checkpoint=boundary,
                    retained_evaluation_rounds=list(retained))

    def test_milestones_and_final_round(self):
        self.assertEqual(required_evaluation_rounds(self.config, {}), set(range(10, 101, 10)))
        self.assertFalse(self.config['evaluation_every_round'])
        short = dict(self.config, rounds=23)
        self.assertEqual(required_evaluation_rounds(short, {}), {10, 20, 23})
        for value in ('0', '999', '1500', '-1000'):
            with self.assertRaises(ValueError):
                configuration(parser().parse_args(['--eval-every-games', value]))

    def test_historical_evaluations_remain_required(self):
        report = dict(evaluation_cadence_amendment=self.amendment())
        wanted = set(range(1, 10)) | set(range(10, 101, 10))
        self.assertEqual(required_evaluation_rounds(self.config, report), wanted)
        report['evaluations'] = {str(n): dict(complete=True,
            aggregate=dict(games=min(n, 3) * 4000)) for n in wanted}
        self.assertEqual(validate_evaluation_completion(report, self.config), 216000)
        del report['evaluations']['20']
        with self.assertRaisesRegex(ValueError, 'missing a complete'):
            validate_evaluation_completion(report, self.config)

    def test_fresh_sparse_run_counts_only_milestones(self):
        report = dict(evaluations={str(n): dict(complete=True, aggregate=dict(games=12000))
                                  for n in range(10, 101, 10)})
        self.assertEqual(validate_evaluation_completion(report, self.config), 120000)
        report['evaluations']['10']['aggregate']['games'] -= 1
        with self.assertRaises(ValueError):
            validate_evaluation_completion(report, self.config)

    def test_resume_accepts_old_configs_without_rewriting_checkpoint_files(self):
        fixture = fixtures.HartfordDriverTests('test_exact_default_recipe_and_no_warm_start_options')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        # Legacy checkpoints do not contain the newly introduced interval key.
        fixture.config.pop('evaluation_every_games')
        zero, one = fixture.make_checkpoint(0), fixture.make_checkpoint(1)
        atomic_json(zero.parent / 'latest.json', dict(checkpoint=one.name))
        hashes = {'0': checkpoint_hashes(zero), '1': checkpoint_hashes(one)}
        report = dict(rounds=[], checkpoints=deepcopy(hashes),
                      evaluation_cadence_amendment=self.amendment(fixture.config, 1, [1]))
        self.assertEqual(restore_history(report, zero.parent, self.config), 1)
        self.assertEqual({str(n): checkpoint_hashes(p) for n, p in enumerate((zero, one))}, hashes)
        fixture.config = self.config
        two = fixture.make_checkpoint(2)
        self.assertEqual(restore_history(report, zero.parent, self.config), 2)
        self.assertEqual(report['checkpoints']['2'], checkpoint_hashes(two))
        report['evaluation_cadence_amendment']['previous_config']['learning_rate'] = .9
        with self.assertRaisesRegex(ValueError, 'training configuration'):
            restore_history(report, zero.parent, self.config)


if __name__ == '__main__':
    unittest.main()
