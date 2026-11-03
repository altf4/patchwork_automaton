"""A longer run retains historical state and does not reinterpret evaluation."""
from copy import deepcopy
import json
import unittest

from examples.train_hartford_ppo import (atomic_json, checkpoint_configuration,
    checkpoint_hashes, evaluation_source_pins, restore_history,
    required_evaluation_rounds, validate_evaluation_completion)
import test_hartford_driver as fixtures


class TargetExtensionTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.HartfordDriverTests('test_exact_default_recipe_and_no_warm_start_options')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        original = dict(fixture.config, rounds=2)
        fixture.config = original
        zero, one = fixture.make_checkpoint(0), fixture.make_checkpoint(1)
        previous = dict(original, collection_mode='mixed-frozen-opponents-v1', game_timeout=20)
        fixture.config = previous
        two = fixture.make_checkpoint(2)
        atomic_json(two.parent / 'latest.json', dict(checkpoint=two.name))
        self.paths = [zero, one, two]
        self.hashes = {str(n): checkpoint_hashes(p) for n, p in enumerate(self.paths)}
        self.config = dict(previous, rounds=4)
        self.report = dict(config=self.config, rounds=[], checkpoints=deepcopy(self.hashes),
            execution_amendments=[dict(previous_config=original, new_config=previous,
                through_checkpoint=1)],
            training_target_amendment=dict(previous_config=previous, new_config=self.config,
                through_checkpoint=2, previous_pins={'source': {'old': 'hash'}}),
            evaluations={'2': {'complete': True}})
        # restore_history permits one checkpoint publication gap, not two.
        self.report['rounds'] = [json.loads((one / 'state.json').read_text())['round_result']]

    def test_resume_preserves_history_and_extends_with_current_config(self):
        self.assertEqual(restore_history(self.report, self.paths[0].parent, self.config), 2)
        self.assertEqual({str(n): checkpoint_hashes(p) for n, p in enumerate(self.paths)}, self.hashes)
        self.fixture.config = self.config
        three = self.fixture.make_checkpoint(3)
        self.assertEqual(restore_history(self.report, three.parent, self.config), 3)
        self.assertEqual(checkpoint_configuration(self.report, self.config, 3), self.config)

    def test_target_extension_rejects_recipe_changes_and_shorter_targets(self):
        for change in ({'learning_rate': .9}, {'rounds': 2}, {'rounds': 1}):
            config = dict(self.config, **change)
            report = deepcopy(self.report)
            report['training_target_amendment']['new_config'] = config
            with self.assertRaisesRegex(ValueError, 'Target extension'):
                checkpoint_configuration(report, config, 0)

    def test_completed_evaluation_keeps_original_provenance(self):
        pins = {'source': {'new': 'hash'}}
        self.assertEqual(evaluation_source_pins(self.report, pins, 2), {'source': {'old': 'hash'}})
        self.assertEqual(evaluation_source_pins(self.report, pins, 3), pins)
        self.report['evaluations']['2']['complete'] = False
        with self.assertRaisesRegex(ValueError, 'unfinished historical'):
            evaluation_source_pins(self.report, pins, 2)

    def test_200k_completion_requires_all_twenty_evaluations(self):
        config = dict(self.config, rounds=200, evaluation_every_games=10000)
        rounds = set(range(10, 201, 10))
        self.assertEqual(required_evaluation_rounds(config, {}), rounds)
        report = {'evaluations': {str(n): {'complete': True, 'aggregate': {'games': 12000}}
                                  for n in rounds}}
        self.assertEqual(validate_evaluation_completion(report, config), 240000)
        del report['evaluations']['110']
        with self.assertRaises(ValueError):
            validate_evaluation_completion(report, config)

    def test_runtime_fix_preserves_both_historical_source_boundaries(self):
        previous = {'source': {'new': 'hash'}}
        current = {'source': {'rules_fix': 'hash'}}
        self.report['runtime_amendments'] = [dict(previous_pins=previous,
            new_pins=current, through_checkpoint=3)]
        self.report['evaluations']['3'] = {'complete': True}
        self.assertEqual(evaluation_source_pins(self.report, current, 4), current)
        self.assertEqual(evaluation_source_pins(self.report, current, 3), previous)
        self.assertEqual(evaluation_source_pins(self.report, current, 2),
                         {'source': {'old': 'hash'}})
        with self.assertRaisesRegex(ValueError, 'chain changed'):
            evaluation_source_pins(self.report, {'different': 'pins'}, 3)


if __name__ == '__main__':
    unittest.main()
