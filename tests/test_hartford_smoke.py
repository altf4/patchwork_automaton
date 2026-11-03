"""The real-engine smoke is bounded, disjoint, and gated before initialization."""
import asyncio
from collections import Counter
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from examples.smoke_hartford_selfplay import (
    configuration, parser, production_config, run, schedule, summarize, verify_behavior)
from examples.train_hartford_ppo import configuration as train_config, parser as train_parser
from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS
from patchwork_automaton.neural.rollout_records import Sample, TrajectoryWriter


class HartfordSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = parser().parse_args(['--output', str(self.root/'smoke')])

    def test_uses_identical_production_random_configuration_and_separate_seed_spaces(self):
        production = production_config(self.args)
        self.assertEqual(production, train_config(train_parser().parse_args([])))
        smoke = configuration(self.args, production)
        self.assertEqual(smoke['games'], 400)
        self.assertEqual(smoke['weight_seed'], production['weight_seed'])
        self.assertEqual(smoke['checkpoint_round'], 0)
        self.assertEqual(smoke['opponent_round'], 0)
        self.assertFalse(smoke['optimize'])
        first, last = smoke['seed_bounds']
        for low, high in smoke['production_seed_bounds'].values():
            self.assertTrue(last < low or high < first)

    def test_all_matchups_have_four_balanced_roles_and_no_global_random_mutation(self):
        before = random.getstate()
        rows = schedule(tuple(HARTFORD_DECKS), seed=self.args.seed, checkpoint_sha256='a'*64)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(rows, schedule(tuple(HARTFORD_DECKS), seed=self.args.seed, checkpoint_sha256='a'*64))
        self.assertEqual(len(rows), 400)
        self.assertEqual({row['index'] for row in rows}, set(range(400)))
        self.assertEqual({row['seed'] for row in rows}, set(range(self.args.seed, self.args.seed+400)))
        cells = Counter((row['learner_deck'], row['opponent_deck']) for row in rows)
        self.assertEqual(len(cells), 100)
        self.assertEqual(set(cells.values()), {4})
        for deck, opponent in cells:
            roles = {(row['learner_seat'], row['starting_player']) for row in rows
                     if (row['learner_deck'], row['opponent_deck']) == (deck, opponent)}
            self.assertEqual(roles, {(0, 'P0'), (0, 'P1'), (1, 'P0'), (1, 'P1')})
        self.assertEqual({row['partition'] for row in rows}, {'smoke'})
        self.assertEqual({(row['behavior_version'], row['opponent_version']) for row in rows}, {(0, 0)})
        for row in rows:
            self.assertEqual(row['decks'][row['learner_seat']], row['learner_deck'])
            self.assertEqual(row['decks'][1-row['learner_seat']], row['opponent_deck'])

    def test_invalid_cohort_or_overlapping_seeds_rejected(self):
        production = production_config(self.args)
        for change in (dict(games=100), dict(seed=production['seed']),
                       dict(seed=production['eval_seed']+1), dict(workers=1),
                       dict(inference_processes=100), dict(seed=-1)):
            args = deepcopy(self.args)
            for key, value in change.items():
                setattr(args, key, value)
            with self.assertRaises(ValueError):
                configuration(args, production)

    def test_preflight_rejects_before_creating_output_or_model(self):
        with patch('examples.smoke_hartford_selfplay.hartford_preflight',
                   return_value=dict(ready=False, errors=['unready engine'])), \
             patch('examples.smoke_hartford_selfplay.initialize_random') as initialize:
            with self.assertRaisesRegex(ValueError, 'unready engine'):
                asyncio.run(run(self.args))
            initialize.assert_not_called()
        self.assertFalse(self.args.output.exists())

    def test_preflight_only_does_not_import_tensorflow_or_create_output(self):
        code = (
            'import asyncio, sys; from examples.smoke_hartford_selfplay import parser, run; '
            f'args=parser().parse_args(["--preflight-only","--output",{str(self.args.output)!r}]); '
            'result=asyncio.run(run(args)); '
            'assert "readiness" in result; '
            'assert not any(n.startswith("tensorflow") for n in sys.modules); '
            'assert not args.output.exists()')
        process = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('production_seed_bounds', process.stdout)

    def test_existing_smoke_evidence_is_not_overwritten(self):
        self.args.output.mkdir()
        sentinel = self.args.output/'manifest.json'
        sentinel.write_text('preserved failure')
        with patch('examples.smoke_hartford_selfplay.hartford_preflight', return_value=dict(ready=True)), \
             patch('examples.smoke_hartford_selfplay.initialize_random') as initialize:
            with self.assertRaisesRegex(FileExistsError, 'never replaced'):
                asyncio.run(run(self.args))
            initialize.assert_not_called()
        self.assertEqual(sentinel.read_text(), 'preserved failure')

    def test_recorded_behavior_reproduction_is_bounded_and_rejects_mismatched_weights(self):
        path = self.root/'trajectory.pkl.gz'
        writer = TrajectoryWriter(path)
        for _ in range(3):
            writer.write(Sample(dict(candidate_mask=np.ones(2, bool)), np.zeros(1, np.float32),
                                True, 0, .25, np.log([.5, .5]).astype(np.float32)))
        writer.close()
        rows = [dict(trajectories=[dict(path=str(path))])]
        def model(inputs, training):
            self.assertFalse(training)
            return dict(logits=np.array([[0., 0.]]), value=np.array([[.25]]))
        with patch('patchwork_automaton.neural.batching.record_batch', return_value={}), \
             patch('patchwork_automaton.neural.devices.device_scope', side_effect=lambda device: nullcontext()):
            result = verify_behavior(model, rows, device='cpu')
            self.assertEqual((result['samples'], result['games']), (2, 1))
            with self.assertRaises(AssertionError):
                verify_behavior(lambda inputs, training: dict(logits=np.array([[0., 1.]]),
                    value=np.array([[.25]])), rows, device='cpu')

    def test_report_keeps_errors_cutoffs_and_adjudications_separate(self):
        row = schedule(tuple(HARTFORD_DECKS), seed=self.args.seed, checkpoint_sha256='a'*64)[0]
        rows = [dict(row, status='loss', learner_won=True, seconds=1., decisions=2),
                dict(row, status='draw', learner_won=False, adjudicated=True,
                     reason='wall_clock_timeout', seconds=60., decisions=20),
                dict(row, status='draw', learner_won=False, seconds=2., decisions=3),
                dict(row, status='error', seconds=3., decisions=4),
                dict(row, status='cutoff', seconds=4., decisions=6000)]
        result = summarize(rows)
        self.assertEqual(result['learner_wins'], 1)
        self.assertEqual(result['learner_losses'], 0)
        self.assertEqual(result['draws'], 2)
        self.assertEqual(result['natural_draws'], 1)
        self.assertEqual(result['natural_completed_games'], 2)
        self.assertEqual(result['natural_completion_rate'], .4)
        self.assertEqual((result['errors'], result['cutoffs']), (1, 1))
        self.assertEqual(result['adjudications'], {'wall_clock_timeout': 1})
        self.assertEqual(result['game_seconds']['maximum'], 60.)

    def test_production_manifest_cannot_change_initialization_mode_or_weight_seed(self):
        path = self.root/'production.json'
        production = production_config(self.args)
        path.write_text(json.dumps(dict(config=production)))
        self.args.production_manifest = path
        self.assertEqual(production_config(self.args), production)
        path.write_text(json.dumps(dict(config=dict(production, weight_seed=42))))
        with self.assertRaisesRegex(ValueError, 'weight seed/playback'):
            production_config(self.args)
        path.write_text(json.dumps(dict(config=dict(production, initialization='imitation'))))
        with self.assertRaisesRegex(ValueError, 'random sampled'):
            production_config(self.args)


if __name__ == '__main__':
    unittest.main()
