"""Frozen checkpoint identity, all-matchup reporting, and durable duel resume."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from patchwork_automaton.rl.hartford_evaluation import (
    evaluate_checkpoint, outcome_summary, validate_row)
from patchwork_automaton.neural.inference_processes import weights_digest
from patchwork_automaton.rl.hartford_schedule import evaluation_schedule


DECKS = tuple(f'deck-{i}' for i in range(10))


class FakeModel:
    def __init__(self, value):
        self.weights = [np.array([value], dtype=np.float32)]

    def get_weights(self):
        return [value.copy() for value in self.weights]

    def set_weights(self, weights):
        self.weights = [value.copy() for value in weights]


def successful_row(game, outcome='win'):
    winner = None if outcome == 'draw' else game['learner_seat'] if outcome == 'win' else 1-game['learner_seat']
    return dict(game, status='draw' if winner is None else 'win' if winner == 0 else 'loss',
                trajectories=[], winner_seat=winner, draw=winner is None,
                learner_won=winner == game['learner_seat'])


class FakeCollector:
    def __init__(self):
        self.seen = []
        self.options = []
        self.closed = False
        self.fail_after = None
        self.corrupt = None

    async def sync_weights(self, model, version):
        self.version = version
        self.policy_digest = weights_digest(model.get_weights())

    async def sync_opponent(self, model, version):
        self.opponent_version = version
        self.opponent_digest = weights_digest(model.get_weights())

    async def run_schedule(self, jobs, *, progress, **options):
        self.options.append(options)
        for job in jobs:
            if self.fail_after is not None and len(self.seen) >= self.fail_after:
                raise RuntimeError('simulated interruption')
            self.seen.append(job['index'])
            row = successful_row(job['game'])
            if self.corrupt:
                row.update(self.corrupt)
            progress(row)
        return []

    async def close(self):
        self.closed = True


class HartfordEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.paths = {}
        for number in range(5):
            path = self.root / f'{number}.keras'
            path.write_text(str(number))
            path.with_suffix('.manifest.json').write_text('{}')
            self.paths[number] = path
        self.model = FakeModel(3)
        self.opponent = FakeModel(0)
        self.collector = FakeCollector()
        self.loader = patch('patchwork_automaton.rl.hartford_evaluation._load_model',
                            side_effect=lambda path: FakeModel(int(Path(path).read_text())))
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.pins = patch('patchwork_automaton.rl.hartford_evaluation._runtime_pins', return_value={'source': 'fixed'})
        self.pins.start()
        self.addCleanup(self.pins.stop)

    def run_evaluation(self, **overrides):
        args = dict(model=self.model, checkpoint_round=3, checkpoint_paths=self.paths,
                    output=self.root / 'evaluation', deck_ids=DECKS, seed=1000000,
                    source_manifest={'source': 'frozen'}, games_per_cell=4, workers=1,
                    runners=1, inference_processes=1, opponent_inference_processes=1,
                    opponent_model=self.opponent, collectors=[self.collector])
        args.update(overrides)
        return asyncio.run(evaluate_checkpoint(**args))

    def test_every_matchup_both_sampled_frozen_versions_and_no_training(self):
        before = weights_digest(self.model.get_weights())
        result = self.run_evaluation()
        self.assertTrue(result['complete'])
        self.assertEqual(result['opponent_rounds'], [2, 1, 0])
        self.assertEqual(result['aggregate']['games'], 1200)
        self.assertEqual(len(set(self.collector.seen)), 1200)
        self.assertEqual(result['score'], 1.)
        self.assertEqual(weights_digest(self.model.get_weights()), before)
        self.assertFalse(self.collector.closed)
        for summary in result['opponents'].values():
            self.assertEqual(summary['games'], 400)
            self.assertEqual(summary['on_play']['games'], 200)
            self.assertEqual(summary['on_draw']['games'], 200)
            self.assertEqual(summary['score_matrix'], [[1.] * 10 for _ in range(10)])
            self.assertEqual({cell['games'] for row in summary['matchups'].values() for cell in row.values()}, {4})
        for options in self.collector.options:
            self.assertIs(options['greedy'], False)
            self.assertIsNone(options['output'])
            self.assertEqual(options['version'], 3)
        self.assertEqual([options['opponent_version'] for options in self.collector.options], [2, 1, 0])

    def test_exact_resume_runs_only_missing_games_and_revalidates_complete_results(self):
        self.collector.fail_after = 7
        with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
            self.run_evaluation()
        self.assertEqual(len(list((self.root/'evaluation').glob('opponent-*/games/game-*.json'))), 7)
        self.collector.fail_after = None
        self.run_evaluation()
        self.assertEqual(len(self.collector.seen), 1200)
        self.assertEqual(len(set(self.collector.seen)), 1200)
        self.run_evaluation()
        self.assertEqual(len(self.collector.seen), 1200)
        path = next((self.root/'evaluation').glob('opponent-*/games/game-*.json'))
        row = json.loads(path.read_text())
        row['learner_won'] = False
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'focal policy perspective'):
            self.run_evaluation()

    def test_resume_rejects_changed_sources_checkpoint_bytes_and_schedule(self):
        self.run_evaluation()
        with self.assertRaisesRegex(ValueError, 'changed on resume'):
            self.run_evaluation(source_manifest={'source': 'changed'})
        with self.assertRaisesRegex(ValueError, 'changed on resume'):
            self.run_evaluation(seed=1000001)
        self.paths[2].with_suffix('.manifest.json').write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, 'changed on resume'):
            self.run_evaluation()

    def test_live_focal_must_match_saved_weights_and_cannot_alias_opponent(self):
        with self.assertRaisesRegex(ValueError, 'saved checkpoint'):
            self.run_evaluation(model=FakeModel(9))
        with self.assertRaisesRegex(ValueError, 'separate model'):
            self.run_evaluation(opponent_model=self.model)

    def test_stale_inference_digest_fails_before_games(self):
        async def stale(model, version):
            await FakeCollector.sync_weights(self.collector, model, version)
            self.collector.policy_digest = 'incorrect'
        self.collector.sync_weights = stale
        with self.assertRaisesRegex(RuntimeError, 'stale policy weights'):
            self.run_evaluation()
        self.assertFalse(self.collector.seen)

    def test_source_changes_during_evaluation_cannot_publish_completion(self):
        with patch('patchwork_automaton.rl.hartford_evaluation._runtime_pins',
                   side_effect=[{'source': 'fixed'}, {'source': 'changed'}]):
            with self.assertRaisesRegex(RuntimeError, 'source or engine changed'):
                self.run_evaluation()
        summary = json.loads((self.root/'evaluation/summary.json').read_text())
        self.assertIs(summary['complete'], False)

    def test_errors_preserved_without_successful_game_or_complete_summary(self):
        self.collector.corrupt = dict(status='cutoff')
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            self.run_evaluation()
        self.assertEqual(len(list((self.root/'evaluation').glob('opponent-*/failures/game-*.json'))), 1)
        self.assertFalse(list((self.root/'evaluation').glob('opponent-*/games/game-*.json')))
        self.assertFalse((self.root/'evaluation/summary.json').exists())
        self.collector.corrupt = None
        with self.assertRaisesRegex(RuntimeError, 'Preserved failed evaluation attempt'):
            self.run_evaluation()
        self.assertEqual(len(self.collector.seen), 1)

    def test_completed_evaluation_cannot_replace_missing_rows_or_trust_changed_summary(self):
        self.run_evaluation()
        summary_path = self.root/'evaluation/summary.json'
        summary = json.loads(summary_path.read_text())
        changed = dict(summary, score=0.)
        summary_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'summary differs'):
            self.run_evaluation()
        self.assertEqual(len(self.collector.seen), 1200)
        summary_path.write_text(json.dumps(summary))
        path = next((self.root/'evaluation').glob('opponent-*/games/game-*.json'))
        original = path.read_text()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing durable'):
            self.run_evaluation()
        path.write_text(original)
        row = json.loads(original)
        row.update(successful_row(row, 'loss'))
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'Published opponent results changed'):
            self.run_evaluation()
        self.assertEqual(len(self.collector.seen), 1200)

    def test_initial_checkpoint_uses_one_distinct_predecessor(self):
        result = self.run_evaluation(model=FakeModel(1), checkpoint_round=1)
        self.assertEqual(result['opponent_rounds'], [0])
        self.assertEqual(result['aggregate']['games'], 400)
        with self.assertRaisesRegex(ValueError, 'no previous'):
            self.run_evaluation(model=FakeModel(0), checkpoint_round=0)

    def test_focal_perspective_draws_adjudications_and_intervals(self):
        planned = evaluation_schedule(DECKS, checkpoint_round=3, opponent_round=2, games_per_cell=4, seed=7)
        game = next(row for row in planned if row['learner_seat'] == 1)
        won = successful_row(game)
        self.assertEqual(won['status'], 'loss')  # Engine status uses seat zero.
        validate_row(won, game, checkpoint_round=3)
        draw = dict(successful_row(game, 'draw'), adjudicated=True, reason='wall_clock_timeout')
        lost = successful_row(game, 'loss')
        natural = successful_row(game, 'draw')
        summary = outcome_summary([won, draw, lost, natural])
        self.assertEqual((summary['wins'], summary['losses'], summary['draws']), (1, 1, 2))
        self.assertEqual(summary['score'], .5)
        self.assertEqual(summary['win_rate'], .25)
        self.assertEqual(summary['natural_draws'], 1)
        self.assertEqual(summary['adjudication_reasons'], {'wall_clock_timeout': 1})
        self.assertLess(summary['score_ci95'][0], summary['score'])
        self.assertGreater(summary['score_ci95'][1], summary['score'])
        with self.assertRaisesRegex(ValueError, 'training trajectories'):
            validate_row(dict(won, trajectories=[{}]), game, checkpoint_round=3)


if __name__ == '__main__':
    unittest.main()
