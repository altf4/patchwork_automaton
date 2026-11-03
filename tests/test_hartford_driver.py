"""Admission, frozen-opponent collection and checkpoint transaction recovery."""
import asyncio
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from examples.train_hartford_ppo import (CHECKPOINT_FILES, atomic_json, checkpoint_hashes,
    collect_round, configuration, parser, read_collection, restore_history, run, settle_all)
from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS
from patchwork_automaton.rl.hartford_schedule import training_schedule, evaluation_game_count


class Recorder:
    """Test transport emits known trajectory bytes, never runs the game engine."""
    def __init__(self):
        self.seen = []
        self.options = []
        self.fail_after = None

    async def run_schedule(self, jobs, *, progress, **options):
        self.options.append(options)
        for job in jobs:
            if self.fail_after is not None and len(self.seen) == self.fail_after:
                raise RuntimeError('transport interrupted')
            self.seen.append(job['index'])
            game = job['game']
            path = options['output'] / f"game-{job['index']}.pkl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"trajectory {job['index']}".encode())
            progress(dict(game, status='win', winner_seat=game['learner_seat'],
                trajectories=[dict(path=str(path), samples=1, player=game['learner_seat'],
                    deck=game['learner_deck'], reward=1., value_sum=0., value_square_sum=0.)]))


class HartfordDriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = parser().parse_args(['--output', str(self.root / 'run')])
        self.config = configuration(self.args)

    def test_exact_default_recipe_and_no_warm_start_options(self):
        self.assertEqual((self.config['rounds'], self.config['games_per_round']), (100, 1000))
        self.assertEqual(self.config['eval_games_per_cell'], 40)
        self.assertEqual(evaluation_game_count(100, 40), 1_188_000)
        self.assertEqual(self.config['initialization'], 'random_actor_and_critic')
        self.assertEqual(self.config['action_selection'], 'sampled')
        self.assertFalse(hasattr(self.args, 'source'))
        self.assertFalse(hasattr(self.args, 'fork_checkpoint'))

    def test_invalid_recipe_and_seed_overlap_rejected(self):
        for change in (dict(eval_games=41), dict(games_per_round=999), dict(rounds=0),
                       dict(runners=3), dict(update_timeout=float('nan')),
                       dict(eval_seed=self.args.seed), dict(weight_seed=2**32)):
            args = deepcopy(self.args)
            for key, value in change.items():
                setattr(args, key, value)
            with self.assertRaises(ValueError):
                configuration(args)

    def test_preflight_gate_runs_before_any_output_or_model_initialization(self):
        report = dict(ready=False, errors=['unsupported test card'])
        with patch('examples.train_hartford_ppo.hartford_preflight', return_value=report), \
             patch('examples.train_hartford_ppo.initialize_random') as initialize:
            with self.assertRaisesRegex(ValueError, 'unsupported test card'):
                asyncio.run(run(self.args))
            initialize.assert_not_called()
        self.assertFalse(self.args.output.exists())

    def test_real_cli_preflight_imports_no_tensorflow_and_creates_no_run(self):
        script = (
            'import asyncio, sys; from examples.train_hartford_ppo import parser, run; '
            f'args=parser().parse_args(["--preflight-only","--output",{str(self.args.output)!r}]); '
            'result=asyncio.run(run(args)); '
            'assert isinstance(result["ready"], bool); '
            'assert not any(n.startswith("tensorflow") for n in sys.modules); '
            'assert not args.output.exists()')
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('unimplemented_mainboard_cards', result.stdout)

    def planned(self, number=4):
        return [dict(game, opponent_checkpoint_sha256=str(game['opponent_version']) * 64)
                for game in training_schedule(tuple(HARTFORD_DECKS), round_id=number, seed=100)]

    def collect(self, recorder, planned=None):
        self.pinned = []
        async def pin(number):
            self.pinned.append(number)
        return asyncio.run(collect_round(planned=planned or self.planned(), number=4,
            directory=self.root / 'rollouts', collectors=[recorder], pin_opponent=pin,
            progress=lambda *args, **kwargs: None))

    def test_collection_pins_four_opponents_and_never_uses_greedy(self):
        recorder = Recorder()
        rows = self.collect(recorder)
        self.assertEqual(self.pinned, [3, 2, 1, 0])
        self.assertEqual(Counter(r['opponent_version'] for r in rows), {3: 700, 2: 100, 1: 100, 0: 100})
        self.assertEqual(Counter(r['partition'] for r in rows), {'training': 800, 'validation': 200})
        for option in recorder.options:
            self.assertIs(option['greedy'], False)
            self.assertEqual(option['version'], 3)
        self.assertEqual(len(read_collection(self.root / 'rollouts', self.planned(), 3)), 1000)
        self.collect(recorder)
        self.assertEqual(len(recorder.seen), 1000)
        self.assertFalse(self.pinned)

    def test_interrupted_collection_reuses_completed_games(self):
        recorder = Recorder()
        recorder.fail_after = 13
        with self.assertRaisesRegex(RuntimeError, 'transport interrupted'):
            self.collect(recorder)
        recorder.fail_after = None
        rows = self.collect(recorder)
        self.assertEqual(len(rows), 1000)
        self.assertEqual(len(set(recorder.seen)), 1000)
        self.assertEqual(len(recorder.seen), 1000)
        damaged = Path(rows[0]['trajectories'][0]['path'])
        damaged.write_bytes(b'damaged')
        with self.assertRaisesRegex(ValueError, 'trajectory changed'):
            self.collect(recorder)

    def make_checkpoint(self, number):
        path = self.root / 'checkpoints' / f'round-{number:04d}'
        path.mkdir(parents=True)
        for name in CHECKPOINT_FILES:
            (path / name).write_text(f'{number}:{name}')
        atomic_json(path / 'state.json', dict(round=number, config=self.config,
                                             round_result=dict(round=number, training_games=800)))
        return path

    def test_checkpoint_commit_recovers_missing_report_and_latest_publication(self):
        zero = self.make_checkpoint(0)
        report = dict(rounds=[], checkpoints={'0': checkpoint_hashes(zero)})
        atomic_json(zero.parent / 'latest.json', dict(checkpoint=zero.name))
        one = self.make_checkpoint(1)
        self.assertEqual(restore_history(report, zero.parent, self.config), 1)
        self.assertEqual(report['rounds'], [dict(round=1, training_games=800)])
        self.assertEqual(report['checkpoints']['1'], checkpoint_hashes(one))
        self.assertEqual(json.loads((zero.parent / 'latest.json').read_text()), dict(checkpoint=one.name))
        self.assertEqual(restore_history(report, zero.parent, self.config), 1)
        self.assertEqual(len(report['rounds']), 1)
        (zero / 'optimizer.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Checkpoint 0 changed'):
            restore_history(report, zero.parent, self.config)

    def test_initial_checkpoint_recovers_without_report(self):
        zero = self.make_checkpoint(0)
        report = dict(rounds=[], checkpoints={})
        self.assertEqual(restore_history(report, zero.parent, self.config), 0)
        self.assertEqual(report['checkpoints']['0'], checkpoint_hashes(zero))

    def test_resume_restores_cleanup_marker_and_rejects_changed_round_report(self):
        zero = self.make_checkpoint(0)
        self.make_checkpoint(1)
        atomic_json(zero.parent / 'latest.json', dict(checkpoint=zero.name))
        rollout = self.root / 'rollouts/round-0001'
        rollout.mkdir(parents=True)
        report = dict(rounds=[], checkpoints={'0': checkpoint_hashes(zero)})
        restore_history(report, zero.parent, self.config)
        self.assertTrue((rollout / 'committed.json').is_file())
        report['rounds'][0]['training_games'] = 1
        with self.assertRaisesRegex(ValueError, 'Round report differs'):
            restore_history(report, zero.parent, self.config)

    def test_resume_rejects_changed_schedule_and_noncontiguous_checkpoints(self):
        self.collect(Recorder())
        changed = self.planned()
        changed[0]['opponent_version'] = 100
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            self.collect(Recorder(), changed)
        zero = self.make_checkpoint(0)
        self.make_checkpoint(2)
        with self.assertRaisesRegex(ValueError, 'not contiguous'):
            restore_history(dict(rounds=[], checkpoints={}), zero.parent, self.config)

    def test_sibling_collectors_settle_before_error_propagates(self):
        finished = []
        async def failure():
            raise RuntimeError('first runner failed')
        async def sibling():
            await asyncio.sleep(.01)
            finished.append(True)
        with self.assertRaisesRegex(RuntimeError, 'first runner failed'):
            asyncio.run(settle_all([failure(), sibling()]))
        self.assertEqual(finished, [True])


if __name__ == '__main__':
    unittest.main()
