"""Independent full-state continuation and mixed scripted-game recovery."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from examples import train_hartford_ppo as driver
from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS
from patchwork_automaton.rl.hartford_ab import training_schedule
import test_hartford_driver as fixtures


class ForkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parent = self.root / 'parent/checkpoints/round-0100'
        self.parent.mkdir(parents=True)
        self.previous = driver.configuration(driver.parser().parse_args(['--skip-evaluations']))
        self.previous.update(architecture='compact-identity-standard-v2', use_identity=True, use_rules=False)
        for name in driver.CHECKPOINT_FILES:
            (self.parent / name).write_bytes(name.encode())
        state = dict(round=100, config=self.previous, optimizer_steps=1234, critic_optimizer_steps=15700,
            trainer_rng_state={'saved': 'exact'}, round_result={'round': 100})
        driver.atomic_json(self.parent / 'state.json', state)
        self.make_external_history()

    def make_external_history(self):
        checkpoints = {'100': driver.checkpoint_hashes(self.parent)}
        for n in (25, 50, 75, 97, 98, 99):
            path = self.parent.parent / f'round-{n:04d}'
            if path.exists():
                shutil.rmtree(path)
            shutil.copytree(self.parent, path)
            state = json.loads((path / 'state.json').read_text())
            state['round'] = n
            driver.atomic_json(path / 'state.json', state)
            checkpoints[str(n)] = driver.checkpoint_hashes(path)
        driver.atomic_json(self.parent.parent.parent / 'report.json',
            dict(status='complete', completed_games=100000, checkpoints=checkpoints))

    def arguments(self, recipe='current'):
        return driver.parser().parse_args(['--architecture', 'compact-identity',
            '--fork-from', str(self.parent), '--fork-from-sha256', driver.checkpoint_bundle_sha256(self.parent),
            '--opponent-recipe', recipe, '--rounds', '50', '--seed', '412610910000',
            '--skip-evaluations', '--output', str(self.root / recipe)])

    def test_child_zero_preserves_files_steps_rng_and_parent(self):
        before = driver.checkpoint_hashes(self.parent)
        for recipe in ('current', 'diverse'):
            args = self.arguments(recipe)
            config = driver.configuration(args)
            args.output.mkdir()
            result = driver.prepare_fork_checkpoint(args.output, config)
            cp = args.output / 'checkpoints/round-0000'
            for name, digest in before.items():
                if name != 'state.json':
                    self.assertEqual(driver.sha256(cp / name), digest)
            state = json.loads((cp / 'state.json').read_text())
            self.assertEqual(state['trainer_rng_state'], {'saved': 'exact'})
            self.assertEqual((state['optimizer_steps'], state['critic_optimizer_steps']), (1234, 15700))
            self.assertEqual((state['round'], state['lineage_checkpoint']), (0, 100))
            self.assertNotIn('round_result', state)
            self.assertEqual(driver.restore_history(result, cp.parent, config), 0)
            self.assertEqual(result['completed_games'], 0)
            self.assertEqual(driver.prepare_fork_checkpoint(args.output, config), result)
        self.assertEqual(driver.checkpoint_hashes(self.parent), before)

    def test_reject_changed_hash_recipe_seed_and_partial_parent(self):
        args = self.arguments()
        for key, value in [('fork_from_sha256', '0' * 64), ('rounds', 100),
                ('skip_evaluations', False), ('seed', self.previous['seed']), ('game_timeout', 60)]:
            modified = deepcopy(args)
            setattr(modified, key, value)
            with self.assertRaises(ValueError):
                driver.configuration(modified)
        report_path = self.parent.parent.parent / 'report.json'
        report = json.loads(report_path.read_text())
        report['status'] = 'running'
        driver.atomic_json(report_path, report)
        with self.assertRaisesRegex(ValueError, 'completed'):
            driver.configuration(args)

    def test_preparation_imports_no_neural_runtime_and_runs_no_training(self):
        args = self.arguments()
        args.prepare_fork_only = True
        with patch.object(driver, 'hartford_preflight', return_value={'ready': True}), \
                patch.object(driver.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 100 * 1024**3})()), \
                patch.object(driver, 'initialize_random', side_effect=AssertionError('must not initialize')):
            result = asyncio.run(driver.run(args))
        self.assertEqual(result['status'], 'prepared')
        self.assertFalse((args.output / 'rollouts').exists())

    def test_full_neural_optimizer_and_rng_restore(self):
        import numpy as np
        from patchwork_automaton.neural.compact_standard_model import CompactActorCritic
        from patchwork_automaton.neural.compact_identity_model import transfer_v1
        from patchwork_automaton.neural.experiment import make_trainer, save_checkpoint, load_checkpoint
        from patchwork_automaton.neural.standard_policy import StandardPolicy
        old = CompactActorCritic()
        old(old.example_inputs())
        model = transfer_v1(old)
        trainer = make_trainer(model, self.previous)
        trainer.compile()
        for optimizer in (trainer.optimizer, trainer.critic_optimizer):
            optimizer.iterations.assign(123)
            for i, variable in enumerate(optimizer.variables[2:], 2):
                variable.assign(np.full(variable.shape, i / 1000, dtype='float32'))
        trainer.rng.random(17)
        expected_rng = deepcopy(trainer.rng.bit_generator.state)
        shutil.rmtree(self.parent)
        save_checkpoint(self.parent.parent, model, trainer, dict(round=100, config=self.previous),
            policy_type=StandardPolicy)
        self.make_external_history()
        args = self.arguments()
        config = driver.configuration(args)
        args.output.mkdir()
        driver.prepare_fork_checkpoint(args.output, config)
        restored, resumed, state = load_checkpoint(args.output / 'checkpoints', policy_type=StandardPolicy)
        for original, actual in zip(model.get_weights(), restored.get_weights()):
            np.testing.assert_array_equal(original, actual)
        for original, actual in ((trainer.optimizer, resumed.optimizer),
                (trainer.critic_optimizer, resumed.critic_optimizer)):
            self.assertEqual(len(original.variables), len(actual.variables))
            for first, second in zip(original.variables, actual.variables):
                np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        self.assertEqual(resumed.rng.bit_generator.state, expected_rng)
        self.assertEqual(state['round'], 0)


class MixedScriptRecoveryTests(unittest.TestCase):
    def test_mixed_collector_routes_script_without_neural_version(self):
        from concurrent.futures import Future
        from types import SimpleNamespace
        from patchwork_automaton.rl import standard_selfplay_processes as transport
        async def refresh():
            return {}
        collector = object.__new__(transport.StandardSelfplayCollector)
        collector.closed = collector.broken = collector.active = False
        collector.replicas = (SimpleNamespace(started=True, version=0, refresh_stats=refresh),
            SimpleNamespace(started=True, version=0, version_digests={0: 'weights'}, refresh_stats=refresh))
        jobs = [dict(game=dict(opponent_kind='scripted', opponent_version=None)),
            dict(game=dict(opponent_kind='frozen_neural', opponent_version=0,
                opponent_checkpoint_sha256='a' * 64), opponent_weights_sha256='weights')]
        class ImmediateExecutor:
            def submit(self, function, *args):
                future = Future()
                future.set_result(function(*args))
                return future
        with patch.object(transport, '_run_game', lambda options: options):
            collector.pool = ImmediateExecutor()
            rows = asyncio.run(collector.run_schedule(jobs, game_runner='standard_ppo'))
            self.assertNotIn('opponent_version', rows[0])
            self.assertNotIn('opponent_sha256', rows[0])
            self.assertEqual(rows[1]['opponent_version'], 0)
            self.assertEqual(rows[1]['opponent_sha256'], 'a' * 64)
            forged = deepcopy(jobs[:1])
            forged[0]['opponent_weights_sha256'] = 'weights'
            with self.assertRaisesRegex(ValueError, 'must not claim'):
                asyncio.run(collector.run_schedule(forged, game_runner='standard_ppo'))

    def test_resume_reuses_neural_and_scripted_results_without_replaying(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            planned = training_schedule(tuple(HARTFORD_DECKS), round_id=1, seed=412610910000, recipe='diverse')
            for game in planned:
                if game['opponent_kind'] == 'frozen_neural':
                    game['opponent_checkpoint_sha256'] = 'a' * 64
            recorder = fixtures.Recorder()
            calls = []
            async def pin(numbers):
                calls.append(numbers)
                self.assertNotIn(None, numbers)
                return {n: str(n) for n in numbers}
            def run():
                return asyncio.run(driver.collect_round(planned=planned, number=1, directory=root,
                    collectors=[recorder], pin_opponent=None, pin_opponents=pin,
                    progress=lambda *a, **kw: None, game_timeout=20))
            recorder.fail_after = 37
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                run()
            preserved = {p: driver.sha256(p) for p in (root / 'results').glob('*.json')}
            recorder.fail_after = None
            rows = run()
            self.assertEqual(len(rows), 1000)
            self.assertEqual(len(recorder.seen), len(set(recorder.seen)))
            self.assertEqual(sum(r['opponent_kind'] == 'scripted' for r in rows), 120)
            self.assertEqual(preserved, {p: driver.sha256(p) for p in preserved})
            self.assertEqual(calls, [[0, 1025, 1050, 1075]] * 2)
            run()
            self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
