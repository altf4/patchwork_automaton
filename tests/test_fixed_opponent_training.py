"""Fixed-opponent PPO records only the neural seat's genuine terminal game."""
import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_neural import AVAILABLE

NATIVE = Path('.local/rustmage/target/release/librustmage.so').is_file()


@unittest.skipUnless(AVAILABLE, 'Requires neural dependencies')
class FixedOpponentConfigurationTests(unittest.TestCase):
    @staticmethod
    def saved_config(args, *, legacy=False):
        config = {key: value for key, value in vars(args).items() if key not in
                  ('output', 'resume', 'rounds', 'plot', 'initialize_from')}
        config.update(inference_processes=2, algorithm='ppo',
                      advantage_scale_floor=1., ppo_clip=.2)
        if legacy:
            config.pop('train_mode')
        return config

    def test_default_training_mode_remains_selfplay(self):
        from examples.train_rustmage import parser
        self.assertEqual(parser().parse_args(['--output', 'unused']).train_mode, 'selfplay')
        fixed = parser().parse_args(['--output', 'unused', '--train-mode', 'versus_baseline'])
        self.assertEqual(fixed.train_mode, 'versus_baseline')

    def test_legacy_resume_accepts_selfplay_and_rejects_opponent_switch(self):
        from examples import train_rustmage as runner
        args = runner.parser().parse_args(['--output', 'unused', '--resume'])
        state = dict(round=0, config=self.saved_config(args, legacy=True),
                     initial_report=dict(rounds=[], evaluations=[]))

        class ReachedRuntime(Exception):
            pass

        with patch.object(runner, 'load_checkpoint', return_value=(None, None, state)), \
                patch.object(runner, 'assert_model_device', side_effect=ReachedRuntime):
            with self.assertRaises(ReachedRuntime):
                asyncio.run(runner.run_experiment(args))
        args.train_mode = 'versus_baseline'
        with patch.object(runner, 'load_checkpoint', return_value=(None, None, state)):
            with self.assertRaisesRegex(ValueError, 'same experiment configuration'):
                asyncio.run(runner.run_experiment(args))

    def test_fixed_resume_refreshes_copied_report_opponent_metadata(self):
        from examples import train_rustmage as runner
        args = runner.parser().parse_args(['--output', 'unused', '--resume',
                                          '--train-mode', 'versus_baseline'])
        report = dict(rounds=[], evaluations=[], training='old self-play description')
        state = dict(round=0, config=self.saved_config(args), initial_report=report)

        class ReachedRuntime(Exception):
            pass

        with patch.object(runner, 'load_checkpoint', return_value=(None, None, state)), \
                patch.object(runner, 'assert_model_device', side_effect=ReachedRuntime):
            with self.assertRaises(ReachedRuntime):
                asyncio.run(runner.run_experiment(args))
        self.assertIn('active FoundationsAggro', report['training'])
        self.assertEqual(report['training_opponent']['recorded_players'], 'learner seat only')
        self.assertEqual(report['training_opponent']['critic_warmup_mode'], 'selfplay')
        self.assertEqual(report['config']['train_mode'], 'versus_baseline')


@unittest.skipUnless(AVAILABLE and NATIVE, 'Requires neural dependencies and RustMage')
class FixedOpponentNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from patchwork_automaton.rl import rustmage_experiment
        from patchwork_automaton.neural.demonstrations import read_demonstration
        from patchwork_automaton.neural.rustmage_features import RustMageEncoder
        cls.collector = rustmage_experiment
        cls.fixtures = tempfile.TemporaryDirectory()
        cls.records = {}
        # The actual scripted policy supplies a deterministic inference fixture.
        # Replaying its encoded decisions still exercises the neural collector,
        # native legality, both private viewers, and real terminal outcomes.
        for index in (0, 1):
            row = cls.collector.play_game(seed=721, index=index, mode='demonstrations',
                                         output=Path(cls.fixtures.name)/str(index))
            if row['status'] not in ('win', 'loss', 'draw'):
                raise AssertionError(row)
            cls.records[index] = {
                f'P{player}': read_demonstration(trajectory)
                for player, trajectory in enumerate(row['trajectories'])}
        cls.model = SimpleNamespace(memory_size=8,
                                    feature_spec=RustMageEncoder().feature_spec)

    @classmethod
    def tearDownClass(cls):
        cls.fixtures.cleanup()

    def inference(self, index):
        import numpy as np
        records = self.records[index]

        class Replay:
            def __init__(self):
                self.offsets = dict(P0=0, P1=0)

            def predict(self, batch, *, identity, **kwargs):
                player = identity[-1].value
                sample = records[player][self.offsets[player]]
                self.offsets[player] += 1
                for key, value in sample.arrays.items():
                    np.testing.assert_array_equal(batch[key], value, err_msg=key)
                logits = np.full((1, len(sample.arrays['candidate_mask'])), -1000., np.float32)
                logits[0, sample.action] = 0.
                return dict(logits=logits, value=np.zeros((1, 1), np.float32),
                            memory_out=batch['memory_in'][None, :]+1.)

        return Replay()

    def test_each_seat_records_only_learner_with_real_terminal_reward(self):
        import numpy as np
        from patchwork_automaton.neural.experiment import read_trajectory
        from patchwork_automaton.neural.rustmage_features import ACTIONS
        from patchwork_automaton.neural.validation import split_games
        with tempfile.TemporaryDirectory() as temporary:
            rows = []
            for seat in (0, 1):
                with self.subTest(seat=seat):
                    output = Path(temporary)/str(seat)
                    row = self.collector.play_game(seed=721, index=seat, mode='versus_baseline',
                        output=output, model=self.model, batcher=self.inference(seat))
                    self.assertIn(row['status'], ('win', 'loss', 'draw'), row.get('error'))
                    self.assertEqual(row['seat'], seat)
                    self.assertEqual(len(row['trajectories']), 1)
                    trajectory = row['trajectories'][0]
                    self.assertEqual(trajectory['player'], seat)
                    self.assertEqual(trajectory['reward'], row['terminal_rewards'][seat])
                    expected_reward = 0. if row['status'] == 'draw' else 1. if row['status'] == 'win' else -1.
                    self.assertEqual(trajectory['reward'], expected_reward)
                    self.assertEqual(Path(trajectory['path']).name, f'game-{seat:05d}-p{seat}.pkl.gz')
                    self.assertEqual(len(list(output.glob('*.pkl.gz'))), 1)
                    samples = list(read_trajectory(trajectory['path']))
                    self.assertEqual(len(samples), trajectory['samples'])
                    self.assertGreater(len(samples), 0)
                    concede = list(ACTIONS).index('Concede')
                    self.assertTrue(all(not np.any(sample.arrays['candidates'][:, concede]) for sample in samples))
                    rows.append(row)
            train, heldout, partition = split_games(rows, fraction=.2, seed=9001, round_id=1)
            self.assertEqual((len(train), len(heldout)), (1, 1))
            self.assertFalse(set(partition['training_indices']) & set(partition['validation_indices']))
            self.assertEqual(sum(len(game['trajectories']) for game in train+heldout), 2)

    def test_evaluation_output_none_never_opens_trajectory_writer(self):
        with patch('patchwork_automaton.neural.experiment.TrajectoryWriter',
                   side_effect=AssertionError('Evaluation must never record')) as writer:
            row = self.collector.play_game(seed=721, index=0, mode='versus_baseline', output=None,
                                          model=self.model, batcher=self.inference(0))
        self.assertIn(row['status'], ('win', 'loss', 'draw'), row.get('error'))
        self.assertEqual(row['trajectories'], [])
        writer.assert_not_called()

    def test_cutoff_and_inference_error_have_no_training_returns(self):
        with tempfile.TemporaryDirectory() as temporary:
            cutoff = self.collector.play_game(seed=721, index=0, mode='versus_baseline',
                output=Path(temporary)/'cutoff', model=self.model,
                batcher=self.inference(0), max_decisions=1)
            self.assertEqual(cutoff['status'], 'cutoff')
            self.assertEqual(cutoff['trajectories'], [])
            self.assertNotIn('terminal_rewards', cutoff)

            class Broken:
                def predict(self, *args, **kwargs):
                    raise RuntimeError('Deliberate inference failure')

            failed = self.collector.play_game(seed=721, index=0, mode='versus_baseline',
                output=Path(temporary)/'error', model=self.model, batcher=Broken())
            self.assertEqual(failed['status'], 'error')
            self.assertEqual(failed['trajectories'], [])
            self.assertNotIn('terminal_rewards', failed)


if __name__ == '__main__':
    unittest.main()
