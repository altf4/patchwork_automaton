"""A branch may change opponents but must retain the PPO recipe and parent bytes."""
import tempfile
import json
from pathlib import Path
import unittest

from examples.train_standard_ppo import (
    CHECKPOINT_FILES, assert_fork_recipe, checkpoint_hashes, parser, resolve_inference_processes,
    validate_parent_copy)


class SelfplayDriverTests(unittest.TestCase):
    def test_fork_recipe_rejects_learning_changes(self):
        parent = dict(algorithm='ppo', learning_rate=.0001, seed=42,
                      milestones=[10, 20, 200], workers=48)
        branch = dict(parent, milestones=[], opponent_mode='frozen_pool',
                      opponent_script_fraction=.2, opponent_inference_processes=2,
                      opponent_snapshot_schedule='round_robin_150_200_plus_every_tenth_committed')
        assert_fork_recipe(parent, branch)
        branch['learning_rate'] = .0002
        with self.assertRaisesRegex(ValueError, 'learning_rate'):
            assert_fork_recipe(parent, branch)


    def test_replica_defaults_and_existing_resume_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            common = ['--output', str(root)]
            selfplay = parser().parse_args(common + ['--opponent-mode', 'frozen_pool'])
            scripted = parser().parse_args(common + ['--opponent-mode', 'legacy'])
            explicit = parser().parse_args(common + ['--opponent-mode', 'frozen_pool',
                                                       '--inference-processes', '6'])
            self.assertEqual(resolve_inference_processes(selfplay), 2)
            self.assertEqual(resolve_inference_processes(scripted), 4)
            self.assertEqual(resolve_inference_processes(explicit), 6)
            (root/'report.json').write_text(json.dumps({'config': {'inference_processes': 4}}))
            resumed = parser().parse_args(common + ['--opponent-mode', 'frozen_pool', '--resume'])
            self.assertEqual(resolve_inference_processes(resumed), 4)
            resumed.inference_processes = 2
            self.assertEqual(resolve_inference_processes(resumed), 2)

    def test_selfplay_fork_may_change_runtime_replicas_only(self):
        parent = dict(algorithm='ppo', milestones=[200], inference_processes=4, learning_rate=.0001)
        branch = dict(parent, milestones=[], inference_processes=2, opponent_mode='frozen_pool')
        assert_fork_recipe(parent, branch)
        with self.assertRaisesRegex(ValueError, 'inference_processes'):
            assert_fork_recipe(parent, dict(branch, opponent_mode='scripted_control'))
        with self.assertRaisesRegex(ValueError, 'learning_rate'):
            assert_fork_recipe(parent, dict(branch, learning_rate=.001))

    def test_parent_copy_integrity_and_branch_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent = root/'parent'/'round-0200'
            copied = root/'branch'/'checkpoints'/'round-0200'
            parent.mkdir(parents=True)
            copied.mkdir(parents=True)
            for name in CHECKPOINT_FILES:
                (parent/name).write_bytes(name.encode())
                (copied/name).write_bytes(name.encode())
            fork = dict(parent_checkpoint=str(parent),
                        parent_checkpoint_sha256=checkpoint_hashes(parent))
            validate_parent_copy(fork, copied.parent)
            (copied/'optimizer.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed after fork'):
                validate_parent_copy(fork, copied.parent)
            args = parser().parse_args(['--output',str(root/'output'),
                '--fork-checkpoint',str(parent),'--opponent-mode','frozen_pool',
                '--rounds','250','--milestones'])
            self.assertIsNone(args.source)
            self.assertEqual(args.milestones, [])
            self.assertEqual(args.rounds, 250)


if __name__ == '__main__':
    unittest.main()
