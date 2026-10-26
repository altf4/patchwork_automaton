"""Normal training adopts runtime caches without changing the learning recipe."""
import asyncio
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE, 'Requires neural dependencies')
class UpdatePipelineAdoptionTests(unittest.TestCase):
    def test_capacity_is_resolved_once_before_gpu_runtime_initialization(self):
        from examples import train_rustmage as runner
        args = runner.parser().parse_args(['--output', 'unused', '--learner-device', 'gpu'])
        events = []
        settings = {'resolved_profile': 'gpu'}
        def resolve(*args, **kwargs):
            events.append('resolve')
            return settings
        def configure(*args, **kwargs):
            events.append('configure')
        async def experiment(args, *, update_pipeline):
            self.assertIs(update_pipeline, settings)
            events.append('experiment')
        with patch.object(runner, 'resolve_update_pipeline', side_effect=resolve) as resolver, \
                patch.object(runner, 'configure_runtime', side_effect=configure), \
                patch.object(runner, 'device_scope', return_value=nullcontext()), \
                patch.object(runner, 'run_experiment', side_effect=experiment):
            asyncio.run(runner.run(args))
        resolver.assert_called_once()
        self.assertEqual(events, ['resolve', 'configure', 'experiment'])

    def test_fresh_training_enables_host_packing_before_updates(self):
        from examples import train_rustmage as runner
        from test_critic_warmup import CriticWarmupTests

        class ReachedRuntime(Exception): pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, _, _ = CriticWarmupTests().fixture(root/'rollouts')
            args = runner.parser().parse_args(['--output', str(root/'new')])

            def check(trainer, device):
                self.assertEqual(device, 'cpu')
                self.assertTrue(trainer.numerical_prepacking)
                self.assertTrue(trainer.cache_policy_batches)
                self.assertIsNone(trainer.policy_cache_device)
                self.assertIsNone(trainer.actor_epoch_cache_device)
                self.assertEqual(int(trainer.optimizer.iterations), 0)
                raise ReachedRuntime

            with patch.object(runner, 'initialize_model', return_value=(model, {})), \
                    patch.object(runner, 'save_checkpoint') as save, \
                    patch.object(runner, 'place_trainer', side_effect=check):
                with self.assertRaises(ReachedRuntime):
                    asyncio.run(runner.run_experiment(args))
            saved = save.call_args.args[3]['config']
            self.assertTrue(runner.RUNTIME_ARGUMENTS.isdisjoint(saved))
            self.assertEqual(saved['train_batch'], args.train_batch)
            self.assertEqual(saved['learning_rate'], args.learning_rate)

    def test_old_checkpoint_adopts_host_or_gpu_profile_without_state_changes(self):
        import numpy as np
        from examples import train_rustmage as runner
        from patchwork_automaton.neural.experiment import save_checkpoint
        from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
        from test_critic_warmup import CriticWarmupTests

        class ReachedRuntime(Exception): pass

        for device, profile in [('cpu', 'host'), ('gpu', 'gpu')]:
            with self.subTest(device=device), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model, trainer, _ = CriticWarmupTests().fixture(root/'rollouts')
                args = runner.parser().parse_args(['--output', str(root/'study'), '--resume',
                    '--learner-device', device, '--learner-gru-backend', 'unfused',
                    '--update-pipeline', profile, '--policy-cache-mib', '64',
                    '--actor-epoch-cache-mib', '32'])
                config = {k:v for k,v in vars(args).items() if k not in
                    runner.RUNTIME_ARGUMENTS | {'output','resume','rounds','plot','initialize_from'}}
                config.update(inference_processes=2, algorithm='ppo', advantage_scale_floor=1., ppo_clip=.2)
                save_checkpoint(args.output/'checkpoints', model, trainer,
                    dict(round=0, config=config, initial_report=dict(rounds=[], evaluations=[])),
                    policy_type=RustMagePolicy)
                checkpoint_root = args.output/'checkpoints'
                before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in checkpoint_root.rglob('*') if p.is_file()}

                def check(restored, selected_device):
                    self.assertEqual(selected_device, device)
                    self.assertTrue(restored.numerical_prepacking)
                    self.assertTrue(restored.cache_policy_batches)
                    self.assertEqual(restored.policy_cache_device, '/GPU:0' if device == 'gpu' else None)
                    self.assertEqual(restored.actor_epoch_cache_device, '/GPU:0' if device == 'gpu' else None)
                    self.assertEqual(restored.policy_cache_device_bytes, 64*1024**2 if device == 'gpu' else 0)
                    self.assertEqual(restored.actor_epoch_cache_bytes, 32*1024**2 if device == 'gpu' else 0)
                    for original, actual in [(model.weights, restored.model.weights),
                            (trainer.optimizer.variables, restored.optimizer.variables),
                            (trainer.critic_optimizer.variables, restored.critic_optimizer.variables)]:
                        self.assertEqual(len(original), len(actual))
                        for left, right in zip(original, actual):
                            np.testing.assert_array_equal(left, right)
                    self.assertEqual(trainer.rng.bit_generator.state, restored.rng.bit_generator.state)
                    if device == 'gpu':
                        raise ReachedRuntime
                    return {}

                # The GPU case checks configuration, never allocating GPU tensors.
                with patch.object(runner, 'assert_model_device', return_value={}), \
                        patch.object(runner, 'place_trainer', side_effect=check), \
                        patch.object(runner, 'GameValidator', side_effect=ReachedRuntime), \
                        patch.object(runner, 'initialize_model') as initialize:
                    with self.assertRaises(ReachedRuntime):
                        asyncio.run(runner.run_experiment(args))
                    initialize.assert_not_called()
                self.assertEqual(before, {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in checkpoint_root.rglob('*') if p.is_file()})
                if device == 'cpu':
                    report = json.loads((args.output/'report.json').read_text())
                    invocation = report['invocations'][-1]
                    self.assertEqual(invocation['update_pipeline']['resolved_profile'], 'host')
                    self.assertIn('src/patchwork_automaton/neural/update_pipeline.py',
                                  invocation['source_sha256'])
                    self.assertTrue(runner.RUNTIME_ARGUMENTS.isdisjoint(report['config']))


if __name__ == '__main__':
    unittest.main()
