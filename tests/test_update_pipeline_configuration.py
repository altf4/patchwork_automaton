"""Portable profile admission must never require importing TensorFlow."""
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import patch

from patchwork_automaton.neural import update_pipeline as pipeline


class UpdatePipelineConfigurationTests(unittest.TestCase):
    uuid = 'GPU-12345678-1234-1234-1234-123456789abc'

    def inventory(self, total=24564, free=24000, *, output=None, visible=None):
        result = SimpleNamespace(stdout=output if output is not None else f'{self.uuid}, {total}, {free}\n')
        environment = {} if visible is None else {'CUDA_VISIBLE_DEVICES': visible}
        return (patch.dict(os.environ, environment, clear=True),
                patch.object(pipeline.shutil, 'which', return_value='/bin/nvidia-smi'),
                patch.object(pipeline.subprocess, 'run', return_value=result))

    def resolve_gpu(self, **kwargs):
        environment, which, run = self.inventory(**kwargs)
        with environment, which, run:
            return pipeline.resolve_update_pipeline('gpu')

    def test_cpu_and_explicit_profiles_do_not_probe(self):
        with patch.object(pipeline, '_gpu_inventory', side_effect=AssertionError('unexpected probe')):
            for profile, device, resolved in [('auto', 'cpu', 'host'), ('host', 'gpu', 'host'),
                                               ('legacy', 'gpu', 'legacy'), ('gpu', 'gpu', 'gpu')]:
                with self.subTest(profile=profile, device=device):
                    settings = pipeline.resolve_update_pipeline(device, profile)
                    self.assertEqual(settings['resolved_profile'], resolved)
                    self.assertEqual(settings['numerical_prepacking'], resolved != 'legacy')
                    self.assertEqual(settings['cache_policy_batches'], resolved != 'legacy')
                    self.assertIsNone(settings['gpu_memory'])
                    if resolved == 'gpu':
                        self.assertEqual(settings['policy_cache_device'], '/GPU:0')
                        self.assertEqual(settings['policy_cache_device_bytes'], 8192 * 1024**2)
                    else:
                        self.assertIsNone(settings['policy_cache_device'])
                        self.assertEqual(settings['policy_cache_device_bytes'], 0)
                        self.assertIsNone(settings['actor_epoch_cache_device'])
                        self.assertEqual(settings['actor_epoch_cache_bytes'], 0)

    def test_single_large_idle_gpu_uses_bounded_device_caches(self):
        for visible in (None, '0', self.uuid):
            with self.subTest(visible=visible):
                settings = self.resolve_gpu(visible=visible)
                self.assertEqual(settings['requested_profile'], 'auto')
                self.assertEqual(settings['resolved_profile'], 'gpu')
                self.assertEqual(settings['gpu_memory']['total_mib'], 24564)
                self.assertEqual(settings['gpu_memory']['free_mib'], 24000)
                self.assertEqual(settings['actor_epoch_cache_bytes'], 8 * 1024**3)
                self.assertEqual(settings['actor_epoch_cache_device'], '/GPU:0')

    def test_small_occupied_multiple_unknown_and_ambiguous_visibility_fall_back(self):
        cases = [dict(total=16384, free=16000), dict(free=22000),
                 dict(output=f'{self.uuid}, 24564, 24000\n{self.uuid}, 24564, 24000\n'),
                 dict(output=''), dict(output=f'{self.uuid}, N/A, N/A\n'),
                 dict(output=f'{self.uuid}, 24564, 25000\n'),
                 dict(output='not-a-uuid, 24564, 24000\n')]
        cases += [dict(visible=value) for value in ('', '-1', '1', '0,1', 'GPU-0000', 'MIG-123', ' 0 ')]
        for case in cases:
            with self.subTest(case=case):
                settings = self.resolve_gpu(**case)
                self.assertEqual(settings['resolved_profile'], 'host')
                self.assertTrue(settings['reason'])
                self.assertEqual(settings['policy_cache_device_bytes'], 0)
                self.assertEqual(settings['actor_epoch_cache_bytes'], 0)

    def test_unavailable_query_timeout_and_wsl_fallback(self):
        with patch.object(pipeline.shutil, 'which', return_value=None), \
                patch.object(Path, 'is_file', return_value=False):
            settings = pipeline.resolve_update_pipeline('gpu')
            self.assertEqual(settings['resolved_profile'], 'host')
            self.assertIn('unavailable', settings['reason'])
        for error in (subprocess.TimeoutExpired('nvidia-smi', 5),
                      subprocess.CalledProcessError(1, 'nvidia-smi'), OSError('unavailable')):
            with self.subTest(error=type(error).__name__), \
                    patch.object(pipeline.shutil, 'which', return_value='/bin/nvidia-smi'), \
                    patch.object(pipeline.subprocess, 'run', side_effect=error):
                self.assertEqual(pipeline.resolve_update_pipeline('gpu')['resolved_profile'], 'host')
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(pipeline.shutil, 'which', return_value=None), \
                patch.object(Path, 'is_file', return_value=True), \
                patch.object(pipeline.os, 'access', return_value=True), \
                patch.object(pipeline.subprocess, 'run', return_value=SimpleNamespace(
                    stdout=f'{self.uuid}, 24564, 24000\n')) as run:
            self.assertEqual(pipeline.resolve_update_pipeline('gpu')['resolved_profile'], 'gpu')
            self.assertEqual(run.call_args.args[0][0], '/usr/lib/wsl/lib/nvidia-smi')
            self.assertEqual(run.call_args.kwargs['timeout'], 5)
            self.assertTrue(run.call_args.kwargs['check'])

    def test_thresholds_and_zero_budgets(self):
        settings = self.resolve_gpu(total=23 * 1024, free=22 * 1024)
        self.assertEqual(settings['resolved_profile'], 'gpu')
        for policy, actor in ((0, 0), (0, 12), (12, 0), (4, 5)):
            with self.subTest(policy=policy, actor=actor):
                settings = pipeline.resolve_update_pipeline('gpu', 'gpu', policy, actor)
                self.assertEqual(settings['policy_cache_device_bytes'], policy * 1024**2)
                self.assertEqual(settings['actor_epoch_cache_bytes'], actor * 1024**2)
                self.assertEqual(settings['policy_cache_device'], '/GPU:0' if policy else None)
                self.assertEqual(settings['actor_epoch_cache_device'], '/GPU:0' if actor else None)

    def test_automatic_profile_does_not_admit_unvalidated_larger_caps(self):
        with patch.object(pipeline, '_gpu_inventory', side_effect=AssertionError('unexpected probe')):
            for policy, actor in ((8193, 8192), (8192, 8193), (16384, 16384)):
                with self.subTest(policy=policy, actor=actor):
                    automatic = pipeline.resolve_update_pipeline('gpu', 'auto', policy, actor)
                    self.assertEqual(automatic['resolved_profile'], 'host')
                    self.assertIn('explicit gpu profile', automatic['reason'])
                    self.assertEqual(automatic['policy_cache_device_bytes'], 0)
                    self.assertEqual(automatic['actor_epoch_cache_bytes'], 0)
                    explicit = pipeline.resolve_update_pipeline('gpu', 'gpu', policy, actor)
                    self.assertEqual(explicit['resolved_profile'], 'gpu')
                    self.assertEqual(explicit['policy_cache_device_bytes'], policy * 1024**2)
                    self.assertEqual(explicit['actor_epoch_cache_bytes'], actor * 1024**2)

    def test_invalid_configuration(self):
        for device in ('cuda', 'CPU', None, 0):
            with self.subTest(device=device), self.assertRaises(ValueError):
                pipeline.resolve_update_pipeline(device)
        for profile in ('unknown', None, 1):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                pipeline.resolve_update_pipeline('gpu', profile)
        with self.assertRaisesRegex(ValueError, 'GPU learner'):
            pipeline.resolve_update_pipeline('cpu', 'gpu')
        for value in (-1, True, False, 1., float('inf'), float('nan'), '8192', None):
            for keyword in ('policy_cache_mib', 'actor_epoch_cache_mib'):
                with self.subTest(value=value, keyword=keyword), self.assertRaises(ValueError):
                    pipeline.resolve_update_pipeline('gpu', **{keyword:value})

    def test_application_only_changes_cache_attributes_and_missing_fields_are_atomic(self):
        settings = pipeline.resolve_update_pipeline('gpu', 'gpu', 7, 9)
        trainer = SimpleNamespace(**{name:None for name in pipeline._CACHE_ATTRIBUTES},
                                  model=object(), optimizer=object(), rng=object(), epochs=2)
        untouched = {name:getattr(trainer, name) for name in ('model', 'optimizer', 'rng', 'epochs')}
        self.assertIsNone(pipeline.apply_update_pipeline(trainer, settings))
        for name in pipeline._CACHE_ATTRIBUTES:
            self.assertEqual(getattr(trainer, name), settings[name])
        for name, value in untouched.items():
            self.assertIs(getattr(trainer, name), value)
        before = dict(vars(trainer))
        incomplete = dict(settings); incomplete.pop('actor_epoch_cache_bytes')
        with self.assertRaises(KeyError):
            pipeline.apply_update_pipeline(trainer, incomplete)
        self.assertEqual(vars(trainer), before)
        with self.assertRaisesRegex(ValueError, 'does not support'):
            pipeline.apply_update_pipeline(SimpleNamespace(), settings)


if __name__ == '__main__':
    unittest.main()
