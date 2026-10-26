"""Device policy inputs are immutable storage, never a PPO recipe change."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE, 'Requires neural dependencies')
class DevicePolicyCacheTests(unittest.TestCase):
    def batch(self):
        return ({'candidate_mask': np.asarray([[True, True], [True, False]]),
                 'advance_memory': np.asarray([[1.], [0.]], np.float32),
                 'empty': np.empty((2, 0), np.int32),
                 'features': np.arange(6, dtype=np.float32).reshape(2, 3)},
                np.asarray([1, 0], np.int32), np.asarray([1., -1.], np.float32),
                np.zeros(2, np.float32), np.ones(2, np.float32),
                np.zeros((2, 2), np.float32))

    def assert_batch(self, reference, actual):
        self.assertEqual(list(reference[0]), list(actual[0]))
        left = [*reference[0].values(), *reference[1:]]
        right = [*actual[0].values(), *actual[1:]]
        self.assertEqual(len(left), len(right))
        for source, tensor in zip(left, right):
            self.assertTrue(tensor.device.endswith('/device:CPU:0'))
            self.assertEqual(source.dtype, tensor.dtype.as_numpy_dtype)
            self.assertEqual(source.shape, tuple(tensor.shape))
            np.testing.assert_array_equal(source, tensor.numpy())

    def test_exact_tensor_values_metadata_once_upload_freeze_and_release(self):
        from patchwork_automaton.neural.packed_rollouts import DevicePolicyBatchCache
        batch = self.batch()
        size = sum(x.nbytes for x in [*batch[0].values(), *batch[1:]])
        cache = DevicePolicyBatchCache(device='/CPU:0', max_bytes=size)
        self.assertIsNone(cache.get((0, 1)))
        converted = cache.put((0, 1), batch)
        self.assert_batch(batch, converted)
        self.assertEqual(cache.counts((0, 1)), (1, 1.))
        self.assertEqual(cache.bytes, size)
        self.assertEqual(cache.uploads, 1)
        # Input mutation cannot alter cached tensors or frozen reduction counts.
        batch[0]['features'][:] = -7
        batch[0]['candidate_mask'][:] = False
        batch[0]['advance_memory'][:] = 9
        self.assertIs(cache.put((0, 1), batch), converted)
        cache.finish()
        self.assertIs(cache.get((0, 1)), converted)
        np.testing.assert_array_equal(converted[0]['features'], np.arange(6).reshape(2, 3))
        self.assertEqual(cache.counts((0, 1)), (1, 1.))
        self.assertEqual(cache.uploads, 1)
        with self.assertRaisesRegex(ValueError, 'finalized'):
            cache.put((2, 3), batch)
        with self.assertRaises(TypeError): cache.entries[(8,)] = converted
        del converted
        cache.close()
        self.assertEqual(cache.bytes, 0)
        self.assertEqual(len(cache.entries), 0)
        with self.assertRaisesRegex(ValueError, 'closed'): cache.get((0, 1))

    def test_budget_admission_is_prefix_and_fallback_is_counted_once(self):
        from patchwork_automaton.neural.packed_rollouts import DevicePolicyBatchCache
        batch = self.batch()
        size = sum(x.nbytes for x in [*batch[0].values(), *batch[1:]])
        cache = DevicePolicyBatchCache(device='/CPU:0', max_bytes=size + 1)
        first = cache.put((0, 1), batch)
        self.assertIsNotNone(first)
        self.assertIsNone(cache.put((2, 3), batch))
        # Even a smaller later batch cannot leapfrog the rejected prefix.
        smaller = ({k: v[:0] for k, v in batch[0].items()}, *[v[:0] for v in batch[1:]])
        self.assertIsNone(cache.put((4,), smaller))
        self.assertIsNone(cache.put((2, 3), batch))
        self.assertEqual(cache.uploads, 1)
        self.assertEqual(cache.fallback_batches, 2)
        self.assertEqual(cache.fallback_bytes, size)
        self.assertEqual(cache.counts((2, 3)), (1, 1.))
        cache.finish()
        self.assertIs(cache.get((0, 1)), first)
        self.assertIsNone(cache.get((2, 3)))
        del first
        cache.close()
        zero = DevicePolicyBatchCache(device='/CPU:0', max_bytes=0)
        self.assertIsNone(zero.put((0, 1), batch))
        self.assertEqual(zero.bytes, 0)
        self.assertEqual(zero.fallback_batches, 1)
        zero.close()

    def test_invalid_budget_device_and_soft_placement_rejected_without_replay(self):
        import tensorflow as tf
        from patchwork_automaton.neural.packed_rollouts import DevicePolicyBatchCache
        for invalid in (True, -1, 1.5):
            with self.subTest(budget=invalid), self.assertRaises(ValueError):
                DevicePolicyBatchCache(device='/CPU:0', max_bytes=invalid)
        for invalid in (None, 1, False, '', '/GPU', '/CPU:-1', '/CPU:99999', '/job:worker/device:CPU:0'):
            with self.subTest(device=invalid), self.assertRaises(ValueError):
                DevicePolicyBatchCache(device=invalid, max_bytes=1)
        cache = DevicePolicyBatchCache(device='/CPU:0', max_bytes=10**6)
        # Simulate a requested GPU that TensorFlow soft-places on CPU.
        cache.device = '/GPU:99999'
        with self.assertRaisesRegex(RuntimeError, 'placement'):
            cache.put((0, 1), self.batch())
        self.assertEqual(cache.uploads, 0)
        self.assertFalse(cache.entries)
        cache.close()
        cache = DevicePolicyBatchCache(device='/CPU:0', max_bytes=10**6)
        with patch.object(tf, 'convert_to_tensor', side_effect=tf.errors.ResourceExhaustedError(None, None, 'OOM')) as convert:
            with self.assertRaises(tf.errors.ResourceExhaustedError):
                cache.put((0, 1), self.batch())
        self.assertEqual(convert.call_count, 1)
        self.assertEqual(cache.uploads, 0)
        self.assertFalse(cache.entries)
        cache.close()

    def test_concurrent_update_cpu_tensor_cache_exact_full_and_prefix_fallback(self):
        from test_critic_warmup import CriticWarmupTests
        from patchwork_automaton.neural.experiment import read_trajectory
        from patchwork_automaton.neural.features import collate
        for requested, host_fallback in ((0, True), ('prefix', True), (10**8, True), ('prefix', False)):
            with self.subTest(budget=requested, host_fallback=host_fallback), tempfile.TemporaryDirectory() as directory:
                fixture = CriticWarmupTests()
                model, host, paths = fixture.fixture(Path(directory)/'host', value_max_steps=2)
                other, device, _ = fixture.fixture(Path(directory)/'device', value_max_steps=2)
                records = list(read_trajectory(paths[0]['path']))[:2]
                first = collate([s.arrays for s in records])
                first['memory_in'] = np.stack([s.memory for s in records])
                first['advance_memory'] = np.asarray([[float(s.advance)] for s in records], np.float32)
                # actions, rewards, chosen logprob, advantage and old full logprobs.
                first_bytes = sum(v.nbytes for v in first.values()) + 4 * 2 * 4 + 2 * len(records[0].log_probs) * 4
                budget = first_bytes if requested == 'prefix' else requested
                for trainer in (host, device):
                    trainer.numerical_prepacking = True
                    trainer.cache_policy_batches = host_fallback
                    trainer.guard_audit = True
                    trainer.measurement_workers = 2
                device.policy_cache_device = '/CPU:0'
                device.policy_cache_device_bytes = budget
                host.compile(); device.compile()
                baseline = host.update_files(paths)
                actual = device.update_files(paths)
                for key in ['loss', 'exact_kl', 'max_state_kl', 'value_loss', 'entropy',
                            'accepted_steps', 'rejected_epochs', 'critic_steps', 'optimizer_steps',
                            'critic_optimizer_steps', 'early_stop', 'samples', 'guard_proposals']:
                    self.assertEqual(baseline[key], actual[key], key)
                self.assertEqual(host.rng.bit_generator.state, device.rng.bit_generator.state)
                for left, right in [(model.variables, other.variables),
                                    (host.optimizer.variables, device.optimizer.variables),
                                    (host.critic_optimizer.variables, device.critic_optimizer.variables)]:
                    self.assertEqual(len(left), len(right))
                    for a, b in zip(left, right): np.testing.assert_array_equal(a, b)
                stats = actual['pipeline']
                self.assertLessEqual(stats['policy_cache_device_bytes'], budget)
                self.assertEqual(stats['policy_cache_device_uploads'] + stats['policy_cache_device_fallback_batches'], 4)
                if budget == 0:
                    self.assertEqual(stats['policy_cache_device_uploads'], 0)
                    self.assertGreater(stats['policy_cache_hits'], 0)
                elif budget == 10**8:
                    self.assertEqual(stats['policy_cache_device_uploads'], 4)
                    self.assertGreater(stats['policy_cache_device_hits'], 0)
                    self.assertEqual(stats['policy_cache_disk_bytes'], 0)
                else:
                    self.assertEqual(stats['policy_cache_device_uploads'], 1)
                    self.assertEqual(stats['policy_cache_device_fallback_batches'], 3)
                    self.assertEqual(stats['policy_cache_disk_bytes'] > 0, host_fallback)

    def test_device_references_release_after_measurement_and_optimizer_exceptions(self):
        from test_critic_warmup import CriticWarmupTests
        from patchwork_automaton.neural.packed_rollouts import DevicePolicyBatchCache
        original = DevicePolicyBatchCache.close
        for method in ('measure_with_drift', 'step'):
            closed = []
            def close(cache):
                original(cache)
                closed.append((cache.closed, cache.bytes, len(cache.entries)))
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                _, trainer, paths = CriticWarmupTests().fixture(directory)
                trainer.numerical_prepacking = trainer.cache_policy_batches = True
                trainer.policy_cache_device = '/CPU:0'
                trainer.policy_cache_device_bytes = 10**8
                trainer.measurement_workers = 2
                with patch.object(DevicePolicyBatchCache, 'close', close), \
                     patch.object(trainer, method, side_effect=RuntimeError('injected failure')):
                    with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                        trainer.update_files(paths)
            self.assertEqual(closed, [(True, 0, 0)])


if __name__ == '__main__': unittest.main()
