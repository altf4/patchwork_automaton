"""Caching fixed epoch inputs must preserve every backtracking PPO update."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE, 'Requires neural dependencies')
class ActorEpochCacheTests(unittest.TestCase):
    def fixture(self, directory):
        from test_critic_warmup import CriticWarmupTests
        model, trainer, paths = CriticWarmupTests().fixture(directory, value_max_steps=2)
        trainer.numerical_prepacking = trainer.cache_policy_batches = True
        trainer.policy_cache_device = '/CPU:0'
        trainer.policy_cache_device_bytes = 10**8
        trainer.guard_audit = True
        trainer.measurement_workers = 2
        trainer.backtracks = 2
        return model, trainer, paths

    def batch_bytes(self, paths):
        from patchwork_automaton.neural.experiment import read_trajectory
        from patchwork_automaton.neural.features import collate
        records = list(read_trajectory(paths[0]['path']))[:2]
        inputs = collate([s.arrays for s in records])
        inputs['memory_in'] = np.stack([s.memory for s in records])
        inputs['advance_memory'] = np.asarray([[float(s.advance)] for s in records], np.float32)
        return sum(v.nbytes for v in inputs.values()) + 4*2*4 + 2*len(records[0].log_probs)*4

    def update_with_two_backtracks_per_epoch(self, trainer, paths):
        # A fixture-only guard schedule ensures both epochs reuse the same
        # shuffled inputs three times. Both implementations run the identical
        # real forwards, gradients, rollbacks and effective learning rates.
        import patchwork_automaton.neural.streaming_ppo as streaming
        original = streaming._guard_status
        calls = []
        def guard(*args):
            natural = original(*args)
            self.assertTrue(natural[1], 'Fixture unexpectedly produced nonfinite metrics')
            calls.append(natural)
            if len(calls) % 3:
                return False, True, ['fixture_backtrack']
            return natural
        with patch.object(streaming, '_guard_status', guard):
            result = trainer.update_files(paths)
        self.assertEqual(len(calls), 6)
        self.assertEqual(result['rejected_epochs'], 4)
        self.assertEqual(result['accepted_steps'], 8)
        return result

    def test_concurrent_replays_exact_with_full_zero_prefix_and_no_prefetch(self):
        for requested, prefetch in ((0, 2), ('prefix', 2), (10**8, 2), ('prefix', 0)):
            with self.subTest(budget=requested, prefetch=prefetch), tempfile.TemporaryDirectory() as directory:
                model, baseline, paths = self.fixture(Path(directory)/'baseline')
                other, candidate, _ = self.fixture(Path(directory)/'candidate')
                budget = self.batch_bytes(paths) if requested == 'prefix' else requested
                for trainer in (baseline, candidate): trainer.prefetch_batches = prefetch
                candidate.actor_epoch_cache_device = '/CPU:0'
                candidate.actor_epoch_cache_bytes = budget
                baseline.compile(); candidate.compile()
                left = self.update_with_two_backtracks_per_epoch(baseline, paths)
                right = self.update_with_two_backtracks_per_epoch(candidate, paths)
                for key in ['loss', 'exact_kl', 'max_state_kl', 'value_loss', 'entropy',
                            'accepted_steps', 'rejected_epochs', 'critic_steps', 'optimizer_steps',
                            'critic_optimizer_steps', 'early_stop', 'samples', 'guard_proposals']:
                    self.assertEqual(left[key], right[key], key)
                self.assertEqual(baseline.rng.bit_generator.state, candidate.rng.bit_generator.state)
                for a, b in [(model.variables, other.variables),
                             (baseline.optimizer.variables, candidate.optimizer.variables),
                             (baseline.critic_optimizer.variables, candidate.critic_optimizer.variables)]:
                    self.assertEqual(len(a), len(b))
                    for x, y in zip(a, b): np.testing.assert_array_equal(x, y)
                x, y = left['pipeline'], right['pipeline']
                # Canonical policy caching remains separate and unchanged.
                for key in ['policy_cache_device_uploads', 'policy_cache_device_hits',
                            'policy_cache_device_bytes', 'policy_cache_device_fallback_batches']:
                    self.assertEqual(x[key], y[key], key)
                epochs = y['actor_epoch_cache_epochs']
                self.assertEqual([r['epoch'] for r in epochs], [1, 2])
                expected = 0 if budget == 0 else 1 if requested == 'prefix' else 4
                for row in epochs:
                    self.assertTrue(row['released'])
                    self.assertEqual(row['uploads'], expected)
                    self.assertEqual(row['fallback_batches'], 4-expected)
                    self.assertEqual(row['hits'], 2*expected)
                    self.assertEqual(row['unique_input_bytes'], 4*self.batch_bytes(paths))
                    self.assertLessEqual(row['logical_bytes'], budget)
                self.assertEqual(y['actor_epoch_cache_uploads'], 2*expected)
                self.assertEqual(y['actor_epoch_cache_hits'], 4*expected)
                # Only first-pass actor admissions and the independent critic
                # can pack; each hit saves exactly one repeated numerical gather.
                self.assertEqual(x['numerical_gathers']-y['numerical_gathers'], 4*expected)

    def test_epoch_caches_close_after_prefetch_join_on_failure_and_reject_oom(self):
        from patchwork_automaton.neural.packed_rollouts import DevicePolicyBatchCache
        import tensorflow as tf
        original_put = DevicePolicyBatchCache.put
        original_close = DevicePolicyBatchCache.close
        for method in ('step', 'measure_actor_with_drift'):
            readers = {}; closed = []
            def put(cache, *args):
                readers.setdefault(id(cache), set()).add(threading.current_thread())
                return original_put(cache, *args)
            def close(cache):
                self.assertTrue(all(not reader.is_alive() for reader in readers.get(id(cache), set())))
                original_close(cache)
                closed.append((cache.closed, cache.bytes, len(cache.entries)))
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                _, trainer, paths = self.fixture(directory)
                # Only actor caches are device-backed in this failure probe.
                trainer.policy_cache_device = None
                trainer.policy_cache_device_bytes = 0
                trainer.actor_epoch_cache_device = '/CPU:0'
                trainer.actor_epoch_cache_bytes = 10**8
                with patch.object(DevicePolicyBatchCache, 'put', put), \
                     patch.object(DevicePolicyBatchCache, 'close', close), \
                     patch.object(trainer, method, side_effect=RuntimeError('injected actor failure')):
                    with self.assertRaisesRegex(RuntimeError, 'injected actor failure'):
                        trainer.update_files(paths)
            self.assertEqual(closed, [(True, 0, 0)])
        with tempfile.TemporaryDirectory() as directory:
            _, trainer, paths = self.fixture(directory)
            trainer.policy_cache_device = None
            trainer.policy_cache_device_bytes = 0
            trainer.actor_epoch_cache_device = '/CPU:0'
            trainer.actor_epoch_cache_bytes = 10**8
            with patch.object(DevicePolicyBatchCache, 'put', side_effect=tf.errors.ResourceExhaustedError(None, None, 'OOM')) as upload:
                with self.assertRaises(tf.errors.ResourceExhaustedError): trainer.update_files(paths)
            # No synchronous replay, fallback retry, or optimizer call is used
            # to conceal an allocation failure. A prefetched sibling may also
            # already have reached its own independent failing preparation.
            self.assertLessEqual(upload.call_count, trainer.prefetch_batches)

    def test_validation_and_critic_warmup_never_create_actor_epoch_cache(self):
        from patchwork_automaton.neural.streaming_ppo import StreamingPPOTrainer
        from patchwork_automaton.neural.packed_rollouts import DevicePolicyBatchCache
        for invalid in (True, -1, .5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'Actor epoch cache budget'):
                StreamingPPOTrainer(object(), actor_epoch_cache_bytes=invalid)
        with self.assertRaisesRegex(ValueError, 'requires an explicit device'):
            StreamingPPOTrainer(object(), actor_epoch_cache_bytes=1)
        with tempfile.TemporaryDirectory() as directory:
            _, trainer, paths = self.fixture(directory)
            trainer.policy_cache_device = None
            trainer.policy_cache_device_bytes = 0
            trainer.actor_epoch_cache_device = '/CPU:0'
            trainer.actor_epoch_cache_bytes = 10**8
            with patch.object(DevicePolicyBatchCache, '__init__', side_effect=AssertionError('Actor cache used during critic warmup')):
                result = trainer.warmup_critic_files(paths)
            self.assertEqual(result['critic_steps'], 2)
            self.assertEqual(result['pipeline']['actor_epoch_cache_epochs'], [])
            self.assertEqual(result['pipeline']['actor_epoch_cache_uploads'], 0)

    def test_rejected_epoch_releases_cache_before_early_stop(self):
        import patchwork_automaton.neural.streaming_ppo as streaming
        with tempfile.TemporaryDirectory() as directory:
            _, trainer, paths = self.fixture(directory)
            trainer.actor_epoch_cache_device = '/CPU:0'
            trainer.actor_epoch_cache_bytes = 10**8
            with patch.object(streaming, '_guard_status', return_value=(False, True, ['fixture_reject'])):
                result = trainer.update_files(paths)
        self.assertTrue(result['early_stop'])
        self.assertEqual(result['accepted_steps'], 0)
        self.assertEqual(result['rejected_epochs'], 3)
        rows = result['pipeline']['actor_epoch_cache_epochs']
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['released'])
        self.assertEqual(rows[0]['uploads'], 4)
        self.assertEqual(rows[0]['hits'], 8)


if __name__ == '__main__': unittest.main()
