"""Compact device caches change storage only, including full PPO backtracking."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE,'Requires neural dependencies')
class CompactDeviceCacheTests(unittest.TestCase):
    def test_exact_decoded_bytes_and_encoded_budget(self):
        from patchwork_automaton.neural.packed_rollouts import DevicePolicyBatchCache
        source=np.zeros((2,4096),np.float32)
        source.view(np.uint32).reshape(-1)[:6]=[0x80000000,0x7fc00001,0x7fc12345,0xff800000,0x7f800000,1]
        batch=({'features':source,'candidate_mask':np.ones((2,2),bool),
                'advance_memory':np.ones((2,1),np.float32)},np.zeros(2,np.int32))
        cache=DevicePolicyBatchCache(device='/CPU:0',max_bytes=4096,compact=True)
        try:
            converted=cache.put((0,1),batch)
            self.assertIsNotNone(converted)
            self.assertLess(cache.bytes,4096)
            self.assertGreater(cache.dense_bytes,32000)
            self.assertEqual(cache.compact_arrays,1)
            source[:]=42
            expected=converted[0]['features'].numpy().tobytes()
            cache.finish()
            actual=cache.get((0,1))[0]['features']
            self.assertEqual(actual.numpy().tobytes(),expected)
            self.assertEqual(actual.numpy().view(np.uint32).reshape(-1)[:6].tolist(),
                             [0x80000000,0x7fc00001,0x7fc12345,0xff800000,0x7f800000,1])
            self.assertEqual(cache.counts((0,1)),(2,2.))
        finally:cache.close()

    def test_exact_concurrent_update_both_adams_rng_and_guard_results(self):
        from test_critic_warmup import CriticWarmupTests
        for budget in (0,40000,10**8):
            with self.subTest(budget=budget),tempfile.TemporaryDirectory() as d:
                m,a,paths=CriticWarmupTests().fixture(Path(d)/'a',value_max_steps=2)
                n,b,_=CriticWarmupTests().fixture(Path(d)/'b',value_max_steps=2)
                for trainer in (a,b):
                    trainer.guard_audit=True;trainer.numerical_prepacking=True
                    trainer.cache_policy_batches=True;trainer.measurement_workers=2
                    trainer.policy_cache_device=trainer.actor_epoch_cache_device='/CPU:0'
                    trainer.policy_cache_device_bytes=trainer.actor_epoch_cache_bytes=budget
                b.compact_device_cache=True
                a.compile();b.compile();x=a.update_files(paths);y=b.update_files(paths)
                for key in ('loss','policy_loss','value_loss','entropy','exact_kl','max_state_kl',
                            'accepted_steps','rejected_epochs','critic_steps','optimizer_steps',
                            'critic_optimizer_steps','early_stop','samples','guard_proposals'):
                    self.assertEqual(x[key],y[key],key)
                self.assertEqual(a.rng.bit_generator.state,b.rng.bit_generator.state)
                for left,right in [(m.variables,n.variables),(a.optimizer.variables,b.optimizer.variables),
                                   (a.critic_optimizer.variables,b.critic_optimizer.variables)]:
                    for u,v in zip(left,right):np.testing.assert_array_equal(np.asarray(u),np.asarray(v))
                self.assertLessEqual(y['pipeline']['policy_cache_device_bytes'],budget)
                self.assertTrue(all(e['released'] for e in y['pipeline']['actor_epoch_cache_epochs']))


if __name__=='__main__':unittest.main()
