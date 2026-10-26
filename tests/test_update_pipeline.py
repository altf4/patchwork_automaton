"""Pipeline scheduling must preserve PPO decisions and join all background work."""
import tempfile
import threading
import unittest
from pathlib import Path
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE,'Install neural dependencies with uv')
class UpdatePipelineTests(unittest.TestCase):
    def helper(self):
        from test_concurrent_ppo import ConcurrentPPOTests
        return ConcurrentPPOTests()

    def assert_joined(self):
        self.assertFalse(any(t.name.startswith(('ppo-prefetch','ppo-measure','ppo-critic'))
                             for t in threading.enumerate()))

    def test_pipeline_preserves_updates_rng_and_backtracking_with_cache_and_spill(self):
        import numpy as np
        helper=self.helper()
        for cache_bytes in (0,16*1024**2):
            with self.subTest(cache_bytes=cache_bytes),tempfile.TemporaryDirectory() as directory:
                model,trajectories,_=helper.fixture(directory)
                other,_,_=helper.fixture(Path(directory)/'other')
                other.set_weights(model.get_weights())
                trainers=[helper.trainer(model,True,cache_bytes),
                    helper.trainer(other,True,cache_bytes,prefetch_batches=2,measurement_workers=3)]
                results=[]
                for trainer in trainers:
                    forward=trainer.measure_actor_with_drift
                    lock=threading.Lock();first=[True]
                    def check(*args):
                        metrics,drift=forward(*args)
                        with lock:
                            reject=first[0];first[0]=False
                        return metrics,np.array([1.,1.]) if reject else drift
                    trainer.measure_actor_with_drift=check
                    results.append(trainer.update_files(trajectories))
                    self.assert_joined()
                helper.assert_state_equal(*trainers)
                for name in ('model','optimizer','critic_optimizer'):
                    left,right=(list(getattr(t,name).variables) for t in trainers)
                    self.assertEqual(len(left),len(right))
                    for a,b in zip(left,right):
                        np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
                for key in ('loss','value_loss','exact_kl','max_state_kl','accepted_steps','critic_steps','rejected_epochs'):
                    self.assertEqual(results[0][key],results[1][key])
                self.assertEqual(results[1]['rejected_epochs'],1)
                pipeline = results[1]['pipeline']
                self.assertEqual(pipeline['prefetch_batches'], 2)
                self.assertEqual(pipeline['measurement_workers'], 3)
                self.assertFalse(pipeline['numerical_prepacking'])
                self.assertFalse(pipeline['cache_policy_batches'])
                self.assertIsNone(pipeline['policy_cache_device'])
                self.assertIsNone(pipeline['actor_epoch_cache_device'])
                self.assertEqual(pipeline['actor_epoch_cache_epochs'], [])
                self.assertLessEqual(results[1]['cache']['retained_bytes'],cache_bytes)

    def test_parallel_measurement_calls_overlap(self):
        helper=self.helper()
        with tempfile.TemporaryDirectory() as directory:
            model,trajectories,_=helper.fixture(directory)
            trainer=helper.trainer(model,True,0,prefetch_batches=2,measurement_workers=3)
            forward=trainer.measure_with_drift
            barrier=threading.Barrier(3,timeout=15)
            lock=threading.Lock();calls=[0]
            def check(*args):
                with lock:
                    index=calls[0];calls[0]+=1
                if index<3:barrier.wait()
                return forward(*args)
            trainer.measure_with_drift=check
            trainer.update_files(trajectories)
            self.assert_joined()

    def test_failures_join_prefetch_measurement_and_optimizer_workers(self):
        helper=self.helper()
        for failing_method in ('step','critic_step','measure_actor_with_drift'):
            with self.subTest(failing_method=failing_method),tempfile.TemporaryDirectory() as directory:
                model,trajectories,_=helper.fixture(directory)
                trainer=helper.trainer(model,True,0,prefetch_batches=2,measurement_workers=3)
                def fail(*args):raise RuntimeError('pipeline failed')
                setattr(trainer,failing_method,fail)
                with self.assertRaisesRegex(RuntimeError,'pipeline failed'):
                    trainer.update_files(trajectories)
                self.assert_joined()

    def test_invalid_pipeline_limits_rejected(self):
        from patchwork_automaton.neural.streaming_ppo import StreamingPPOTrainer
        for options in ({'prefetch_batches':-1},{'prefetch_batches':True},{'measurement_workers':0}):
            with self.assertRaisesRegex(ValueError,'prefetch depth or measurement'):
                StreamingPPOTrainer(None,**options)
