"""Independent PPO scheduling preserves updates, rollback, and durable state."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE, 'Install neural dependencies with uv')
class ConcurrentPPOTests(unittest.TestCase):
    def fixture(self, directory):
        import numpy as np
        import tensorflow as tf
        from test_separate_critic import SeparateCriticTests
        from patchwork_automaton.neural.training import Sample
        from patchwork_automaton.neural.experiment import TrajectoryWriter
        model, _, batch, config = SeparateCriticTests().setup_model()
        trajectories = []
        for game in range(4):
            path = Path(directory)/f'{game}.pkl.gz'
            writer = TrajectoryWriter(path)
            try:
                for position in range(4):
                    row = (game+position) % 2
                    inputs = {k: v[row:row+1].copy() for k,v in batch.items()}
                    inputs['memory_in'][0, 0] = (game*4+position)/20
                    inputs['advance_memory'][:] = position != 3
                    output = model(inputs)
                    logp = np.asarray(tf.nn.log_softmax(output['logits']))[0]
                    arrays = {k: v[0] for k,v in inputs.items() if k not in ('memory_in','advance_memory')}
                    writer.write(Sample(arrays, inputs['memory_in'][0], position != 3,
                        position % 2, float(output['value'][0,0]), logp.copy()))
            finally:
                writer.close()
            trajectories.append(dict(path=str(path), samples=4, reward=1. if game % 2 else -1.))
        return model, trajectories, config

    def trainer(self, model, concurrent, cache_bytes, **options):
        from patchwork_automaton.neural.streaming_ppo import StreamingPPOTrainer
        trainer = StreamingPPOTrainer(model, concurrent_updates=concurrent,
            cache_bytes=cache_bytes, seed=821, batch_size=4, epochs=2,
            learning_rate=.0001, target_kl=.02, max_state_kl=.1, backtracks=1, **options)
        trainer.compile()
        return trainer

    def assert_state_equal(self, left, right):
        import numpy as np
        for a,b in zip(left.model.get_weights(), right.model.get_weights()):
            np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7)
        for name in ('optimizer','critic_optimizer'):
            for a,b in zip(getattr(left,name).variables,getattr(right,name).variables):
                np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7)
        self.assertEqual(left.rng.bit_generator.state,right.rng.bit_generator.state)

    def test_serial_and_concurrent_updates_match_with_spill_cache_and_backtracking(self):
        import numpy as np
        for cache_bytes, rejection in ((0,'first'), (16*1024**2,'none'), (0,'all')):
            with self.subTest(cache_bytes=cache_bytes,rejection=rejection), tempfile.TemporaryDirectory() as directory:
                model, trajectories, _ = self.fixture(directory)
                other, _, _ = self.fixture(Path(directory)/'other')
                other.set_weights(model.get_weights())
                trainers = [self.trainer(m,parallel,cache_bytes) for m,parallel in ((model,False),(other,True))]
                results = []
                actor_before = [np.asarray(v).copy() for v in model.actor.variables]
                for trainer in trainers:
                    original = trainer.measure_actor_with_drift
                    calls = [0]
                    def measure(*args):
                        terms, drift = original(*args)
                        calls[0] += 1
                        if rejection == 'all' or rejection == 'first' and calls[0] == 1:
                            drift = np.array([1.,1.])
                        return terms, drift
                    trainer.measure_actor_with_drift = measure
                    results.append(trainer.update_files(trajectories))
                self.assert_state_equal(*trainers)
                for key in ('loss','policy_loss','value_loss','value_loss_before','exact_kl',
                            'accepted_steps','critic_steps','rejected_epochs','early_stop'):
                    np.testing.assert_allclose(results[0][key],results[1][key],rtol=1e-6,atol=1e-7)
                for result in results:
                    self.assertEqual(result['critic_steps'],8)
                    self.assertLessEqual(result['cache']['retained_bytes'],cache_bytes)
                    self.assertEqual(result['cache']['spill_bytes'] == 0,cache_bytes != 0)
                if rejection != 'none': self.assertGreater(results[0]['rejected_epochs'],0)
                if rejection == 'all':
                    self.assertEqual(results[0]['accepted_steps'],0)
                    for a,b in zip(actor_before,other.actor.variables): np.testing.assert_array_equal(a,b)

    def test_optimizer_steps_really_overlap_and_actor_checks_never_read_critic(self):
        with tempfile.TemporaryDirectory() as directory:
            model, trajectories, _ = self.fixture(directory)
            trainer = self.trainer(model,True,0)
            barrier = threading.Barrier(2,timeout=15)
            for name in ('step','critic_step'):
                original = getattr(trainer,name)
                def wrap(function):
                    first = [True]
                    def step(*args):
                        if first[0]:
                            first[0] = False
                            barrier.wait()
                        return function(*args)
                    return step
                setattr(trainer,name,wrap(original))
            result = trainer.update_files(trajectories)
            self.assertEqual(result['update_schedule'],'concurrent')
            import numpy as np
            from patchwork_automaton.neural.batching import example_batch
            batch = example_batch(model.memory_size,model.feature_spec)
            zeros = np.zeros(1,np.float32)
            with patch.object(model,'value_output',side_effect=AssertionError('Read critic during actor check')):
                trainer._measure_actor_with_drift(batch,np.zeros(1,np.int32),zeros,zeros,zeros,
                                                  np.zeros((1,1),np.float32))

    def test_actor_failure_joins_critic_and_critic_failure_propagates(self):
        for fail_actor in (True,False):
            with self.subTest(fail_actor=fail_actor), tempfile.TemporaryDirectory() as directory:
                model, trajectories, _ = self.fixture(directory)
                trainer = self.trainer(model,True,0)
                actor_started = threading.Event(); critic_finished = threading.Event()
                original_step = trainer.step
                original_fit = trainer.fit_critic
                def actor(*args):
                    actor_started.set()
                    if fail_actor: raise RuntimeError('actor failed')
                    return original_step(*args)
                def critic(batches):
                    try:
                        if not actor_started.wait(15): raise AssertionError('actor never started')
                        if not fail_actor: raise RuntimeError('critic failed')
                        return original_fit(batches)
                    finally: critic_finished.set()
                trainer.step = actor; trainer.fit_critic = critic
                with self.assertRaisesRegex(RuntimeError,'actor failed' if fail_actor else 'critic failed'):
                    trainer.update_files(trajectories)
                self.assertTrue(critic_finished.is_set())
                self.assertFalse(any(t.name.startswith('ppo-critic') for t in threading.enumerate()))

    def test_concurrent_checkpoint_resume_preserves_schedule_and_rng(self):
        from patchwork_automaton.neural.experiment import save_checkpoint,load_checkpoint
        from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
        with tempfile.TemporaryDirectory() as directory:
            model, trajectories, config = self.fixture(directory)
            trainer = self.trainer(model,True,16*1024**2,prefetch_batches=2,measurement_workers=3)
            config.update(concurrent_updates=True,learning_rate=.0001,train_batch=4,ppo_epochs=2,
                          target_kl=.02,max_state_kl=.1,advantage_scale_floor=.1,
                          prefetch_batches=2,measurement_workers=3)
            trainer.update_files(trajectories)
            save_checkpoint(Path(directory)/'checkpoints',model,trainer,dict(round=1,config=config),policy_type=RustMagePolicy)
            _, restored, _ = load_checkpoint(Path(directory)/'checkpoints')
            self.assertTrue(restored.concurrent_updates)
            self.assertEqual(restored.prefetch_batches,2)
            self.assertEqual(restored.measurement_workers,3)
            trainer.update_files(trajectories); restored.update_files(trajectories)
            self.assert_state_equal(trainer,restored)

    def test_shared_network_cannot_enable_concurrent_updates(self):
        from patchwork_automaton.neural.policy import untrained_policy
        from patchwork_automaton.neural.streaming_ppo import StreamingPPOTrainer
        model = untrained_policy(seed=19,width=16,memory_size=16,blocks=0).model
        with self.assertRaisesRegex(ValueError,'separate actor and critic'):
            StreamingPPOTrainer(model,concurrent_updates=True)
