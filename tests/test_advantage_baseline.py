"""Actor baseline ablations retain real critic data and frozen behavior statistics."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE, 'Requires neural dependencies')
class AdvantageBaselineTests(unittest.TestCase):
    def fixture(self, directory):
        from test_critic_warmup import CriticWarmupTests
        return CriticWarmupTests().fixture(directory,cache_bytes=0)

    def trajectories_with_values(self, trajectories, directory, values):
        from patchwork_automaton.neural.experiment import read_trajectory, TrajectoryWriter
        result = []; offset = 0
        for i,trajectory in enumerate(trajectories):
            path = Path(directory)/f'{i}.pkl.gz'
            writer = TrajectoryWriter(path)
            try:
                for sample in read_trajectory(trajectory['path']):
                    writer.write(replace(sample,value=float(values[offset])))
                    offset += 1
            finally:
                writer.close()
            result.append(dict(trajectory,path=str(path)))
        self.assertEqual(offset,len(values))
        return result

    def make_trainer(self, source, source_trainer, kind, baseline=None):
        import numpy as np
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.ppo import PPOTrainer
        from patchwork_automaton.neural.streaming_ppo import StreamingPPOTrainer
        model = type(source).from_config(source.get_config())
        model(example_batch(model.memory_size,model.feature_spec))
        # Keras get_weights excludes recurrent seed-generator state. Copy all
        # variables so default/explicit runs begin from the exact same state.
        for target,value in zip(model.variables,source.variables):
            target.assign(np.asarray(value))
        options = dict(learning_rate=.001,batch_size=2,epochs=1,target_kl=100.,
                       max_state_kl=100.,trust_guard=True,value_max_steps=1)
        if baseline is not None:
            options['advantage_baseline'] = baseline
        if kind == 'streamed':
            trainer = StreamingPPOTrainer(model,seed=29,cache_bytes=0,
                concurrent_updates=True,prefetch_batches=2,**options)
        else:
            trainer = PPOTrainer(model,**options)
        for name in ('optimizer','critic_optimizer'):
            for target,value in zip(getattr(trainer,name).variables,getattr(source_trainer,name).variables):
                target.assign(np.asarray(value))
        return model,trainer

    def update(self, trainer, kind, trajectories):
        import numpy as np
        from patchwork_automaton.neural.experiment import read_trajectory
        from patchwork_automaton.neural.features import collate
        if kind == 'streamed':
            return trainer.update_files(trajectories),trainer.rng.bit_generator.state
        samples = [s for t in trajectories for s in read_trajectory(t['path'])]
        inputs = collate([s.arrays for s in samples])
        inputs['memory_in'] = np.stack([s.memory for s in samples])
        inputs['advance_memory'] = np.asarray([[float(s.advance)] for s in samples],np.float32)
        rewards = [t['reward'] for t in trajectories for _ in range(t['samples'])]
        actions = np.asarray([s.action for s in samples],np.int32)
        old = np.stack([s.log_probs for s in samples])
        rng = np.random.default_rng(29)
        result = trainer.update(inputs,actions,rewards,old[np.arange(len(samples)),actions],
            [s.value for s in samples],rng=rng,old_log_probs=old)
        return result,rng.bit_generator.state

    def assert_state_equal(self, left, right, *, critic_only=False):
        import numpy as np
        left_model,left_trainer = left; right_model,right_trainer = right
        a,b = (left_model.critic,right_model.critic) if critic_only else (left_model,right_model)
        for x,y in zip(a.variables,b.variables):
            np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
        for name in (('critic_optimizer',) if critic_only else ('optimizer','critic_optimizer')):
            for x,y in zip(getattr(left_trainer,name).variables,getattr(right_trainer,name).variables):
                np.testing.assert_array_equal(np.asarray(x),np.asarray(y))

    def test_known_formula_in_memory_and_streamed_keeps_behavior_probabilities(self):
        import numpy as np
        from patchwork_automaton.neural.ppo import normalized_advantages, PPOTrainer
        from patchwork_automaton.neural.experiment import read_trajectory
        with tempfile.TemporaryDirectory() as directory:
            model,original,paths = self.fixture(Path(directory)/'source')
            values = np.array([2.,-1.,.5,-.25,-2.,1.,-.5,.25],np.float32)
            paths = self.trajectories_with_values(paths,Path(directory)/'values',values)
            rewards = np.array([1.]*4+[-1.]*4,np.float32)
            samples = [s for t in paths for s in read_trajectory(t['path'])]
            old_chosen = np.array([s.log_probs[s.action] for s in samples])
            for kind in ('in_memory','streamed'):
                for baseline in ('critic','zero'):
                    with self.subTest(kind=kind,baseline=baseline):
                        _,trainer = self.make_trainer(model,original,kind,baseline)
                        seen = []; chosen = []
                        trainer.fit_critic = lambda batches: {}
                        def step(inputs,actions,returns,old,advantages):
                            seen.extend(np.asarray(advantages)); chosen.extend(np.asarray(old))
                            return 0.
                        trainer.step = step
                        result,_ = self.update(trainer,kind,paths)
                        rng = np.random.default_rng(29)
                        rng.permutation(8)  # Reserved critic order precedes actor order.
                        order = rng.permutation(8)
                        expected = normalized_advantages(rewards,values if baseline=='critic' else np.zeros(8),
                            np.ones(8,bool),trainer.advantage_scale_floor)
                        np.testing.assert_array_equal(seen,expected[order])
                        np.testing.assert_array_equal(chosen,old_chosen[order])
                        self.assertEqual(result['advantage_baseline'],baseline)
                        self.assertEqual(result['behavior_critic']['mse'],float(np.mean((rewards-values)**2)))
            with self.assertRaisesRegex(ValueError,'baseline'):
                PPOTrainer(model,advantage_baseline='invalid')

    def test_explicit_critic_matches_default_weights_adam_and_rng_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            model,original,paths = self.fixture(directory)
            for kind in ('in_memory','streamed'):
                with self.subTest(kind=kind):
                    default = self.make_trainer(model,original,kind)
                    explicit = self.make_trainer(model,original,kind,'critic')
                    reference,rng = self.update(default[1],kind,paths)
                    result,other_rng = self.update(explicit[1],kind,paths)
                    self.assertEqual(reference,result)
                    self.assertEqual(rng,other_rng)
                    self.assert_state_equal(default,explicit)

    def test_zero_actor_ignores_behavior_values_and_critic_update_is_unchanged(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            model,original,paths = self.fixture(Path(directory)/'source')
            values = np.array([2.,-1.,.5,-.25,-2.,1.,-.5,.25],np.float32)
            paths = self.trajectories_with_values(paths,Path(directory)/'values',values)
            # A uniform shift would also cancel under centered critic
            # advantages. Vary individual predictions to distinguish modes.
            altered_values = values+np.array([5.,-3.,7.,-2.,4.,-6.,8.,-1.],np.float32)
            altered = self.trajectories_with_values(paths,Path(directory)/'altered',altered_values)
            for kind in ('in_memory','streamed'):
                with self.subTest(kind=kind):
                    zero = self.make_trainer(model,original,kind,'zero')
                    changed = self.make_trainer(model,original,kind,'zero')
                    critic = self.make_trainer(model,original,kind,'critic')
                    result,rng = self.update(zero[1],kind,paths)
                    changed_result,changed_rng = self.update(changed[1],kind,altered)
                    critic_result,critic_rng = self.update(critic[1],kind,paths)
                    self.assert_state_equal(zero,changed)
                    self.assert_state_equal(zero,critic,critic_only=True)
                    self.assertEqual(rng,changed_rng)
                    self.assertEqual(rng,critic_rng)
                    self.assertEqual(result['value_loss_before'],changed_result['value_loss_before'])
                    self.assertEqual(result['critic_steps'],critic_result['critic_steps'])
                    self.assertEqual(result['behavior_critic']['prediction_mean'],float(values.mean()))
                    self.assertEqual(changed_result['behavior_critic']['prediction_mean'],float(altered_values.mean()))
                    self.assertNotEqual(result['behavior_critic']['mse'],changed_result['behavior_critic']['mse'])
                    self.assertTrue(any(not np.array_equal(np.asarray(x),np.asarray(y))
                        for x,y in zip(zero[0].actor.variables,critic[0].actor.variables)))


if __name__ == '__main__':
    unittest.main()
