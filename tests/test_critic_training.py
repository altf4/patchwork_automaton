"""Critic ablations preserve actors/checkpoints and train ordered recurrence."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE, 'Requires neural dependencies')
class CriticTrainingTests(unittest.TestCase):
    def fixture(self, directory):
        from test_critic_warmup import CriticWarmupTests
        return CriticWarmupTests().fixture(directory,cache_bytes=0,value_max_steps=2)

    def clone(self, model, source, **options):
        import numpy as np
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.streaming_ppo import StreamingPPOTrainer
        other = type(model).from_config(model.get_config())
        other(example_batch(other.memory_size,other.feature_spec))
        for target,value in zip(other.variables,model.variables):target.assign(np.asarray(value))
        trainer = StreamingPPOTrainer(other,seed=29,batch_size=8,epochs=2,
            learning_rate=.001,value_max_steps=2,cache_bytes=0,
            concurrent_updates=True,prefetch_batches=2,**options)
        for name in ('optimizer','critic_optimizer'):
            for target,value in zip(getattr(trainer,name).variables,getattr(source,name).variables):
                target.assign(np.asarray(value))
        return other,trainer

    def test_head_only_preserves_actor_representation_and_all_adam_slots(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            source,original,paths = self.fixture(directory)
            model,trainer = self.clone(source,original,critic_update_scope='head')
            actor = [np.asarray(v).copy() for v in model.actor.variables]
            actor_adam = [np.asarray(v).copy() for v in trainer.optimizer.variables]
            head = {id(v) for v in model.critic.value_head.trainable_variables}
            representation = [(v,np.asarray(v).copy()) for v in model.critic.variables if id(v) not in head]
            head_before = [np.asarray(v).copy() for v in model.critic.value_head.variables]
            optimizer_slots = len(trainer.critic_optimizer.variables)
            result = trainer.warmup_critic_files(paths)
            self.assertEqual(result['critic_update_scope'],'head')
            self.assertEqual(result['critic_steps'],2)
            self.assertEqual(len(trainer.critic_optimizer.variables),optimizer_slots)
            self.assertEqual(optimizer_slots,len(original.critic_optimizer.variables))
            for v,before in representation:np.testing.assert_array_equal(v,before)
            for before,v in zip(actor,model.actor.variables):np.testing.assert_array_equal(v,before)
            for before,v in zip(actor_adam,trainer.optimizer.variables):np.testing.assert_array_equal(v,before)
            self.assertTrue(any(not np.array_equal(a,np.asarray(b)) for a,b in zip(head_before,model.critic.value_head.variables)))

    def test_default_matches_explicit_controls_and_sequence_is_actor_independent(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            source,original,paths = self.fixture(directory)
            model,default = self.clone(source,original)
            other,explicit = self.clone(source,original,critic_update_scope='all',critic_unroll=1)
            reference = default.update_files(paths)
            result = explicit.update_files(paths)
            self.assertEqual(reference,result)
            self.assertEqual(default.rng.bit_generator.state,explicit.rng.bit_generator.state)
            for a,b in zip(model.variables,other.variables):np.testing.assert_array_equal(a,b)
            for name in ('optimizer','critic_optimizer'):
                for a,b in zip(getattr(default,name).variables,getattr(explicit,name).variables):
                    np.testing.assert_array_equal(a,b)
            sequence_model,sequence = self.clone(source,original,critic_unroll=8)
            actor = [np.asarray(v).copy() for v in sequence_model.actor.variables]
            adam = [np.asarray(v).copy() for v in sequence.optimizer.variables]
            critic = [np.asarray(v).copy() for v in sequence_model.critic.variables]
            result = sequence.warmup_critic_files(paths)
            self.assertEqual(result['critic_steps'],2)
            self.assertEqual(result['critic_optimizer_steps'],2)
            self.assertEqual(result['critic_sequence_batch'],1)
            self.assertEqual(result['critic_sequence_positions'],8)
            self.assertEqual(result['critic_value_samples'],8)
            self.assertEqual(result['critic_sequence_padding'],8)
            self.assertGreater(result['cache']['spill_bytes'],0)
            for before,v in zip(actor,sequence_model.actor.variables):np.testing.assert_array_equal(v,before)
            for before,v in zip(adam,sequence.optimizer.variables):np.testing.assert_array_equal(v,before)
            self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in sequence_model.critic.variables))
            self.assertTrue(any(not np.array_equal(a,np.asarray(b)) for a,b in zip(critic,sequence_model.critic.variables)))
            with self.assertRaisesRegex(ValueError,'ordered trajectories'):
                sequence.update(None,None,None,None,None,rng=None)

    def test_sequence_gradient_crosses_time_and_padding_has_no_effect(self):
        import keras
        import numpy as np
        import tensorflow as tf
        from patchwork_automaton.neural.sequence_critic import sequence_step
        class TemporalCritic(keras.Model):
            memory_size = 1
            def __init__(self):
                super().__init__()
                self.recurrence = self.add_weight(shape=(),initializer='ones')
                self.value_head = keras.layers.Dense(1,use_bias=False,kernel_initializer='ones')
            def call(self, inputs, training=False):
                state = inputs['memory_in']+self.recurrence*inputs['context'][:,:1]
                return dict(memory_out=state,value=self.value_head(state)*inputs['context'][:,1:2])
        def make():
            critic = TemporalCritic()
            critic(dict(memory_in=tf.zeros((1,1)),context=tf.zeros((1,2))))
            optimizer = keras.optimizers.Adam(.01)
            optimizer.build(critic.trainable_variables)
            return SimpleNamespace(model=SimpleNamespace(critic=critic),critic_update_scope='all',
                                   critic_optimizer=optimizer,clip_norm=100.)
        plain,padded = make(),make()
        def batch(first,second):
            return dict(context=tf.constant([first,second],tf.float32),advance_memory=tf.ones((2,1)))
        chunks = (batch([1.,0.],[100.,100.]),batch([0.,1.],[100.,100.]))
        actual = sequence_step(padded,chunks,tf.constant([[0.],[7.]]),tf.constant([0.,1000.]),
                               tf.constant([[1.,0.],[1.,0.]]))
        reference = sequence_step(plain,tuple({k:v[:1] for k,v in b.items()} for b in chunks),
                                  tf.zeros((1,1)),tf.zeros(1),tf.ones((2,1)))
        self.assertLess(float(plain.model.critic.recurrence),1.)
        self.assertEqual(float(actual[2]),2.)
        self.assertEqual(float(actual[3]),.5)
        np.testing.assert_array_equal(actual[1][1],[7.])
        self.assertEqual(float(actual[0]),float(reference[0]))
        for a,b in zip(plain.model.critic.variables,padded.model.critic.variables):np.testing.assert_array_equal(a,b)
        for a,b in zip(plain.critic_optimizer.variables,padded.critic_optimizer.variables):np.testing.assert_array_equal(a,b)

    def test_sequence_scheduler_orders_records_resets_and_carries_bounded_memory(self):
        import numpy as np
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.sequence_critic import fit_sequence_critic
        template = {k:v[0].copy() for k,v in example_batch(1).items()
                    if k not in ('memory_in','advance_memory')}
        trajectories = [dict(samples=n,reward=r) for n,r in ((10,1.),(5,-1.),(9,0.))]
        reads = []; batches = []
        def get_record(identity,position):
            arrays = {k:v.copy() for k,v in template.items()}
            arrays['context'][:2] = identity,position
            reads.append((identity,position))
            return SimpleNamespace(arrays=arrays,advance=True)
        def step(inputs,memory,returns,valid):
            self.assertLessEqual(valid.size,8)
            batches.append(valid.size)
            for column in range(valid.shape[1]):
                positions = [inputs[t]['context'][column,1] for t in range(4) if valid[t,column]]
                self.assertEqual(positions,list(range(int(memory[column,0]),int(memory[column,0])+len(positions))))
                identity = int(inputs[0]['context'][column,0])
                self.assertEqual(returns[column],trajectories[identity]['reward'])
            return 0.,memory+valid.sum(axis=0)[:,None],valid.sum(),0.
        trainer = SimpleNamespace(batch_size=8,critic_unroll=4,epochs=1,value_max_steps=None,
            model=SimpleNamespace(critic=SimpleNamespace(memory_size=1)),critic_sequence_step=step,
            critic_optimizer=SimpleNamespace(iterations=0,learning_rate=SimpleNamespace(numpy=lambda:np.array(.001))),
            critic_update_scope='all')
        result = fit_sequence_critic(trainer,trajectories,iter([np.arange(24)]),get_record)
        self.assertEqual(sorted(reads),[(i,p) for i,t in enumerate(trajectories) for p in range(t['samples'])])
        self.assertEqual(result['critic_sequence_positions'],24)
        self.assertEqual(result['critic_value_samples'],24)
        self.assertEqual(result['critic_sequence_batch'],2)
        self.assertEqual(result['critic_sequence_padding'],sum(batches)-24)


if __name__ == '__main__':
    unittest.main()
