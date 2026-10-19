"""Independent gradients, memory, trust-region rollback and durable optimizers."""
import tempfile
import unittest
from test_neural import AVAILABLE

if AVAILABLE:
    import keras
    import numpy as np
    import tensorflow as tf
    from patchwork_automaton.neural.model import SeparateActorCritic
    from patchwork_automaton.neural.rustmage_features import RustMageEncoder
    from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
    from patchwork_automaton.neural.batching import example_batch
    from patchwork_automaton.neural.experiment import make_trainer, save_checkpoint, load_checkpoint


@unittest.skipUnless(AVAILABLE, 'Requires neural extra')
class SeparateCriticTests(unittest.TestCase):
    def setup_model(self):
        keras.utils.set_random_seed(19)
        encoder = RustMageEncoder()
        model = SeparateActorCritic(encoder.card_count, encoder.feature_spec,
            actor_width=16, actor_memory=8, actor_blocks=0, critic_width=16, critic_memory=8)
        batch = {k: np.repeat(v, 2, axis=0) for k,v in example_batch(model.memory_size, encoder.feature_spec).items()}
        for key in ('candidates','candidate_refs','candidate_mask'):
            batch[key] = np.repeat(batch[key], 2, axis=1)
        batch['candidates'][:, 1, 0] = 1.
        batch['entities'][:, 0, 0] = [1., -1.]
        batch['card_ids'][:, 0] = 1
        batch['semantics'][:, 0, 0, 0] = 1.
        batch['memory_in'][:] = .1
        model(batch)
        config = dict(algorithm='ppo',learning_rate=.001,train_batch=2,ppo_epochs=1,seed=19)
        trainer = make_trainer(model, config); trainer.compile()
        return model, trainer, batch, config

    def test_value_limit_spans_epochs_closes_streams_and_resets_per_update(self):
        model, _, batch, config = self.setup_model()
        config.update(ppo_epochs=4,value_max_steps=3,value_learning_rate=.00007)
        trainer = make_trainer(model,config)
        opened = []; closed = []
        def batches():
            opened.append(True)
            try:
                for _ in range(2): yield batch,np.array([1.,-1.],np.float32)
            finally: closed.append(True)
        for invocation in (1,2):
            stats = trainer.fit_critic(batches)
            self.assertEqual(stats['critic_steps'],3)
            self.assertEqual(stats['critic_optimizer_steps'],3*invocation)
            self.assertTrue(stats['value_step_limit_reached'])
            self.assertEqual(len(opened),2*invocation)
            self.assertEqual(len(closed),len(opened))
        self.assertAlmostEqual(float(trainer.optimizer.learning_rate.numpy()),.001)
        self.assertAlmostEqual(float(trainer.critic_optimizer.learning_rate.numpy()),.00007)
        trainer.value_max_steps = 20
        stats = trainer.fit_critic(batches)
        self.assertEqual(stats['critic_steps'],8)
        self.assertFalse(stats['value_step_limit_reached'])

    def test_invalid_value_settings_and_shared_network_rejected(self):
        from patchwork_automaton.neural.ppo import PPOTrainer
        model, _, _, _ = self.setup_model()
        for settings in ({'value_max_steps':v} for v in (0,-1,1.5,True)):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError,'Value step limit'):
                PPOTrainer(model,**settings)
        for rate in (0,-1,float('inf'),float('nan')):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError,'Value learning rate'):
                PPOTrainer(model,value_learning_rate=rate)
        for settings in ({'value_max_steps':3},{'value_learning_rate':.0001}):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError,'separate policy and value'):
                PPOTrainer(model.actor,**settings)

    def test_in_memory_value_limit_does_not_change_policy_shuffle(self):
        from patchwork_automaton.neural.ppo import PPOTrainer
        model, _, batch, _ = self.setup_model()
        other, _, _, _ = self.setup_model()
        other.set_weights(model.get_weights())
        old = np.asarray(tf.nn.log_softmax(model(batch)['logits']))[:,0]
        values = np.asarray(model(batch)['value'])[:,0]
        returns = np.array([1.,-1.],np.float32)
        rngs = [np.random.default_rng(821),np.random.default_rng(821)]
        for current,limit,rng in zip((model,other),(None,1),rngs):
            trainer = PPOTrainer(current,epochs=3,batch_size=1,value_max_steps=limit)
            trainer.update(batch,np.zeros(2,np.int32),returns,old,values,rng=rng)
        for a,b in zip(model.actor.variables,other.actor.variables): np.testing.assert_array_equal(a,b)
        self.assertEqual(rngs[0].bit_generator.state,rngs[1].bit_generator.state)

    def test_critic_learns_without_changing_actor_and_memories_are_independent(self):
        model, trainer, batch, _ = self.setup_model()
        self.assertFalse({id(v) for v in model.actor.variables} & {id(v) for v in model.critic.variables})
        actor_weights = [np.asarray(v).copy() for v in model.actor.variables]
        before = {k:np.asarray(v) for k,v in model(batch).items()}
        targets = np.array([1., -1.], np.float32)
        embedding = np.asarray(model.critic.card_embedding.embeddings).copy()
        initial = np.mean((before['value'][:, 0] - targets)**2)
        for _ in range(50): trainer.critic_step(batch, targets)
        after = {k:np.asarray(v) for k,v in model(batch).items()}
        self.assertLess(np.mean((after['value'][:, 0] - targets)**2), initial * .1)
        self.assertFalse(np.array_equal(embedding, model.critic.card_embedding.embeddings))
        for a,b in zip(actor_weights, model.actor.variables): np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(before['logits'], after['logits'])
        np.testing.assert_array_equal(before['memory_out'][:, :8], after['memory_out'][:, :8])
        changed = dict(batch, memory_in=batch['memory_in'].copy())
        changed['memory_in'][:, 8:] += 1.
        np.testing.assert_array_equal(model(batch)['logits'], model(changed)['logits'])
        paused = dict(batch, advance_memory=np.zeros((2,1),np.float32))
        np.testing.assert_array_equal(model(paused)['memory_out'], batch['memory_in'])

    def test_actor_update_and_rollback_preserve_independent_critic(self):
        model, trainer, batch, _ = self.setup_model()
        actor = [np.asarray(v).copy() for v in model.actor.variables]
        snapshot = trainer.snapshot()
        old = np.asarray(tf.nn.log_softmax(model(batch)['logits']))[:, 0]
        returns = np.array([1., -1.], np.float32)
        trainer.critic_step(batch, returns)
        critic = [np.asarray(v).copy() for v in model.critic.variables]
        critic_optimizer = [np.asarray(v).copy() for v in trainer.critic_optimizer.variables]
        trainer.step(batch, np.zeros(2,np.int32), returns, old, returns)
        self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(actor, model.actor.variables)))
        for a,b in zip(critic, model.critic.variables): np.testing.assert_array_equal(a,b)
        trainer.restore(snapshot)
        for a,b in zip(actor, model.actor.variables): np.testing.assert_array_equal(a,b)
        for a,b in zip(critic, model.critic.variables): np.testing.assert_array_equal(a,b)
        for a,b in zip(critic_optimizer, trainer.critic_optimizer.variables): np.testing.assert_array_equal(a,b)

    def test_checkpoint_restores_both_networks_optimizers_and_memories(self):
        model, trainer, batch, config = self.setup_model()
        returns = np.array([1., -1.], np.float32)
        old = np.asarray(tf.nn.log_softmax(model(batch)['logits']))[:, 0]
        trainer.critic_step(batch, returns)
        trainer.step(batch, np.zeros(2,np.int32), returns, old, returns)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory,model,trainer,dict(round=1,config=config),policy_type=RustMagePolicy)
            loaded, restored, _ = load_checkpoint(directory)
            self.assertEqual(loaded.memory_size,16)
            for key, value in model(batch).items(): np.testing.assert_allclose(value, loaded(batch)[key], atol=1e-6)
            for name in ('optimizer','critic_optimizer'):
                for a,b in zip(getattr(trainer,name).variables,getattr(restored,name).variables):
                    np.testing.assert_array_equal(a,b)
            self.assertEqual(trainer.rng.bit_generator.state, restored.rng.bit_generator.state)
            trainer.critic_step(batch, returns); restored.critic_step(batch, returns)
            np.testing.assert_allclose(model(batch)['value'],loaded(batch)['value'],atol=1e-6)
