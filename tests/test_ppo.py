"""PPO clipping, behavior statistics, legal masks and actual update direction."""
import unittest
from test_neural import AVAILABLE

if AVAILABLE:
    import numpy as np
    import tensorflow as tf
    import keras
    from patchwork_automaton.neural.ppo import PPOTrainer, clipped_surrogate, normalized_advantages
    from patchwork_automaton.neural.batching import example_batch


@unittest.skipUnless(AVAILABLE, 'Install neural dependencies with uv')
class PPOTests(unittest.TestCase):
    def test_small_critic_errors_do_not_become_unit_scale_advantages(self):
        result = normalized_advantages(np.ones(4), np.array([.999, .998, .997, -1.]),
                                       np.array([True, True, True, False]), .1)
        np.testing.assert_allclose(result[:3], [-.01, 0., .01], atol=1e-6)
        np.testing.assert_array_equal(normalized_advantages([1., 1.], [0., 0.], [True, True], .1), [1., 1.])

    def test_clipping_respects_both_advantage_signs(self):
        ratios = tf.constant([1.5, .5, 1.5, .5])
        advantage = tf.constant([1., 1., -1., -1.])
        with tf.GradientTape() as tape:
            tape.watch(ratios)
            objective = clipped_surrogate(tf.math.log(ratios), tf.zeros(4), advantage, .2)
        np.testing.assert_allclose(objective, [1.2, .5, -1.5, -.8])
        np.testing.assert_allclose(tape.gradient(objective, ratios), [0., 1., -1., 0.])

    def test_updates_reuse_frozen_behavior_and_stop_on_kl(self):
        class Toy(keras.Model):
            memory_size = 2
            def __init__(self):
                super().__init__()
                self.scores = self.add_weight(shape=(2,), initializer='zeros')
                self.value = self.add_weight(shape=(1,), initializer='zeros')
            def call(self, inputs, training=False):
                n = tf.shape(inputs['entities'])[0]
                return dict(logits=tf.broadcast_to(self.scores[None], (n, 2)),
                            value=tf.broadcast_to(self.value[None], (n, 1)))
        inputs = example_batch(2)
        inputs['candidate_mask'] = np.ones((1, 2), bool)
        for reward in (1., -1.):
            model = Toy()
            trainer = PPOTrainer(model, learning_rate=.1, epochs=5, target_kl=1e-6)
            old_logp = np.array([-np.log(2)], np.float32)
            before = old_logp.copy()
            result = trainer.update(inputs, [0], [reward], old_logp, [0.], rng=np.random.default_rng(1))
            self.assertGreater(float(model.scores[0] - model.scores[1]) * reward, 0.)
            self.assertEqual(result['steps'], 1)
            self.assertTrue(result['early_stop'])
            np.testing.assert_array_equal(old_logp, before)
            with self.assertRaises(ValueError):
                trainer.update(inputs, [2], [reward], old_logp, [0.], rng=np.random.default_rng(1))

    def test_multiple_epochs_keep_original_probability_and_advantage(self):
        class Toy(keras.Model):
            memory_size = 2
            def __init__(self):
                super().__init__()
                self.scores = self.add_weight(shape=(2,), initializer='zeros')
                self.value = self.add_weight(shape=(1,), initializer='zeros')
            def call(self, inputs, training=False):
                return dict(logits=self.scores[None], value=self.value[None])
        inputs = example_batch(2)
        inputs['candidate_mask'] = np.ones((1, 2), bool)
        model = Toy()
        trainer = PPOTrainer(model, learning_rate=.2, epochs=4, target_kl=100., entropy_weight=0.)
        result = trainer.update(inputs, [0], [1.], [-np.log(2)], [0.], rng=np.random.default_rng(3))
        self.assertEqual(result['steps'], 4)
        self.assertFalse(result['early_stop'])
        self.assertGreater(float(model.value[0]), 0.)
        self.assertEqual(result['clip_fraction'], 1.)
        # The original advantage is still one despite the improving critic.
        self.assertAlmostEqual(result['policy_loss'], -1.2, places=5)

    def test_forced_choice_has_no_actor_gradient(self):
        class Forced(keras.Model):
            memory_size = 2
            def __init__(self):
                super().__init__()
                self.score = self.add_weight(shape=(1,), initializer='zeros')
            def call(self, inputs, training=False):
                return dict(logits=self.score[None], value=tf.zeros((1, 1)))
        model = Forced()
        trainer = PPOTrainer(model, epochs=2)
        result = trainer.update(example_batch(2), [0], [1.], [0.], [0.], rng=np.random.default_rng(2))
        self.assertEqual(float(model.score[0]), 0.)
        self.assertEqual(result['policy_loss'], 0.)
        self.assertEqual(result['approx_kl'], 0.)
        self.assertEqual(result['steps'], 2)

    def test_rejected_epoch_restores_weights_and_adam_state(self):
        class Toy(keras.Model):
            memory_size = 2
            def __init__(self):
                super().__init__()
                self.scores = self.add_weight(shape=(2,), initializer='zeros')
                self.value = self.add_weight(shape=(1,), initializer='zeros')
            def call(self, inputs, training=False):
                return dict(logits=self.scores[None], value=self.value[None])
        inputs=example_batch(2);inputs['candidate_mask']=np.ones((1,2),bool)
        model=Toy();trainer=PPOTrainer(model,learning_rate=.2,epochs=4,trust_guard=True,
                                      target_kl=1e-12,max_state_kl=1e-12,backtracks=1)
        variables=list(model.trainable_variables)+list(trainer.optimizer.variables)
        saved=[np.asarray(v).copy() for v in variables]
        result=trainer.update(inputs,[0],[1.],[-np.log(2)],[0.],rng=np.random.default_rng(1),
                              old_log_probs=np.full((1,2),-np.log(2),np.float32))
        self.assertEqual(result['steps'],0)
        self.assertEqual(result['rejected_epochs'],2)
        for variable,value in zip(variables,saved):np.testing.assert_array_equal(np.asarray(variable),value)

    def test_guarded_epoch_backtracks_until_distribution_is_within_limits(self):
        class Toy(keras.Model):
            memory_size = 2
            def __init__(self):
                super().__init__()
                self.scores = self.add_weight(shape=(2,), initializer='zeros')
            def call(self, inputs, training=False):
                return dict(logits=self.scores[None],value=tf.zeros((1,1)))
        inputs=example_batch(2);inputs['candidate_mask']=np.ones((1,2),bool)
        model=Toy();trainer=PPOTrainer(model,learning_rate=.2,epochs=1,trust_guard=True,
                                      target_kl=.001,max_state_kl=.002,backtracks=6)
        result=trainer.update(inputs,[0],[1.],[-np.log(2)],[0.],rng=np.random.default_rng(1),
                              old_log_probs=np.full((1,2),-np.log(2),np.float32))
        self.assertEqual(result['steps'],1)
        self.assertGreater(result['rejected_epochs'],0)
        self.assertLessEqual(result['exact_kl'],.001)
        self.assertLessEqual(result['max_state_kl'],.002)
        self.assertGreater(float(model.scores[0]-model.scores[1]),0.)
