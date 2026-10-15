"""Terminal reward signs and actual policy-gradient direction."""
import unittest
from types import SimpleNamespace
from test_neural import AVAILABLE, observation

if AVAILABLE:
    import numpy as np
    import keras
    from keras import ops
    from patchwork_automaton.neural.features import Encoder
    from patchwork_automaton.neural.training import Sample, TerminalActorCritic, terminal_reward, RolloutPolicy
    from patchwork_automaton.neural.policy import untrained_policy


@unittest.skipUnless(AVAILABLE, 'Install neural dependencies with uv')
class TrainingTests(unittest.TestCase):
    def test_only_verified_terminal_outcomes_get_rewards(self):
        for won, draw, expected in ((True, None, 1), (False, None, -1), (None, True, 0)):
            self.assertEqual(terminal_reward(SimpleNamespace(won=won, draw=draw)), expected)
        for value in (None, SimpleNamespace(won=None, draw=None)):
            with self.assertRaises(ValueError): terminal_reward(value)

    def test_win_increases_and_loss_decreases_selected_action_probability(self):
        class Toy(keras.Model):
            def __init__(self):
                super().__init__()
                self.scores = self.add_weight(shape=(2,), initializer='zeros')
                self.value = self.add_weight(shape=(1,), initializer='zeros')
            def call(self, inputs, training=False):
                count = ops.shape(inputs['entities'])[0]
                return {'logits': ops.broadcast_to(self.scores[None], (count, 2)),
                        'value': ops.broadcast_to(self.value[None], (count, 1))}
        arrays = Encoder().encode(observation()).arrays
        for reward in (1., -1., 0.):
            model = Toy()
            trainer = TerminalActorCritic(model, learning_rate=.01, batch_size=1)
            sample = Sample(arrays, np.zeros(2, np.float32), True, 0, 0.)
            result = trainer.update([([sample, sample], reward)])
            difference = float(model.scores[0] - model.scores[1])
            if reward: self.assertGreater(difference * reward, 0)
            else: self.assertEqual(difference, 0)
            # Two minibatches must still produce exactly one on-policy update.
            self.assertEqual(result['optimizer_steps'], 1)
            self.assertTrue(np.isfinite(result['loss']))

    def test_rollout_copies_model_inputs_and_reset_discards_them(self):
        initial = untrained_policy(seed=1, width=16, memory_size=16, blocks=1)
        policy = RolloutPolicy(initial.model, initial.encoder, seed=1)
        record = policy.act(observation())
        self.assertEqual(len(policy.trajectory), len(record.selections))
        sample = policy.trajectory[0]
        self.assertEqual(sample.action, record.selections[0].index)
        self.assertTrue(sample.arrays['candidate_mask'][sample.action])
        self.assertEqual(sample.memory.shape, (16,))
        policy.reset()
        self.assertEqual(policy.trajectory, [])
