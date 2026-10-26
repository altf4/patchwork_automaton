"""One finite-gradient barrier must guard every model and Adam mutation."""
import unittest
from unittest.mock import patch
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE, 'Requires neural dependencies')
class GradientCheckTests(unittest.TestCase):
    def fixture(self, *, separate=True):
        import keras
        import numpy as np
        import tensorflow as tf
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.ppo import PPOTrainer

        class TinyNetwork(keras.Model):
            memory_size = 2
            def __init__(self, *, policy, value):
                super().__init__()
                self.hidden = keras.layers.Dense(4, activation='tanh')
                self.policy_head = keras.layers.Dense(2) if policy else None
                self.value_head = keras.layers.Dense(1) if value else None
                # Exercise the existing None-gradient → dense-zero path too.
                self.unused = self.add_weight(shape=(2,), initializer='ones', name='unused')

            def call(self, inputs, training=False):
                hidden = self.hidden(inputs['entities'][:, 0, :2])
                output = {}
                if self.policy_head is not None:
                    output['logits'] = self.policy_head(hidden)
                output['value'] = (self.value_head(hidden) if self.value_head is not None
                                   else tf.zeros_like(hidden[:, :1]))
                return output

        class TinySeparate(keras.Model):
            separate_value = True
            memory_size = 2
            deterministic_forward = True
            def __init__(self):
                super().__init__()
                self.actor = TinyNetwork(policy=True, value=False)
                self.critic = TinyNetwork(policy=False, value=True)

            def actor_output(self, inputs, training=False):
                return self.actor(inputs, training=training)

            def value_output(self, inputs, training=False):
                return self.critic(inputs, training=training)

            def call(self, inputs, training=False):
                return dict(self.actor_output(inputs, training=training),
                            value=self.value_output(inputs, training=training)['value'])

        keras.utils.set_random_seed(918)
        model = TinySeparate() if separate else TinyNetwork(policy=True, value=True)
        batch = {k: np.repeat(v, 3, axis=0) for k, v in example_batch(2).items()}
        for key in ('candidates', 'candidate_refs', 'candidate_mask'):
            batch[key] = np.repeat(batch[key], 2, axis=1)
        batch['entities'][:, 0, :2] = [[.2, -.4], [.8, .1], [-.3, .7]]
        model(batch)
        trainer = PPOTrainer(model, learning_rate=.001, value_learning_rate=.002 if separate else None,
                             epochs=2, batch_size=2, entropy_weight=.01)
        actions = np.array([0, 1, 0], np.int32)
        returns = np.array([1., -1., 1.], np.float32)
        output = model(batch)
        old = np.asarray(tf.nn.log_softmax(output['logits']))[np.arange(3), actions]
        values = np.asarray(output['value'])[:, 0]
        return model, trainer, batch, actions, returns, old, values

    @staticmethod
    def state(model, trainer):
        import numpy as np
        variables = list(model.variables) + list(trainer.optimizer.variables)
        if trainer.separate_value:
            variables += list(trainer.critic_optimizer.variables)
        return [np.asarray(v).copy() for v in variables]

    def test_empty_and_mixed_float_dtypes_have_one_assertion(self):
        import numpy as np
        import tensorflow as tf
        from patchwork_automaton.neural.ppo import _assert_finite_gradients
        for dtype in (tf.float16, tf.bfloat16, tf.float32, tf.float64):
            with self.subTest(dtype=dtype.name):
                self.assertTrue(bool(_assert_finite_gradients(
                    [tf.zeros((0, 2), dtype), tf.zeros((2,), dtype)], 'finite test')))
                for invalid in (np.nan, np.inf, -np.inf):
                    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'finite test'):
                        _assert_finite_gradients([tf.constant([invalid], dtype)], 'finite test')
        self.assertTrue(bool(_assert_finite_gradients([], 'empty')))
        @tf.function(autograph=False)
        def mixed(a, b):
            return _assert_finite_gradients([a, b], 'mixed')
        graph = mixed.get_concrete_function(tf.TensorSpec([None], tf.float32),
                                           tf.TensorSpec([None], tf.float64)).graph
        self.assertEqual(sum(op.type == 'Assert' for op in graph.get_operations()), 1)
        self.assertFalse(any(op.type.startswith('CheckNumerics') for op in graph.get_operations()))

    def test_finite_updates_match_legacy_checks_weights_both_adams_and_rng(self):
        import numpy as np
        import tensorflow as tf
        from patchwork_automaton.neural import ppo

        def legacy(gradients, message):
            checked = [tf.debugging.assert_all_finite(g, message) for g in gradients]
            with tf.control_dependencies(checked):
                return tf.constant(True)

        for separate in (False, True):
            with self.subTest(separate=separate):
                candidate = self.fixture(separate=separate)
                reference = self.fixture(separate=separate)
                rngs = [np.random.default_rng(79), np.random.default_rng(79)]
                results = []
                for fixture, rng, check in ((candidate, rngs[0], ppo._assert_finite_gradients),
                                             (reference, rngs[1], legacy)):
                    model, trainer, inputs, actions, returns, old, values = fixture
                    with patch.object(ppo, '_assert_finite_gradients', check):
                        # Two rollouts also compare already-populated Adam slots.
                        results.append([trainer.update(inputs, actions, returns, old, values, rng=rng)
                                        for _ in range(2)])
                for a, b in zip(self.state(*candidate[:2]), self.state(*reference[:2])):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(results[0], results[1])
                self.assertEqual(rngs[0].bit_generator.state, rngs[1].bit_generator.state)

    def test_one_assertion_precedes_all_optimizer_writes_and_invalid_updates_are_atomic(self):
        import numpy as np
        import tensorflow as tf
        for kind in ('shared_actor', 'separate_actor', 'critic'):
            model, trainer, inputs, actions, returns, old, values = self.fixture(separate=kind != 'shared_actor')
            arguments = (inputs, returns) if kind == 'critic' else (inputs, actions, returns, old, returns)
            original = trainer.critic_step if kind == 'critic' else trainer.step
            # Start from nonzero moments and iteration counters, not fresh slots.
            original(*arguments)
            for position in (0, -1):
                with self.subTest(kind=kind, position=position):
                    bad = tf.Variable(0., trainable=False)
                    gradient = tf.GradientTape.gradient
                    def injected(tape, *args, **kwargs):
                        gradients = list(gradient(tape, *args, **kwargs))
                        present = [i for i, g in enumerate(gradients) if g is not None]
                        selected = present[position]
                        g = gradients[selected]
                        gradients[selected] = tf.where(tf.math.is_finite(bad), g,
                            tf.fill(tf.shape(g), tf.cast(bad, g.dtype)))
                        return gradients
                    method = trainer._critic_step if kind == 'critic' else trainer._step
                    compiled = tf.function(method, input_signature=original.input_signature, autograph=False)
                    with patch.object(tf.GradientTape, 'gradient', injected):
                        concrete = compiled.get_concrete_function()
                    ops = concrete.graph.get_operations()
                    assertions = [op for op in ops if op.type == 'Assert']
                    self.assertEqual(len(assertions), 1)
                    self.assertFalse(any(op.type.startswith('CheckNumerics') for op in ops))
                    def depends_on_assertion(op):
                        pending, seen = [op], set()
                        while pending:
                            current = pending.pop()
                            if current in seen: continue
                            if current is assertions[0]: return True
                            seen.add(current)
                            pending.extend(current.control_inputs)
                            pending.extend(t.op for t in current.inputs)
                        return False
                    writes = [op for op in ops if op.type.startswith(('Assign', 'ResourceApply'))]
                    self.assertGreater(len(writes), 2)
                    for write in writes:
                        self.assertTrue(depends_on_assertion(write), write.name)
                    before = self.state(model, trainer)
                    for invalid in (np.nan, np.inf, -np.inf):
                        bad.assign(invalid)
                        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'Nonfinite'):
                            compiled(*arguments)
                        for a, b in zip(before, self.state(model, trainer)):
                            np.testing.assert_array_equal(a, b)
                    # The same traced graph can still perform valid updates.
                    bad.assign(0.)
                    self.assertTrue(np.isfinite(compiled(*arguments)))
                    self.assertTrue(any(not np.array_equal(a, b)
                                        for a, b in zip(before, self.state(model, trainer))))


if __name__ == '__main__': unittest.main()
