"""Backend-only GRU changes must preserve masking, weights and derivatives."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE, 'Requires neural dependencies')
class FusedGRUTests(unittest.TestCase):
    def test_training_backend_defaults_and_cpu_rejection(self):
        from examples.train_rustmage import parser, resolve_learner_gru_backend
        args = parser().parse_args(['--output', 'unused'])
        self.assertEqual(resolve_learner_gru_backend(args), 'unfused')
        args.learner_device = 'gpu'
        self.assertEqual(resolve_learner_gru_backend(args), 'cudnn-memory')
        args.learner_gru_backend = 'unfused'
        self.assertEqual(resolve_learner_gru_backend(args), 'unfused')
        args.learner_gru_backend = 'cudnn-memory'; args.learner_device = 'cpu'
        with self.assertRaisesRegex(ValueError, 'GPU learner'):
            resolve_learner_gru_backend(args)

    def test_legacy_gpu_resume_selects_backend_before_loading_without_recipe_change(self):
        import asyncio
        from examples import train_rustmage as runner
        args = runner.parser().parse_args(['--output', 'unused', '--resume', '--learner-device', 'gpu'])
        config = {k:v for k,v in vars(args).items() if k not in
                  ('output','resume','rounds','plot','initialize_from','learner_gru_backend')}
        config.update(inference_processes=2, algorithm='ppo', advantage_scale_floor=1., ppo_clip=.2)
        state = dict(round=0, config=config, initial_report=dict(rounds=[],evaluations=[]))
        class ReachedRuntime(Exception): pass
        with patch.object(runner,'load_checkpoint',return_value=(None,None,state)) as load, \
                patch.object(runner,'assert_model_device',side_effect=ReachedRuntime):
            with self.assertRaises(ReachedRuntime): asyncio.run(runner.run_experiment(args))
            self.assertEqual(load.call_args.kwargs, {'gru_backend':'cudnn-memory'})
        self.assertNotIn('learner_gru_backend', config)

    def test_backend_is_selected_before_compile_and_both_adams_and_rng_restore(self):
        import hashlib
        import numpy as np
        from test_critic_warmup import CriticWarmupTests
        from patchwork_automaton.neural import experiment
        from patchwork_automaton.neural.recurrent import gru_backend_summary
        from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            model, trainer, _ = CriticWarmupTests().fixture(root/'data', cache_bytes=16*1024**2, value_max_steps=2)
            config = dict(algorithm='ppo', learning_rate=.001, train_batch=2, ppo_epochs=2,
                          seed=19, value_max_steps=2, concurrent_updates=True,
                          prefetch_batches=2, rollout_cache_mib=16)
            experiment.save_checkpoint(root/'checkpoints',model,trainer,dict(round=0,config=config),policy_type=RustMagePolicy)
            original = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (root/'checkpoints').rglob('*') if p.is_file()}
            make = experiment.make_trainer
            def checked_make(loaded, config):
                self.assertTrue(all(r['use_cudnn']=='auto' for r in gru_backend_summary(loaded)))
                return make(loaded,config)
            # CPU 'auto' exercises the same backend-before-compile ordering
            # without requiring a GPU in the portable regression suite.
            with patch.object(experiment,'make_trainer',side_effect=checked_make):
                loaded, restored, state = experiment.load_checkpoint(root/'checkpoints',gru_backend='auto')
            for left,right in [(model.weights,loaded.weights),(trainer.optimizer.variables,restored.optimizer.variables),
                               (trainer.critic_optimizer.variables,restored.critic_optimizer.variables)]:
                self.assertEqual(len(left),len(right))
                for a,b in zip(left,right):np.testing.assert_array_equal(a,b)
            self.assertEqual(trainer.rng.bit_generator.state,restored.rng.bit_generator.state)
            self.assertEqual(state['config'],config)
            portable,_,_ = experiment.load_checkpoint(root/'checkpoints')
            self.assertTrue(all(r['use_cudnn'] is False for r in gru_backend_summary(portable)))
            self.assertEqual(original,{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (root/'checkpoints').rglob('*') if p.is_file()})

    def test_empty_and_right_padded_rows_outputs_and_gradients(self):
        import keras
        import numpy as np
        import tensorflow as tf
        from patchwork_automaton.neural.recurrent import masked_gru
        keras.utils.set_random_seed(208)
        # 'auto' exercises the empty-row adapter on CPU, with exactly the same
        # underlying cell equations. The separate GPU benchmark requires True.
        a = keras.layers.GRU(5, use_cudnn=False)
        b = keras.layers.GRU(5, use_cudnn='auto')
        x = tf.constant(np.random.default_rng(2).normal(size=(4, 5, 3)), tf.float32)
        initial = tf.constant(np.random.default_rng(3).normal(size=(4, 5)), tf.float32)
        masks = [np.arange(5)[None, :] < np.array([0, 1, 3, 5])[:, None],
                 np.zeros((4, 5), bool)]
        a(x); b(x); b.set_weights(a.get_weights())
        def evaluate(layer, mask):
            with tf.GradientTape() as tape:
                tape.watch([x, initial])
                y = masked_gru(layer, x, mask=mask, initial_state=initial, training=True)
                # Match the model's empty-history restoration as well.
                memory = tf.where(tf.reduce_any(mask, axis=1, keepdims=True), y, initial)
                loss = tf.reduce_sum(y * y) + tf.reduce_sum(memory * .3)
            return y, memory, tape.gradient(loss, [x, initial] + layer.trainable_variables)
        for mask in masks:
            with self.subTest(mask=mask.tolist()):
                ya, ma, ga = evaluate(a, mask)
                yb, mb, gb = evaluate(b, mask)
                for left, right in [(ya, yb), (ma, mb), *zip(ga, gb)]:
                    np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)
                np.testing.assert_array_equal(yb.numpy()[~mask.any(axis=1)], 0.)
                np.testing.assert_array_equal(gb[0].numpy()[~mask.any(axis=1)], 0.)

    def test_checkpoint_weights_and_default_cpu_backend_survive_roundtrip(self):
        import keras
        import numpy as np
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.model import SeparateActorCritic
        from patchwork_automaton.neural.recurrent import configure_gru_backend
        from patchwork_automaton.neural.rustmage_features import RustMageEncoder
        encoder = RustMageEncoder()
        model = SeparateActorCritic(encoder.card_count, encoder.feature_spec,
            actor_width=8, actor_memory=4, actor_blocks=0, critic_width=8, critic_memory=4)
        batch = example_batch(model.memory_size, encoder.feature_spec)
        expected = model(batch)
        weights = [np.asarray(v).copy() for v in model.weights]
        config = model.get_config()
        selective = configure_gru_backend(model, 'cudnn-memory')
        self.assertEqual(sum(x['use_cudnn'] is True for x in selective), 4)
        self.assertIs(model.actor.semantic_gru.use_cudnn, False)
        self.assertIs(model.critic.semantic_gru.use_cudnn, False)
        changed = configure_gru_backend(model, 'auto')
        self.assertEqual(len(changed), 6)
        self.assertEqual(config, model.get_config())
        actual = model(batch)
        for key in expected:
            np.testing.assert_allclose(expected[key], actual[key], rtol=1e-5, atol=1e-6)
        for a, b in zip(weights, model.weights): np.testing.assert_array_equal(a, b)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'policy.keras'; model.save(path)
            restored = keras.models.load_model(path)
            self.assertTrue(all(x['use_cudnn'] is False for x in configure_gru_backend(restored, 'unfused')))
            for a, b in zip(weights, restored.weights): np.testing.assert_array_equal(a, b)
            for key, value in restored(batch).items():
                np.testing.assert_allclose(expected[key], value, rtol=1e-5, atol=1e-6)
        with self.assertRaises(ValueError): configure_gru_backend(model, 'unknown')


if __name__ == '__main__': unittest.main()
