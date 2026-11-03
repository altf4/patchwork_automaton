"""CPU neural smoke coverage using synthetic records, never an engine/game."""
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('numpy', 'keras', 'tensorflow'))


@unittest.skipUnless(AVAILABLE, 'Requires the project neural extra')
class HartfordNeuralTests(unittest.TestCase):
    def test_random_checkpoint_roundtrip_ppo_and_readonly_holdout(self):
        if os.environ.get('PATCHWORK_HARTFORD_NEURAL_SMOKE_CHILD') != '1':
            # Keep neural imports out of the test supervisor just as the
            # production game-worker supervisor remains TensorFlow-free.
            project = Path(__file__).resolve().parents[1]
            with tempfile.TemporaryDirectory() as cache:
                env = dict(os.environ, PATCHWORK_HARTFORD_NEURAL_SMOKE_CHILD='1',
                    CUDA_VISIBLE_DEVICES='-1', TF_NUM_INTRAOP_THREADS='2',
                    TF_NUM_INTEROP_THREADS='1', MPLCONFIGDIR=cache,
                    PYTHONPATH=os.pathsep.join((str(project), str(project / 'src'),
                                               os.environ.get('PYTHONPATH', ''))))
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    'HartfordNeuralTests.test_random_checkpoint_roundtrip_ppo_and_readonly_holdout', '-v'],
                    cwd=project, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return
        import keras
        import numpy as np
        import tensorflow as tf
        from examples import train_hartford_ppo as driver
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.devices import assert_model_device
        from patchwork_automaton.neural.experiment import load_checkpoint, save_checkpoint
        from patchwork_automaton.neural.inference_processes import weights_digest
        from patchwork_automaton.neural.rollout_records import Sample, TrajectoryWriter
        from patchwork_automaton.neural.standard_policy import StandardPolicy
        from patchwork_automaton.neural.validation import GameValidator, training_normalization

        config = driver.configuration(driver.parser().parse_args(['--preflight-only','--architecture','legacy']))
        # Retain the production architecture and optimizer hyperparameters;
        # bound this synthetic test to one actor and one critic minibatch.
        config.update(train_batch=2, ppo_epochs=1, value_max_steps=1,
                      concurrent_updates=False, prefetch_batches=0, measurement_workers=1)
        was_eager = tf.config.functions_run_eagerly()
        tf.config.run_functions_eagerly(True)
        self.addCleanup(tf.config.run_functions_eagerly, was_eager)
        self.addCleanup(keras.backend.clear_session)

        with tf.device('/CPU:0'), patch.dict(os.environ, PATCHWORK_ROLLOUT_STORAGE='pickle'):
            model, trainer = driver.initialize_random(config, device='cpu')
            assert_model_device(model, 'cpu')
            self.assertEqual(model.memory_size, 96)
            self.assertTrue(model.separate_value)
            self.assertFalse({id(v) for v in model.actor.variables} & {id(v) for v in model.critic.variables})
            for network in (model.actor, model.critic):
                self.assertTrue(any(np.any(np.asarray(v)) for v in network.trainable_variables))
            for optimizer in (trainer.optimizer, trainer.critic_optimizer):
                self.assertEqual(int(optimizer.iterations), 0)
                self.assertGreater(len(optimizer.variables), 2)
                for slot in optimizer.variables[2:]:
                    self.assertFalse(np.any(np.asarray(slot)))
            initial = driver.training_state(model, trainer)
            other, other_trainer = driver.initialize_random(config, device='cpu')
            self.assertEqual(driver.training_state(other, other_trainer), initial)
            del other, other_trainer
            gc.collect()

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = save_checkpoint(root / 'checkpoints', model, trainer,
                    dict(round=0, config=config), policy_type=StandardPolicy)
                saved = json.loads((checkpoint / 'state.json').read_text())
                self.assertEqual((saved['round'], saved['optimizer_steps'], saved['critic_optimizer_steps']), (0, 0, 0))
                loaded, restored, state = load_checkpoint(root / 'checkpoints',
                    policy_type=StandardPolicy, gru_backend='unfused')
                assert_model_device(loaded, 'cpu')
                self.assertEqual(driver.training_state(loaded, restored), initial)
                self.assertEqual(state['trainer_rng_state'], trainer.rng.bit_generator.state)
                self.assertEqual(state['config'], config)
                for expected, actual in zip(model.get_weights(), loaded.get_weights(), strict=True):
                    np.testing.assert_array_equal(actual, expected)
                for name in ('optimizer', 'critic_optimizer'):
                    expected = getattr(trainer, name).variables
                    actual = getattr(restored, name).variables
                    for a, b in zip(expected, actual, strict=True):
                        np.testing.assert_array_equal(a, b)
                # Compare the next shuffle as well as the stored PCG state.
                np.testing.assert_array_equal(trainer.rng.permutation(11), restored.rng.permutation(11))
                del model, trainer
                gc.collect()

                games = []
                for index, reward in enumerate((1., -1., 1.)):
                    batch = example_batch(loaded.memory_size, loaded.feature_spec)
                    for key in ('candidates', 'candidate_refs', 'candidate_mask'):
                        batch[key] = np.repeat(batch[key], 2, axis=1)
                    batch['candidates'][:, 1, 0] = 1.
                    batch['entities'][:, 0, 0] = float(index + 1)
                    output = loaded(batch, training=False)
                    arrays = {k: v[0].copy() for k, v in batch.items()
                              if k not in ('memory_in', 'advance_memory')}
                    logp = np.asarray(tf.nn.log_softmax(output['logits']))[0]
                    writer = TrajectoryWriter(root / f'synthetic-{index}.pkl.gz')
                    try:
                        writer.write(Sample(arrays, batch['memory_in'][0].copy(), True,
                            index % 2, float(output['value'][0, 0]), logp.copy()))
                    finally:
                        writer.close()
                    games.append(dict(index=index, seed=100 + index,
                        partition='training' if index < 2 else 'validation',
                        trajectories=[dict(path=str(writer.path), samples=1, reward=reward)]))
                training, heldout = games[:2], games[2:]
                normalization = training_normalization(training, config['advantage_scale_floor'])
                self.assertEqual(normalization['training_choices'], 2)
                before_actor = weights_digest(loaded.actor.get_weights())
                before_critic = weights_digest(loaded.critic.get_weights())
                result = restored.update_files([t for g in training for t in g['trajectories']])
                self.assertEqual(result['samples'], 2)
                self.assertGreater(result['accepted_steps'], 0)
                self.assertEqual(int(restored.optimizer.iterations), 1)
                self.assertEqual(int(restored.critic_optimizer.iterations), 1)
                self.assertNotEqual(weights_digest(loaded.actor.get_weights()), before_actor)
                self.assertNotEqual(weights_digest(loaded.critic.get_weights()), before_critic)
                before_holdout = driver.training_state(loaded, restored)
                validation = GameValidator(loaded, batch_size=2).evaluate(heldout, normalization)
                self.assertEqual(validation['attempted_games'], 1)
                self.assertEqual(validation['terminal_games'], 1)
                self.assertEqual(validation['games'][0]['index'], 2)
                self.assertTrue(np.isfinite(validation['mean_policy_kl']))
                self.assertEqual(driver.training_state(loaded, restored), before_holdout)
                # Also exercise restore with nonzero Adam moments and an
                # advanced shuffle RNG, which a fresh-only roundtrip cannot.
                save_checkpoint(root / 'checkpoints', loaded, restored,
                    dict(round=1, config=config), policy_type=StandardPolicy)
                del loaded, restored
                gc.collect()
                resumed, resumed_trainer, resumed_state = load_checkpoint(root / 'checkpoints',
                    policy_type=StandardPolicy, gru_backend='unfused')
                self.assertEqual(resumed_state['round'], 1)
                self.assertEqual(driver.training_state(resumed, resumed_trainer), before_holdout)
                self.assertEqual(int(resumed_trainer.optimizer.iterations), 1)
                self.assertEqual(int(resumed_trainer.critic_optimizer.iterations), 1)


if __name__ == '__main__':
    unittest.main()
