"""Streamed value warmup must never optimize or roll back the cloned actor."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE, 'Requires neural dependencies')
class CriticWarmupTests(unittest.TestCase):
    def fixture(self, directory, *, cache_bytes=0, value_max_steps=1):
        import keras
        import numpy as np
        import tensorflow as tf
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.experiment import TrajectoryWriter
        from patchwork_automaton.neural.model import SeparateActorCritic
        from patchwork_automaton.neural.rustmage_features import RustMageEncoder
        from patchwork_automaton.neural.streaming_ppo import StreamingPPOTrainer
        from patchwork_automaton.neural.training import Sample
        keras.utils.set_random_seed(19)
        encoder = RustMageEncoder()
        model = SeparateActorCritic(encoder.card_count, encoder.feature_spec,
            actor_width=8, actor_memory=4, actor_blocks=0, critic_width=8, critic_memory=4)
        batch = {k: np.repeat(v, 2, axis=0)
                 for k,v in example_batch(model.memory_size, encoder.feature_spec).items()}
        for key in ('candidates','candidate_refs','candidate_mask'):
            batch[key] = np.repeat(batch[key], 2, axis=1)
        batch['candidates'][:, 1, 0] = 1.
        batch['entities'][:, 0, 0] = [1., -1.]
        model(batch)
        trainer = StreamingPPOTrainer(model, seed=29, batch_size=2, epochs=2,
            learning_rate=.001, value_max_steps=value_max_steps, cache_bytes=cache_bytes,
            concurrent_updates=True, prefetch_batches=2)
        # Nonzero existing Adam slots/counter make preservation stronger than
        # checking that an untouched, newly initialized optimizer stays zero.
        old = np.asarray(tf.nn.log_softmax(model(batch)['logits']))
        trainer.step(batch, np.array([0,1], np.int32), np.array([1.,-1.], np.float32),
                     old[[0,1],[0,1]], np.array([1.,-1.], np.float32))
        output = {k: np.asarray(v) for k,v in model(batch).items()}
        probabilities = np.asarray(tf.nn.log_softmax(output['logits']))
        trajectories = []
        for row,reward in enumerate((1., -1.)):
            path = Path(directory)/f'{row}.pkl.gz'
            writer = TrajectoryWriter(path)
            try:
                arrays = {k:v[row].copy() for k,v in batch.items()
                          if k not in ('memory_in','advance_memory')}
                for _ in range(4):
                    writer.write(Sample(arrays, batch['memory_in'][row].copy(), True, row,
                        float(output['value'][row,0]), probabilities[row].copy()))
            finally:
                writer.close()
            trajectories.append(dict(path=str(path), samples=4, reward=reward))
        return model,trainer,trajectories

    def test_critic_updates_preserve_actor_weights_adam_and_cap(self):
        import copy
        import numpy as np
        for cache_bytes in (0, 16*1024**2):
            with self.subTest(cache_bytes=cache_bytes), tempfile.TemporaryDirectory() as directory:
                model,trainer,trajectories = self.fixture(directory,cache_bytes=cache_bytes)
                actor = [np.asarray(v).copy() for v in model.actor.variables]
                adam = [np.asarray(v).copy() for v in trainer.optimizer.variables]
                critic = [np.asarray(v).copy() for v in model.critic.variables]
                rng = copy.deepcopy(trainer.rng.bit_generator.state)
                with patch.object(trainer,'step',side_effect=AssertionError('Actor step called')), \
                     patch.object(trainer,'snapshot',side_effect=AssertionError('Actor snapshot called')), \
                     patch.object(trainer,'restore',side_effect=AssertionError('Actor rollback called')):
                    result = trainer.warmup_critic_files(trajectories,max_steps=3)
                self.assertTrue(result['critic_only'])
                self.assertEqual(result['phase'],'critic_warmup')
                self.assertEqual(result['update_schedule'],'critic_only')
                self.assertEqual(result['accepted_steps'],0)
                self.assertEqual(result['accepted_actor_steps'],0)
                self.assertEqual(result['optimizer_steps'],1)
                self.assertEqual(result['critic_steps'],3)
                self.assertEqual(result['critic_optimizer_steps'],3)
                self.assertEqual(result['value_max_steps'],3)
                self.assertTrue(result['value_step_limit_reached'])
                self.assertEqual(trainer.value_max_steps,1)
                self.assertNotEqual(trainer.rng.bit_generator.state,rng)
                for before,after in zip(actor,model.actor.variables):
                    np.testing.assert_array_equal(before,np.asarray(after))
                for before,after in zip(adam,trainer.optimizer.variables):
                    np.testing.assert_array_equal(before,np.asarray(after))
                self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(critic,model.critic.variables)))
                self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in model.critic.variables))
                self.assertLessEqual(result['cache']['retained_bytes'],cache_bytes)
                self.assertEqual(result['cache']['spill_bytes']>0,cache_bytes==0)
                self.assertTrue(result['terminal_rewards_only'])

    def test_default_budget_then_normal_ppo_still_updates_actor(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            model,trainer,trajectories = self.fixture(directory)
            self.assertEqual(trainer.warmup_critic_files(trajectories)['critic_steps'],1)
            actor = [np.asarray(v).copy() for v in model.actor.variables]
            result = trainer.update_files(trajectories)
            self.assertNotIn('critic_only',result)
            self.assertGreater(result['accepted_steps'],0)
            self.assertEqual(result['optimizer_steps'],1+result['accepted_steps'])
            self.assertEqual(result['critic_steps'],1)
            self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(actor,model.actor.variables)))

    def test_rejects_shared_model_invalid_budget_and_restores_after_failure(self):
        from dataclasses import replace
        from patchwork_automaton.neural.experiment import read_trajectory, TrajectoryWriter
        from patchwork_automaton.neural.streaming_ppo import StreamingPPOTrainer
        with tempfile.TemporaryDirectory() as directory:
            model,trainer,trajectories = self.fixture(directory)
            shared = StreamingPPOTrainer(model.actor,seed=29,batch_size=2,epochs=1)
            with self.assertRaisesRegex(ValueError,'separate actor and critic'):
                shared.warmup_critic_files(trajectories,max_steps=2)
            for limit in (0,-1,1.5,True):
                with self.subTest(limit=limit), self.assertRaisesRegex(ValueError,'positive integer'):
                    trainer.warmup_critic_files(trajectories,max_steps=limit)
            with patch.object(trainer,'fit_critic',side_effect=RuntimeError('Failed critic')):
                with self.assertRaisesRegex(RuntimeError,'Failed critic'):
                    trainer.warmup_critic_files(trajectories,max_steps=3)
            self.assertEqual(trainer.value_max_steps,1)
            invalid = [dict(trajectories[0],reward=.25)]
            with self.assertRaisesRegex(ValueError,'terminal win/loss/draw'):
                trainer.warmup_critic_files(invalid,max_steps=2)
            self.assertEqual(trainer.value_max_steps,1)
            # Critic-only fitting still requires actual on-policy records,
            # rather than treating expert labels as fabricated PPO behavior.
            path = Path(directory)/'missing-behavior.pkl.gz'
            writer = TrajectoryWriter(path)
            try:
                for sample in read_trajectory(trajectories[0]['path']):
                    writer.write(replace(sample,log_probs=None))
            finally:
                writer.close()
            with self.assertRaisesRegex(ValueError,'behavior probabilities'):
                trainer.warmup_critic_files([dict(trajectories[0],path=str(path))],max_steps=2)
            self.assertEqual(trainer.value_max_steps,1)


if __name__ == '__main__':
    unittest.main()
