"""Full-game PPO uses frozen behavior probabilities and terminal outcomes."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE,'Install neural dependencies with uv')
class StreamingPPOTests(unittest.TestCase):
    def test_value_diagnostics_distinguish_constant_bias_from_prediction(self):
        from patchwork_automaton.neural.ppo import value_prediction_diagnostics
        constant = value_prediction_diagnostics([1,1,1,-1,100], [.5,.5,.5,.5,100], [1,1,1,1,0])
        self.assertEqual(constant['samples'], 4)
        self.assertEqual(constant['mse'], .75)
        self.assertEqual(constant['zero_baseline_mse'], 1.)
        self.assertEqual(constant['constant_baseline_mse'], .75)
        self.assertEqual(constant['prediction_std'], 0.)
        self.assertEqual(constant['explained_variance'], 0.)
        perfect = value_prediction_diagnostics([1,-1], [1,-1], [1,1])
        self.assertEqual(perfect['mse'], 0.)
        self.assertEqual(perfect['explained_variance'], 1.)
        self.assertIsNone(value_prediction_diagnostics([1,1], [0,0], [1,1])['explained_variance'])

    def test_global_shuffle_mixes_games_and_preserves_sample_alignment(self):
        import numpy as np
        from patchwork_automaton.neural.streaming_ppo import StreamingPPOTrainer
        from patchwork_automaton.neural.experiment import read_trajectory, TrajectoryWriter
        from patchwork_automaton.neural.training import Sample
        from patchwork_automaton.neural.ppo import normalized_advantages
        with tempfile.TemporaryDirectory() as directory:
            model, originals = self.fixture(directory)
            trajectories = []
            expected = {}
            for game, original in enumerate(originals):
                template = next(read_trajectory(original['path']))
                path = Path(directory)/f'long-{game}.pkl.gz'
                writer = TrajectoryWriter(path)
                try:
                    for position in range(96):
                        identity = game*96 + position
                        memory = template.memory.copy(); memory[0] = identity
                        sample = Sample(template.arrays, memory, True, template.action,
                            identity / 1000., template.log_probs)
                        writer.write(sample)
                        expected[identity] = sample, original['reward']
                finally: writer.close()
                trajectories.append(dict(path=str(path), samples=96, reward=original['reward']))
            returns = np.array([expected[i][1] for i in range(192)], np.float32)
            values = np.array([expected[i][0].value for i in range(192)], np.float32)
            choices = np.array([expected[i][0].arrays['candidate_mask'].sum()>1 for i in range(192)])
            advantages = normalized_advantages(returns, values, choices, 1.)
            orders = []
            for cache_bytes in (0, 16*1024**2):
                trainer = StreamingPPOTrainer(model, seed=21, epochs=2, batch_size=16,
                    cache_bytes=cache_bytes, target_kl=100., max_state_kl=100., advantage_scale_floor=1.)
                seen = {'critic': [], 'actor': []}; pure = []
                def capture(inputs, rewards, kind):
                    ids = np.asarray(inputs['memory_in'])[:,0].astype(int)
                    np.testing.assert_array_equal(rewards, returns[ids])
                    seen[kind].extend(ids.tolist())
                    if kind=='critic': pure.append(len(set(rewards))==1)
                    return ids
                def critic_batches(batches):
                    for _ in range(2):
                        for inputs, rewards in batches(): capture(inputs, rewards, 'critic')
                    return {}
                def actor_step(inputs, actions, rewards, chosen, adv):
                    ids = capture(inputs, rewards, 'actor')
                    np.testing.assert_allclose(adv, advantages[ids], atol=1e-7)
                    for row, identity in enumerate(ids):
                        sample = expected[identity][0]
                        self.assertEqual(actions[row], sample.action)
                        self.assertEqual(chosen[row], sample.log_probs[sample.action])
                    return 0.
                trainer.fit_critic = critic_batches
                trainer.step = actor_step
                result = trainer.update_files(trajectories)
                self.assertEqual(result['rejected_epochs'], 0)
                for kind in seen:
                    for epoch in range(2):
                        self.assertEqual(sorted(seen[kind][epoch*192:(epoch+1)*192]), list(range(192)))
                self.assertLess(sum(pure), len(pure)*.1)
                self.assertEqual(result['sample_order'], 'global_shuffle_v1')
                self.assertEqual(result['cache']['spill_bytes'] == 0, cache_bytes != 0)
                orders.append((seen, trainer.rng.bit_generator.state))
            self.assertEqual(orders[0], orders[1])

    def test_cached_shared_validation_matches_uncached_separate_passes(self):
        import numpy as np
        from patchwork_automaton.neural.streaming_ppo import StreamingPPOTrainer
        from patchwork_automaton.neural.experiment import read_trajectory
        with tempfile.TemporaryDirectory() as directory:
            model, trajectories = self.fixture(Path(directory)/'first')
            other, _ = self.fixture(Path(directory)/'second')
            other.set_weights(model.get_weights())
            options = dict(seed=21, batch_size=2, epochs=2, learning_rate=.0001,
                advantage_scale_floor=1., target_kl=.01, max_state_kl=.05)
            uncached = StreamingPPOTrainer(model, cache_bytes=0, **options)
            cached = StreamingPPOTrainer(other, cache_bytes=16*1024**2, **options)
            uncached.compile(); cached.compile()
            # Keep the former two-forward validation as an independent oracle.
            uncached.measure_with_drift = lambda *args: (uncached.measure(*args[:-1]),
                                                        uncached.drift(args[0], args[-1]))
            with patch('patchwork_automaton.neural.streaming_ppo.read_trajectory', wraps=read_trajectory) as reads:
                reference = uncached.update_files(trajectories)
                uncached_reads = reads.call_count
                reads.reset_mock()
                result = cached.update_files(trajectories)
                self.assertEqual(reads.call_count, len(trajectories))
                self.assertGreater(uncached_reads, reads.call_count)
            self.assertEqual(result['accepted_steps'], reference['accepted_steps'])
            self.assertEqual(result['rejected_epochs'], reference['rejected_epochs'])
            self.assertGreater(result['cache']['batch_hits'], 0)
            self.assertLessEqual(result['cache']['retained_bytes'], cached.cache_bytes)
            self.assertEqual(reference['cache']['retained_bytes'], 0)
            self.assertEqual(uncached.rng.bit_generator.state, cached.rng.bit_generator.state)
            for left, right in zip(model.get_weights(), other.get_weights()):
                np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
            for left, right in zip(uncached.optimizer.variables, cached.optimizer.variables):
                np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)

    def test_rollout_cache_accounts_for_views_and_obeys_small_budget(self):
        import numpy as np
        from patchwork_automaton.neural.rollout_cache import RolloutCache, retained_bytes
        array = np.ones(1000, np.float32)
        self.assertGreaterEqual(retained_bytes(array[:1]), array.nbytes)
        cache = RolloutCache(1024)
        self.assertFalse(cache.put('too_big', array))
        self.assertEqual(cache.bytes, 0)
        self.assertTrue(cache.put('small', (1, 2)))
        self.assertEqual(cache.get('small'), (1, 2))
        self.assertLessEqual(cache.bytes, cache.limit)

    def fixture(self, directory, *, missing_behavior=False):
        import numpy as np
        from patchwork_automaton.neural.policy import untrained_policy
        from patchwork_automaton.neural.scenarios import sample
        from patchwork_automaton.neural.features import collate
        from patchwork_automaton.neural.training import Sample
        from patchwork_automaton.neural.experiment import TrajectoryWriter
        policy=untrained_policy(seed=19,width=16,memory_size=16,blocks=1,stable_heads=True)
        rng=np.random.default_rng(20);trajectories=[]
        for index,reward in enumerate((1.,-1.)):
            path=Path(directory)/f'{index}.pkl.gz';writer=TrajectoryWriter(path)
            try:
                for task in ('mulligan','mulligan_bottom','bolt_target'):
                    scenario=sample(task,rng);arrays=policy.encoder.encode(scenario.observation).arrays
                    inputs=collate([arrays]);inputs.update(memory_in=np.zeros((1,16),np.float32),advance_memory=np.ones((1,1),np.float32))
                    output=policy.model(inputs);logp=np.asarray(output['logits'])[0]
                    logp=logp-logp.max();logp=logp-np.log(np.exp(logp).sum())
                    action=int(rng.choice(len(logp),p=np.exp(logp)))
                    writer.write(Sample(arrays,np.zeros(16,np.float32),True,action,float(output['value'][0,0]),
                                        None if missing_behavior else logp.copy()))
            finally:writer.close()
            trajectories.append(dict(path=str(path),samples=3,reward=reward))
        return policy.model,trajectories

    def test_streamed_epochs_are_guarded_and_use_terminal_returns(self):
        import numpy as np
        from patchwork_automaton.neural.streaming_ppo import StreamingPPOTrainer
        from patchwork_automaton.neural.experiment import read_trajectory
        with tempfile.TemporaryDirectory() as directory:
            model,trajectories=self.fixture(directory)
            before=[s.log_probs.copy() for t in trajectories for s in read_trajectory(t['path'])]
            trainer=StreamingPPOTrainer(model,seed=21,batch_size=2,epochs=2,learning_rate=.0001,
                advantage_scale_floor=1.,target_kl=.01,max_state_kl=.05,trust_guard=True)
            trainer.compile();result=trainer.update_files(trajectories)
            self.assertEqual(result['samples'],6)
            self.assertTrue(result['terminal_rewards_only'])
            self.assertGreater(result['accepted_steps'],0)
            self.assertEqual(result['optimizer_steps'],result['accepted_steps'])
            self.assertLessEqual(result['largest_accepted_kl'],.01)
            self.assertLessEqual(result['largest_accepted_state_kl'],.05)
            after=[s.log_probs for t in trajectories for s in read_trajectory(t['path'])]
            for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
            self.assertTrue(all(np.isfinite(v).all() for v in model.get_weights()))

    def test_missing_behavior_or_nonterminal_rewards_cannot_train(self):
        from patchwork_automaton.neural.streaming_ppo import StreamingPPOTrainer
        with tempfile.TemporaryDirectory() as directory:
            model,trajectories=self.fixture(directory,missing_behavior=True)
            trainer=StreamingPPOTrainer(model,batch_size=2,epochs=1)
            with self.assertRaisesRegex(ValueError,'behavior probabilities'):trainer.update_files(trajectories)
            trajectories[0]['reward']=.2
            with self.assertRaisesRegex(ValueError,'terminal win/loss/draw'):trainer.update_files(trajectories)

    def test_incomplete_trajectory_is_rejected(self):
        from patchwork_automaton.neural.streaming_ppo import StreamingPPOTrainer
        with tempfile.TemporaryDirectory() as directory:
            model,trajectories=self.fixture(directory);trajectories[0]['samples']+=1
            trainer=StreamingPPOTrainer(model,batch_size=2,epochs=1)
            with self.assertRaisesRegex(ValueError,'Incomplete trajectory'):trainer.update_files(trajectories)

    def test_checkpoint_restores_ppo_optimizer_and_sampling_rng(self):
        import numpy as np
        from patchwork_automaton.neural.streaming_ppo import StreamingPPOTrainer
        from patchwork_automaton.neural.experiment import save_checkpoint,load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            model,trajectories=self.fixture(directory)
            config=dict(algorithm='ppo',learning_rate=.0001,train_batch=2,ppo_epochs=2,
                        target_kl=.01,max_state_kl=.05,advantage_scale_floor=1.,seed=31)
            trainer=StreamingPPOTrainer(model,seed=5031,batch_size=2,epochs=2,learning_rate=.0001,
                                        advantage_scale_floor=1.,target_kl=.01,max_state_kl=.05)
            trainer.compile();trainer.update_files(trajectories)
            # PPO takes multiple optimizer steps within one completed round.
            self.assertGreater(int(trainer.optimizer.iterations),1)
            path=Path(directory)/'checkpoint'
            save_checkpoint(path,model,trainer,dict(round=1,config=config))
            restored,other,state=load_checkpoint(path)
            self.assertEqual(state['optimizer_steps'],int(trainer.optimizer.iterations))
            trainer.update_files(trajectories);other.update_files(trajectories)
            for a,b in zip(model.get_weights(),restored.get_weights()):np.testing.assert_allclose(a,b,atol=1e-7)
            for a,b in zip(trainer.optimizer.variables,other.optimizer.variables):np.testing.assert_allclose(a,b,atol=1e-7)
