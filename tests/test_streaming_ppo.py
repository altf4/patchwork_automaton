"""Full-game PPO uses frozen behavior probabilities and terminal outcomes."""
import tempfile
from pathlib import Path
import unittest
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE,'Install neural dependencies with uv')
class StreamingPPOTests(unittest.TestCase):
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
