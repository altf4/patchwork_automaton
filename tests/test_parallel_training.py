"""Batch routing, numerical equivalence, actor isolation and optimizer resume."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from test_neural import AVAILABLE, observation

if AVAILABLE:
    import numpy as np
    from patchwork_automaton.neural.policy import untrained_policy
    from patchwork_automaton.neural.batching import InferenceBatcher, BatchedPolicy, example_batch
    from patchwork_automaton.neural.experiment import (TrajectoryWriter, StreamingTrainer,
        save_checkpoint, load_checkpoint)
    from patchwork_automaton.neural.training import Sample, TerminalActorCritic
    from patchwork_automaton.neural.features import Encoder


@unittest.skipUnless(AVAILABLE,'Install neural dependencies with uv')
class ParallelTests(unittest.TestCase):
    def test_batched_outputs_match_individual_and_cannot_mix_versions(self):
        async def check():
            model=untrained_policy(seed=7,width=16,memory_size=16,blocks=1).model
            service=await InferenceBatcher(model,max_batch=4,wait_ms=10).start()
            try:
                inputs=[example_batch(16) for _ in range(4)]
                for i,b in enumerate(inputs):
                    b['memory_in'][:]=i/10
                    for key in ('candidates','candidate_refs','candidate_mask'):
                        b[key]=np.repeat(b[key],i+1,axis=1)
                expected=[{k:np.asarray(v) for k,v in model(b).items()} for b in inputs]
                actual=await asyncio.gather(*(service._request(b,0,('game',i)) for i,b in enumerate(inputs)))
                for first,second in zip(expected,actual):
                    for key in first:np.testing.assert_allclose(first[key],second[key],atol=2e-6)
                self.assertEqual(service.stats()['mean_batch'],4)
                self.assertEqual(service.stats()['graph_traces'],1)
                service.set_version(1)
                with self.assertRaises(RuntimeError):await service._request(inputs[0],0,('stale',))
            finally:await service.close()
        asyncio.run(check())

    def test_two_actor_states_and_conditional_digits_are_isolated(self):
        async def check():
            from patchwork_automaton.core import Option
            initial=untrained_policy(seed=9,width=16,memory_size=16,blocks=1)
            service=await InferenceBatcher(initial.model,wait_ms=5).start()
            executor=ThreadPoolExecutor(2)
            first=BatchedPolicy(initial.model,service,version=0,seed=1)
            second=BatchedPolicy(initial.model,service,version=0,seed=2)
            obs=observation(); other=replace(obs,actor=obs.players[1].id,
                 decision=replace(obs.decision,token=replace(obs.decision.token,actor=obs.players[1].id)))
            obs=replace(obs,decision=replace(obs.decision,options=(Option('amount','integer','number',{'minimum':0,'maximum':17}),)))
            try:
                a,b=await asyncio.gather(asyncio.get_running_loop().run_in_executor(executor,first.act,obs),
                                         asyncio.get_running_loop().run_in_executor(executor,second.act,other))
                self.assertTrue(0<=a.response.value<=17)
                self.assertGreater(len(a.selections),1)
                self.assertTrue(set(first.states).isdisjoint(second.states))
                saved=next(iter(second.states.values())).copy()
                first.states[next(iter(first.states))]=np.full_like(saved,99)
                np.testing.assert_array_equal(next(iter(second.states.values())),saved)
            finally:
                executor.shutdown();await service.close()
        asyncio.run(check())

    def test_streaming_gradient_matches_reference_and_resume_preserves_next_update(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            initial=untrained_policy(seed=10,width=16,memory_size=16,blocks=1)
            arrays=Encoder().encode(observation()).arrays
            batch=example_batch(16);initial.model(batch)
            trainer=StreamingTrainer(initial.model,batch_size=1);trainer.compile()
            initial.save(path/'initial.keras')
            from patchwork_automaton.neural.policy import NeuralPolicy
            reference=NeuralPolicy.load(path/'initial.keras').model
            sample=Sample(arrays,np.zeros(16,np.float32),True,0,0.)
            writer=TrajectoryWriter(path/'rollout.gz');writer.write(sample);writer.write(sample);writer.close()
            trajectories=[dict(path=str(writer.path),samples=2,reward=1.)]
            trainer.update_files(trajectories)
            TerminalActorCritic(reference,batch_size=1).update([([sample,sample],1.)])
            for a,b in zip(initial.model.get_weights(),reference.get_weights()):np.testing.assert_allclose(a,b,atol=2e-6)
            rng=np.random.default_rng(99)
            state={'round':1,'completed_games':1,'rng_state':rng.bit_generator.state,
                   'config':{'learning_rate':.0003,'train_batch':1}}
            save_checkpoint(path/'checkpoints',initial.model,trainer,state)
            model2,trainer2,state2=load_checkpoint(path/'checkpoints')
            restored=np.random.default_rng();restored.bit_generator.state=state2['rng_state']
            self.assertEqual(int(rng.integers(100000)),int(restored.integers(100000)))
            trainer.update_files(trajectories);trainer2.update_files(trajectories)
            for a,b in zip(initial.model.get_weights(),model2.get_weights()):np.testing.assert_allclose(a,b,atol=1e-7)
            self.assertEqual(int(trainer2.optimizer.iterations),2)
