"""Permanent compact Hartford PPO, persistence and read-only integration."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')

from examples import train_hartford_ppo as driver
from patchwork_automaton.neural.compact_standard import VERSION,encode,observation,pack,PackedBuilder
from test_compact_standard import fixture


class ConfigurationTests(unittest.TestCase):
    def test_compact_default_is_explicit_and_legacy_config_stays_compatible(self):
        current=driver.configuration(driver.parser().parse_args([]))
        old=driver.configuration(driver.parser().parse_args(['--architecture','legacy']))
        self.assertEqual(current['architecture'],'compact-standard-v1')
        self.assertFalse(current['recurrent_memory'])
        self.assertEqual(current['compact_input_mib'],512)
        self.assertNotIn('architecture',old)
        for k in ('learning_rate','value_learning_rate','train_batch','ppo_epochs','entropy_weight',
                  'target_kl','max_state_kl','value_max_steps','ppo_clip','advantage_scale_floor',
                  'seed','weight_seed','eval_seed','initialization','action_selection','concurrent_updates'):
            self.assertEqual(current[k],old[k],k)
        self.assertEqual(driver.memory_runtime_policy(current)['profile'],'sparse-device-resident-v1')

    def test_bad_input_budget_and_architecture_resume_are_rejected(self):
        with self.assertRaises(ValueError):driver.configuration(driver.parser().parse_args(['--compact-input-mib','0']))
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run';out.mkdir()
            driver.atomic_json(out/'manifest.json',dict(config={'experiment':'legacy'}))
            args=driver.parser().parse_args(['--output',str(out),'--resume'])
            with patch.object(driver,'hartford_preflight',return_value={'ready':True}),patch.object(driver,'require_hartford_ready'):
                with self.assertRaisesRegex(ValueError,'fresh output directory'):asyncio.run(driver.run(args))


class CompactPPOTests(unittest.TestCase):
    def config(self,**updates):
        config=driver.configuration(driver.parser().parse_args([]))
        config.update(train_batch=2,ppo_epochs=2,value_max_steps=2,concurrent_updates=False)
        config.update(updates);return config

    def files(self,root,model):
        import tensorflow as tf
        from patchwork_automaton.neural.batching import record_batch
        from patchwork_automaton.neural.rollout_records import Sample,TrajectoryWriter
        games=[]
        for i,reward in enumerate((1.,-1.,1.)):
            writer=TrajectoryWriter(root/f'game-{i}.pkl.gz')
            for j in range(2):
                data=fixture();data['entities'][0,3]=i+j/10
                raw=observation(data)
                sample=Sample(raw,np.zeros(0,np.float32),True,j%3,0.,None)
                b=record_batch(model,[sample]);out=model(b,training=False)
                sample.value=float(out['value'][0,0]);sample.log_probs=tf.nn.log_softmax(out['logits']).numpy()[0]
                writer.write(sample)
            writer.close()
            games.append(dict(index=i,seed=991+i,trajectories=[dict(path=str(writer.path),samples=2,reward=reward)]))
        return games

    def test_ppo_checkpoint_resume_holdout_and_sparse_inference(self):
        import keras
        import tensorflow as tf
        from patchwork_automaton.neural.batching import InferenceBatcher,model_example_batch,record_batch
        from patchwork_automaton.neural.experiment import save_checkpoint,load_checkpoint
        from patchwork_automaton.neural.rollout_records import read_trajectory
        from patchwork_automaton.neural.standard_policy import StandardPolicy
        from patchwork_automaton.neural.validation import GameValidator,training_normalization
        from patchwork_automaton.neural.inference_processes import weights_digest
        config=self.config()
        with tf.device('/CPU:0'),tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,trainer=driver.initialize_random(config,device='cpu')
            self.assertEqual(model.memory_size,0);self.assertEqual(model.input_format,VERSION)
            self.assertFalse({id(v) for v in model.actor.variables}&{id(v) for v in model.critic.variables})
            games=self.files(root,model);training=games[:2];heldout=games[2:]
            before_actor=weights_digest(model.actor.get_weights());before_critic=weights_digest(model.critic.get_weights())
            result=trainer.update_files([t for g in training for t in g['trajectories']])
            self.assertEqual(result['samples'],4)
            self.assertEqual(result['accepted_steps'],4)
            self.assertEqual((int(trainer.optimizer.iterations),int(trainer.critic_optimizer.iterations)),(4,2))
            self.assertNotEqual(before_actor,weights_digest(model.actor.get_weights()))
            self.assertNotEqual(before_critic,weights_digest(model.critic.get_weights()))
            before=driver.training_state(model,trainer)
            norm=training_normalization(training,config['advantage_scale_floor'])
            validation=GameValidator(model,batch_size=2).evaluate(heldout,norm)
            self.assertEqual(validation['terminal_games'],1);self.assertTrue(np.isfinite(validation['mean_policy_kl']))
            self.assertEqual(driver.training_state(model,trainer),before)
            save_checkpoint(root/'checkpoints',model,trainer,dict(round=1,config=config),policy_type=StandardPolicy)
            loaded,restored,state=load_checkpoint(root/'checkpoints',policy_type=StandardPolicy,gru_backend='unfused')
            self.assertEqual(driver.training_state(loaded,restored),before)
            sample=next(read_trajectory(games[0]['trajectories'][0]['path']))
            direct=model(record_batch(model,[sample]),training=False)
            np.testing.assert_array_equal(direct['logits'],loaded(record_batch(loaded,[sample]),training=False)['logits'])
            async def inference():
                # The sandbox's selector can miss executor wakeups. A bounded
                # timer also mirrors the optional replica polling runtime.
                loop=asyncio.get_running_loop();running=True
                def tick():
                    if running:loop.call_later(.01,tick)
                loop.call_later(.01,tick)
                service=InferenceBatcher(loaded,max_batch=2,wait_ms=0,device='cpu')
                await service.start()
                try:
                    return await asyncio.to_thread(service.predict,dict(sample.arrays,memory_in=np.zeros(0,np.float32),
                        advance_memory=np.ones(1,np.float32)),version=0,identity=('test',),unbatched=True)
                finally:
                    await service.close();running=False
            prediction=asyncio.run(inference())
            np.testing.assert_allclose(prediction['logits'],np.asarray(direct['logits']),rtol=1e-5,atol=1e-6)
            # Reloaded Adam moments and RNG must drive the same next update.
            left=trainer.update_files([t for g in training for t in g['trajectories']])
            right=restored.update_files([t for g in training for t in g['trajectories']])
            self.assertEqual(left['accepted_steps'],right['accepted_steps'])
            self.assertEqual(driver.training_state(model,trainer),driver.training_state(loaded,restored))
        keras.backend.clear_session()

    def test_rejection_restores_actor_and_adam_and_preserves_shuffle_order(self):
        import tensorflow as tf
        from patchwork_automaton.neural.inference_processes import weights_digest
        with tf.device('/CPU:0'),tempfile.TemporaryDirectory() as tmp:
            model,trainer=driver.initialize_random(self.config(),device='cpu');trainer.backtracks=1
            games=self.files(Path(tmp),model);trajectories=[t for g in games[:2] for t in g['trajectories']]
            weights=weights_digest(model.actor.get_weights())
            adam=weights_digest([np.asarray(v) for v in trainer.optimizer.variables])
            expected=np.random.default_rng();expected.bit_generator.state=deepcopy(trainer.rng.bit_generator.state)
            for _ in range(trainer.epochs+1):expected.permutation(4)
            with patch('patchwork_automaton.neural.compact_ppo._guard_status',return_value=(False,True,['forced_test'])):
                result=trainer.update_files(trajectories)
            self.assertTrue(result['early_stop']);self.assertEqual(result['accepted_steps'],0);self.assertEqual(result['rejected_epochs'],2)
            self.assertEqual(weights_digest(model.actor.get_weights()),weights)
            self.assertEqual(weights_digest([np.asarray(v) for v in trainer.optimizer.variables]),adam)
            self.assertEqual(trainer.rng.bit_generator.state,expected.bit_generator.state)

    def test_budget_fails_before_optimizer_mutation(self):
        import tensorflow as tf
        with tf.device('/CPU:0'),tempfile.TemporaryDirectory() as tmp:
            model,trainer=driver.initialize_random(self.config(),device='cpu');games=self.files(Path(tmp),model)
            trainer.input_budget_bytes=16;before=driver.training_state(model,trainer)
            with self.assertRaisesRegex(ValueError,'input budget'):
                trainer.update_files(games[0]['trajectories'])
            self.assertEqual(driver.training_state(model,trainer),before)

    def test_reference_readout_preserves_distinct_relational_context(self):
        import tensorflow as tf
        from patchwork_automaton.neural.compact_standard_model import TensorStore,CompactActorCritic
        data=fixture(2)
        data['entities'][1]=data['entities'][0];data['semantics'][1]=data['semantics'][0]
        role=np.r_[1.,data['semantics'][0,0],data['entities'][0]].astype(np.float32)
        from patchwork_automaton.neural.compact_standard import KINDS,ROLE
        data['candidates'][1]=data['candidates'][0]
        data['candidates'][:,KINDS:KINDS+ROLE]=role
        data['relations'][:]=0;data['relations'][0,2,1]=1
        data['candidate_refs']=np.asarray([[1,0,0],[0,1,0]],np.float32)
        with tf.device('/CPU:0'):
            b=TensorStore(pack([encode(data)])).batch(tf.constant([0],tf.int32))
            model=CompactActorCritic();out=model(b)['logits'].numpy()[0]
            self.assertGreater(abs(float(out[0]-out[1])),1e-9)


if __name__=='__main__':unittest.main()
