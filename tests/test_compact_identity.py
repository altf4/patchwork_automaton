"""Input collisions, privacy, object permutations, transfer, packing and resume."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from patchwork_automaton.core import Handle,Response
from patchwork_automaton.backends.rustmage import OBSERVATION_SCHEMA
from patchwork_automaton.neural.rustmage_observation import _Object,_Player,PLAYERS
from patchwork_automaton.neural.compact_identity import IdentityEncoder,pack,PackedBuilder
from patchwork_automaton.neural.identity_features import card_id,descriptor_matrix,FEATURE_NAMES
from patchwork_automaton.neural import compact_standard as legacy

class FixtureObservation(SimpleNamespace):
    def _replace(self,**kwargs):return type(self)(**(vars(self)|kwargs))

def observation(names=('Spell Pierce','Spell Snare'),*,reverse=False,handles=(10,20),hidden=False,lands=3,life=12):
    objects={}
    for name,identifier in zip(names,handles):
        handle=Handle('rustmage','object',str(identifier))
        prop=dict(id=identifier,name=name,card=name,zone='Hand',controller='P0',kinds=8,colors=2,
            keywords=0,power=None,toughness=None,counters={},face_down=hidden)
        objects[handle]=_Object(handle,name,'hand',PLAYERS['P0'],prop)
    for i in range(lands):
        identifier=100+i;handle=Handle('rustmage','object',str(identifier))
        prop=dict(id=identifier,name='Island',card='Island',zone='Battlefield',controller='P0',kinds=2,
            colors=0,keywords=0,power=None,toughness=None,counters={})
        objects[handle]=_Object(handle,'Island','battlefield',PLAYERS['P0'],prop)
    if reverse:objects=dict(reversed(list(objects.items())))
    players=tuple(_Player(PLAYERS[f'P{i}'],f'P{i}',life,2,40,
        dict(life=life,hand_count=2,library_count=40,mana=[0,2 if i else 0,0,0,0,0])) for i in range(2))
    raw=dict(schema=OBSERVATION_SCHEMA,decision='priority',step='Main1',active='P0',turn=5,
        stack=[],waiting_triggers=[],objects=[o.properties for o in objects.values()])
    choices=tuple((Response(str(i)),dict(kind='AnnounceCast',card=h)) for i,h in enumerate(handles))
    obs=FixtureObservation(backend='rustmage',decision=SimpleNamespace(kind='priority',token=None),
        actor=PLAYERS['P0'],objects=objects,players=players,native=raw)
    return obs,choices

def record(**kwargs):
    obs,choices=observation(**kwargs)
    return IdentityEncoder().encode(obs,choices=choices).arrays['_compact']

class InputTests(unittest.TestCase):
    def test_old_collision_is_split_by_identity_and_typed_rules(self):
        r=record();self.assertEqual(r.roles[0,0],r.roles[1,0])
        self.assertNotEqual(r.identity_roles[0,0],r.identity_roles[1,0])
        np.testing.assert_array_equal(r.identity_ids[r.identity_roles[:,0]],
            [card_id('Spell Pierce'),card_id('Spell Snare')])
        table=descriptor_matrix();i={k:j for j,k in enumerate(FEATURE_NAMES)}
        pierce,snare=table[card_id('Spell Pierce')],table[card_id('Spell Snare')]
        self.assertEqual(pierce[i['counter.tax_fixed']],1)
        self.assertEqual(snare[i['counter.hard']],1)
        self.assertEqual(pierce[i['target.excluded_creature']],1)
        self.assertGreater(snare[i['target.min_mana']],0)
        marsh,tomb=table[card_id('Blooming Marsh')],table[card_id('Overgrown Tomb')]
        self.assertEqual(marsh[i['entry.land_count_at_most']],1)
        self.assertEqual(tomb[i['entry.pay_life']],1)
        self.assertNotEqual(marsh.tolist(),tomb.tolist())

    def test_hidden_and_unknown_identities_stay_unknown(self):
        for names in [('Spell Pierce','Spell Snare'),('Forest','Island')]:
            r=record(names=names,hidden=True)
            np.testing.assert_array_equal(r.identity_ids[r.identity_roles[:,0]],[0,0])
        self.assertEqual(card_id('Unregistered future card'),0)
        np.testing.assert_array_equal(descriptor_matrix()[0],0)
        from patchwork_automaton.neural.compact_identity import _normalized_packet
        for packet in [dict(source_card='Spell Pierce',face_down=True),
                dict(source=dict(card='Spell Pierce',hideInfo=True)),
                dict(source=dict(card='Spell Pierce',object=dict(faceDown=True)))]:
            self.assertIsNone(_normalized_packet(packet)['source_card'])

    def test_version_boundary_and_budget(self):
        r=record()
        with self.assertRaisesRegex(ValueError,'input budget'):PackedBuilder(16).append(r)
        obs,choices=observation();old=IdentityEncoder().encode(obs,choices=choices).arrays
        with self.assertRaisesRegex(ValueError,'v1 rollouts'):PackedBuilder().append(legacy.encode(old))
        data=pack([r,record(lands=1)])
        self.assertEqual(data['identity_roles'].shape,(4,3))
        self.assertLess(r.nbytes,sum(v.nbytes for k,v in old.items() if k!='_compact'))
        malformed=deepcopy(r);malformed.identity_ids[1]=2**30
        with self.assertRaisesRegex(ValueError,'sidecar'):pack([malformed])
        malformed=deepcopy(r);malformed.identity_roles=malformed.identity_roles.astype(np.float32)
        with self.assertRaisesRegex(ValueError,'sidecar'):pack([malformed])

    def test_named_stack_target_without_live_object_keeps_public_identity(self):
        from patchwork_automaton.neural.standard_features import StandardEncoder
        obs,choices=observation();arrays=StandardEncoder().encode(obs,choices=choices).arrays
        choices=tuple((response,dict(kind='ChooseTarget',target={'Stack':77})) for response,_ in choices)
        for concealed in (False,True):
            obs.native['stack']=[dict(id=77,source_card='Spell Pierce',face_down=concealed)]
            rich=IdentityEncoder().enrich(obs,choices,arrays)['_compact']
            self.assertEqual(rich.identity_ids[rich.identity_roles[0,1]],0 if concealed else card_id('Spell Pierce'))

class NeuralTests(unittest.TestCase):
    def test_transfer_permutation_packing_and_serialization(self):
        import tensorflow as tf
        import keras
        from patchwork_automaton.neural.compact_standard_model import CompactActorCritic,TensorStore as OldStore
        from patchwork_automaton.neural.compact_identity_model import transfer_v1,TensorStore
        keras.utils.set_random_seed(123)
        old=CompactActorCritic();old(old.example_inputs())
        model=transfer_v1(old)
        rs=[record(),record(reverse=True,handles=(888,999)),record(lands=1)]
        data=pack(rs);b=TensorStore(data).batch(tf.constant([2,0,1,0],tf.int32))
        expected=OldStore(legacy.pack(rs)).batch(tf.constant([2,0,1,0],tf.int32))
        for key in ['logits','value']:
            np.testing.assert_array_equal(model(b)[key],old(expected)[key])
        # Exercise the residual, not only its zero initialization.
        for side in (model.actor,model.critic):
            layer=side.residual.output_layer
            layer.kernel.assign(tf.ones_like(layer.kernel)*.03)
        output=model(b)
        self.assertNotAlmostEqual(float(output['logits'][1,0]),float(output['logits'][1,1]),places=7)
        np.testing.assert_allclose(output['logits'][1],output['logits'][2],rtol=1e-5,atol=1e-6)
        for i,ri in enumerate([2,0,1,0]):
            single=TensorStore(pack([rs[ri]])).batch(tf.constant([0],tf.int32))
            np.testing.assert_allclose(output['logits'][i],model(single)['logits'][0],rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(output['value'][i],model(single)['value'][0],rtol=1e-5,atol=1e-6)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'policy.keras';model.save(path)
            restored=keras.saving.load_model(path)
            for key in ['logits','value']:np.testing.assert_array_equal(restored(b)[key],output[key])

    def test_new_identity_inputs_receive_gradients(self):
        import tensorflow as tf
        from patchwork_automaton.neural.compact_standard_model import CompactActorCritic
        from patchwork_automaton.neural.compact_identity_model import transfer_v1,TensorStore
        old=CompactActorCritic();old(old.example_inputs());model=transfer_v1(old)
        b=TensorStore(pack([record()])).batch(tf.constant([0],tf.int32))
        model.actor.residual.output_layer.kernel.assign(tf.ones_like(model.actor.residual.output_layer.kernel)*.01)
        with tf.GradientTape() as tape:
            loss=-tf.nn.log_softmax(model(b)['logits'])[0,1]
        gradient=tape.gradient(loss,model.actor.residual.identity.embeddings)
        self.assertGreater(float(tf.reduce_sum(tf.abs(tf.convert_to_tensor(gradient)))),0.)

    def test_ppo_holdout_optimizer_rng_resume_and_compiled_inference(self):
        import asyncio
        import tensorflow as tf
        from examples import train_hartford_ppo as driver
        from patchwork_automaton.neural.batching import record_batch,InferenceBatcher
        from patchwork_automaton.neural.compact_identity_model import transfer_v1
        from patchwork_automaton.neural.experiment import make_trainer,save_checkpoint,load_checkpoint
        from patchwork_automaton.neural.rollout_records import Sample,TrajectoryWriter
        from patchwork_automaton.neural.standard_policy import StandardPolicy
        from patchwork_automaton.neural.validation import GameValidator,training_normalization
        config=driver.configuration(driver.parser().parse_args([]))
        config.update(train_batch=2,ppo_epochs=2,value_max_steps=2,concurrent_updates=False)
        old,_=driver.initialize_random(config,device='cpu')
        model=transfer_v1(old)
        self.assertFalse(model.use_rules)
        config['architecture']='compact-identity-standard-v2'
        trainer=make_trainer(model,config);trainer.compile()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);games=[]
            for i,reward in enumerate((1.,-1.,1.)):
                writer=TrajectoryWriter(root/f'game-{i}.pkl.gz')
                for j in range(2):
                    r=record(lands=i+1,life=10+j)
                    sample=Sample(dict(_compact=r,candidate_mask=np.ones(2,bool)),np.zeros(0,np.float32),True,j%2,0.,None)
                    output=model(record_batch(model,[sample]))
                    sample.value=float(output['value'][0,0])
                    sample.log_probs=tf.nn.log_softmax(output['logits']).numpy()[0]
                    writer.write(sample)
                writer.close()
                games.append(dict(index=i,seed=400+i,trajectories=[dict(path=str(writer.path),samples=2,reward=reward)]))
            trajectories=[t for g in games[:2] for t in g['trajectories']]
            metrics=trainer.update_files(trajectories)
            self.assertEqual(metrics['samples'],4)
            self.assertEqual(metrics['pipeline']['profile'],model.input_format)
            self.assertEqual(metrics['accepted_steps'],4)
            before=driver.training_state(model,trainer)
            held=GameValidator(model,batch_size=2).evaluate(games[2:],training_normalization(games[:2],config['advantage_scale_floor']))
            self.assertEqual(held['terminal_games'],1)
            self.assertEqual(driver.training_state(model,trainer),before)
            save_checkpoint(root/'checkpoints',model,trainer,dict(round=1,config=config),policy_type=StandardPolicy)
            loaded,restored,_=load_checkpoint(root/'checkpoints',policy_type=StandardPolicy)
            self.assertEqual(driver.training_state(loaded,restored),before)
            async def inference():
                loop=asyncio.get_running_loop();running=True
                def tick():
                    if running:loop.call_later(.01,tick)
                loop.call_later(.01,tick)
                service=InferenceBatcher(loaded,max_batch=2,wait_ms=0,device='cpu')
                await service.start()
                try:
                    return await asyncio.to_thread(service.predict,dict(sample.arrays,memory_in=np.zeros(0,np.float32),
                        advance_memory=np.ones(1,np.float32)),version=0,identity=('identity-test',),unbatched=True)
                finally:
                    await service.close();running=False
            expected=model(record_batch(model,[sample]))
            prediction=asyncio.run(inference())
            np.testing.assert_allclose(prediction['logits'],expected['logits'],rtol=1e-5,atol=1e-6)
            trainer.update_files(trajectories);restored.update_files(trajectories)
            self.assertEqual(driver.training_state(model,trainer),driver.training_state(loaded,restored))


class NativeParityTests(unittest.TestCase):
    def test_private_native_choices_and_identity_sidecar_match_reference(self):
        import json
        from examples.experiment_compact_standard import Game,DECKS
        from patchwork_automaton.neural.direct_standard import arrays,string,library_path
        from patchwork_automaton.neural.rustmage_observation import learning_observation
        from patchwork_automaton.neural.standard_features import StandardEncoder
        from patchwork_automaton.backends.rustmage import _plain
        if not library_path().is_file():self.skipTest('Requires the guarded direct Standard build')
        decisions=0
        for i in range(3):
            row=dict(index=i,seed=73410+i,decks=[DECKS[i],DECKS[(i+1)%3]],starting_player=f'P{i%2}')
            game=Game(row)
            try:
                while True:
                    view,meta=game.observe()
                    if view is None:break
                    action=game.script(view,meta.actor)
                    count=game.size(view)
                    if count>1:
                        raw=json.loads(string(game.lib,game.lib.ds_debug_json(view)))
                        fake=SimpleNamespace(session='identity-parity',handle=Handle('rustmage','game',str(i)),_raw=lambda _:raw)
                        obs=learning_observation(fake,f'P{meta.actor}')
                        encoder=IdentityEncoder();choices=encoder.expand_options(obs.decision)
                        native=[json.loads(string(game.lib,game.lib.ds_choice_debug_json(view,j))) for j in range(count)]
                        self.assertEqual([_plain(a) for _,a in choices],native)
                        old=arrays(game.lib,game.lib.ds_encode(view))
                        rich=encoder.enrich(obs,choices,old)['_compact']
                        reference=encoder.encode(obs,choices=choices).arrays
                        for key in old:
                            np.testing.assert_allclose(old[key],reference[key],rtol=1e-6,atol=1e-6,err_msg=key)
                        for key in ('identity_ids','identity_base_nodes','identity_roles','identity_context'):
                            np.testing.assert_array_equal(getattr(rich,key),getattr(reference['_compact'],key))
                        decisions+=1
                    game.apply(view,action=action)
            finally:game.close()
        self.assertGreater(decisions,30)


class ReplicaTests(unittest.TestCase):
    def test_identity_payload_through_independent_compiled_replica(self):
        import asyncio
        import multiprocessing as mp
        import os
        from unittest.mock import patch
        from examples.experiment_hartford_identity import with_executor_shutdown
        from patchwork_automaton.neural.compact_identity_model import CompactIdentityActorCritic,TensorStore
        from patchwork_automaton.neural.inference_processes import InferenceProcessGroup
        from patchwork_automaton.rl.rustmage_processes import _RemoteInference
        import tensorflow as tf
        async def run():
            model=CompactIdentityActorCritic();model(model.example_inputs())
            r=record();b=TensorStore(pack([r])).batch(tf.constant([0]))
            expected=model(b)
            server,client=mp.get_context('spawn').Pipe()
            group=InferenceProcessGroup([server],replicas=1,timeout=20)
            try:
                await group.start(model)
                payload=dict(_compact=r,candidate_mask=np.ones(2,bool),memory_in=np.zeros(0,np.float32),
                    advance_memory=np.ones(1,np.float32))
                prediction=await asyncio.to_thread(_RemoteInference(client).predict,payload,
                    version=0,identity=('identity-replica',),unbatched=True)
                for k in ('logits','value'):np.testing.assert_allclose(prediction[k],expected[k],rtol=1e-5,atol=1e-6)
                self.assertEqual((await group.refresh_stats())['requests'],1)
            finally:
                await group.close();client.close()
        # Restricted sandboxes may not wake selectors for executor callbacks.
        with patch.dict(os.environ,{'PATCHWORK_ARENA_POLL_INFERENCE':'1'}):
            asyncio.run(with_executor_shutdown(run()))

if __name__=='__main__':unittest.main()
