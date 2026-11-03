"""Mixed immutable checkpoints, exact resume, and approved game deadlines."""
import asyncio
from collections import Counter
from copy import deepcopy
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from examples import train_hartford_ppo as driver
import test_hartford_driver as fixtures


class MixedCollectionTests(unittest.TestCase):
    def test_default_deadline_and_legacy_compatibility(self):
        current=driver.configuration(driver.parser().parse_args([]))
        legacy=driver.configuration(driver.parser().parse_args(['--architecture','legacy']))
        self.assertEqual(current['game_timeout'],20)
        self.assertEqual(current['collection_mode'],'mixed-frozen-opponents-v1')
        self.assertEqual(legacy['game_timeout'],60);self.assertNotIn('collection_mode',legacy)
        for value in ('0','nan','inf','-1'):
            with self.assertRaisesRegex(ValueError,'timeout'):driver.configuration(driver.parser().parse_args(['--game-timeout',value]))

    def test_all_versions_share_one_schedule_and_interrupted_resume_keeps_games(self):
        fixture=fixtures.HartfordDriverTests('test_exact_default_recipe_and_no_warm_start_options');fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        planned=fixture.planned();recorder=fixtures.Recorder();pinned=[]
        async def forbidden(number):raise AssertionError('Serial pin must never run')
        async def bank(numbers):pinned.append(numbers);return {n:str(n) for n in numbers}
        def run():return asyncio.run(driver.collect_round(planned=planned,number=4,directory=fixture.root/'rollouts',
            collectors=[recorder],pin_opponent=forbidden,pin_opponents=bank,game_timeout=20,progress=lambda *a,**kw:None))
        recorder.fail_after=13
        with self.assertRaisesRegex(RuntimeError,'interrupted'):run()
        before={str(p):driver.sha256(p) for p in (fixture.root/'rollouts/results').glob('*.json')}
        recorder.fail_after=None;rows=run()
        self.assertEqual(len(rows),1000);self.assertEqual(len(set(recorder.seen)),1000)
        self.assertEqual(Counter(x['opponent_version'] for x in rows),{3:700,2:100,1:100,0:100})
        self.assertEqual(pinned,[[0,1,2,3],[0,1,2,3]])
        self.assertTrue(all(options['game_timeout']==20 and 'opponent_version' not in options for options in recorder.options))
        self.assertEqual(before,{p:driver.sha256(p) for p in before})
        run();self.assertEqual(len(pinned),2)

    def test_execution_amendment_preserves_historical_checkpoint_config(self):
        new=driver.configuration(driver.parser().parse_args([]));old=dict(new,game_timeout=60);old.pop('collection_mode')
        report=dict(execution_amendments=[dict(previous_config=old,new_config=new,through_checkpoint=9)])
        self.assertEqual(driver.checkpoint_configuration(report,new,0),old)
        self.assertEqual(driver.checkpoint_configuration(report,new,9),old)
        self.assertEqual(driver.checkpoint_configuration(report,new,10),new)
        report['execution_amendments'][0]['previous_config']['learning_rate']=.9
        with self.assertRaisesRegex(ValueError,'training configuration'):driver.checkpoint_configuration(report,new,9)

    def test_paused_adoption_preserves_checkpoints_and_saved_games(self):
        import test_hartford_memory_recovery as recovery
        fixture=recovery.MemoryRecoveryTests('test_adoption_preserves_checkpoint_and_reuses_all_games');fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        new=deepcopy(fixture.config);old=dict(new,game_timeout=60);old.pop('collection_mode')
        state=json.loads((fixture.cp/'state.json').read_text());state['config']=old;driver.atomic_json(fixture.cp/'state.json',state)
        checkpoint=driver.checkpoint_hashes(fixture.cp)
        manifest=deepcopy(fixture.old);manifest['config']=old
        report=deepcopy(fixture.report);report.update(config=old,checkpoints={'0':checkpoint})
        driver.atomic_json(fixture.output/'manifest.json',manifest);driver.atomic_json(fixture.output/'report.json',report)
        driver.atomic_json(fixture.guard,dict(status='stopped',reason='signal',guard_exit_code=143,child_pid=2147483647))
        pins=deepcopy(fixture.pins)
        for path in ('src/patchwork_automaton/neural/inference_processes.py','src/patchwork_automaton/rl/standard_selfplay_processes.py','src/patchwork_automaton/rl/hartford_evaluation.py'):
            pins['source'][path]='new source'
        before={str(p):driver.sha256(p) for p in fixture.directory.rglob('*') if p.is_file()}
        with patch('examples.train_hartford_ppo.os.kill',side_effect=ProcessLookupError):
            amendment=driver.adopt_collection_policy(fixture.output,fixture.original,fixture.guard,new,pins)
        self.assertEqual(amendment['preserved_games'],1000)
        self.assertEqual(checkpoint,driver.checkpoint_hashes(fixture.cp))
        self.assertEqual(before,{p:driver.sha256(p) for p in before})
        result=json.loads((fixture.output/'report.json').read_text());self.assertEqual(result['status'],'paused')
        self.assertEqual(driver.checkpoint_configuration(result,new,0),old)


class FrozenBankTests(unittest.TestCase):
    def test_four_versions_route_concurrently_and_reject_unknown_or_mutation(self):
        from patchwork_automaton.neural.batching import record_batch
        from patchwork_automaton.neural.compact_standard import observation
        from patchwork_automaton.neural.compact_standard_model import CompactActorCritic
        from patchwork_automaton.neural.inference_processes import InferenceProcessGroup,weights_digest
        from patchwork_automaton.neural.rollout_records import Sample
        from patchwork_automaton.rl.rustmage_processes import _RemoteInference
        from test_compact_standard import fixture
        import tensorflow as tf
        async def run():
            loop=asyncio.get_running_loop();active=True
            def tick():
                if active:loop.call_later(.01,tick)
            loop.call_later(.01,tick)
            with tf.device('/CPU:0'):
                model=CompactActorCritic();data=observation(fixture())
                sample=Sample(data,np.zeros(0,np.float32),True,0,0.)
                batch=record_batch(model,[sample]);model(batch)
                base=model.get_weights();bank={};expected={}
                for number in range(4):
                    weights=[a.copy() for a in base];weights[-1]+=number
                    weights[len(model.actor.get_weights())-1]+=.1*number
                    model.set_weights(weights);bank[number]=weights
                    expected[number]={k:np.asarray(v) for k,v in model(batch).items()}
            pairs=[mp.get_context('spawn').Pipe() for _ in range(4)]
            group=InferenceProcessGroup([pair[0] for pair in pairs],replicas=1)
            clients=[_RemoteInference(pair[1]) for pair in pairs]
            request=dict(data,memory_in=np.zeros(0,np.float32),advance_memory=np.ones(1,np.float32))
            async def predict(client,number):
                return await asyncio.to_thread(client.predict,request,version=number,identity=('mixed-test',number),unbatched=True)
            try:
                await group.start(model,version_weights=bank)
                self.assertEqual(group.version_digests,{n:weights_digest(w) for n,w in bank.items()})
                for order in (list(range(4)),list(reversed(range(4)))):
                    outputs=await asyncio.gather(*(predict(client,n) for client,n in zip(clients,order)))
                    for output,n in zip(outputs,order):
                        for key in expected[n]:np.testing.assert_allclose(output[key],expected[n][key],rtol=1e-5,atol=1e-6)
                with self.assertRaisesRegex(RuntimeError,'Unknown frozen opponent'):await predict(clients[0],99)
                with self.assertRaisesRegex(RuntimeError,'cannot be mutated'):await group.sync(model,5)
                stats=await group.refresh_stats();self.assertEqual(stats['requests'],8)
                self.assertEqual(len(stats['replicas']),1)
            finally:
                await group.close()
                for _,client in pairs:client.close()
                active=False
            self.assertTrue(all(not process.is_alive() for _,process in group.handles))
        with patch.dict(os.environ,{'PATCHWORK_ARENA_POLL_INFERENCE':'1'}):asyncio.run(run())


class MixedEvaluationTests(unittest.TestCase):
    def test_three_predecessors_share_one_queue_and_resume_only_missing_results(self):
        import test_hartford_evaluation as fixtures
        fixture=fixtures.HartfordEvaluationTests('test_every_matchup_both_sampled_frozen_versions_and_no_training');fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        seen=[];batches=[];fail=[True]
        class Collector(fixtures.FakeCollector):
            def __init__(self,model,opponent_model,**options):
                super().__init__();self.model=model
            async def start(self,*,version,opponent_weights):
                from patchwork_automaton.neural.inference_processes import weights_digest
                self.policy_digest=weights_digest(self.model.get_weights())
                self.replicas=[None,SimpleNamespace(version_digests={n:weights_digest(w) for n,w in opponent_weights.items()})]
            async def run_schedule(self,jobs,*,progress,**options):
                batches.append([job['game']['opponent_version'] for job in jobs])
                for job in jobs:
                    if fail[0] and len(seen)==7:raise RuntimeError('mixed interruption')
                    seen.append(job['index']);progress(fixtures.successful_row(job['game']))
        with patch('patchwork_automaton.rl.standard_selfplay_processes.StandardSelfplayCollector',Collector):
            with self.assertRaisesRegex(RuntimeError,'mixed interruption'):
                fixture.run_evaluation(collectors=None,mixed_opponents=True,game_timeout=20)
            fail[0]=False;result=fixture.run_evaluation(collectors=None,mixed_opponents=True,game_timeout=20)
            fixture.run_evaluation(collectors=None,mixed_opponents=True,game_timeout=20)
        self.assertEqual(result['aggregate']['games'],1200)
        self.assertEqual(len(seen),len(set(seen)));self.assertEqual(len(seen),1200)
        self.assertEqual(set(batches[0][:3]),{0,1,2})
        self.assertEqual(Counter(batches[0]),{0:400,1:400,2:400})


if __name__=='__main__':unittest.main()
