"""Independent inference processes preserve snapshots, routing and round barriers."""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE,'Install neural dependencies with uv')
class InferenceReplicaTests(unittest.TestCase):
    def fixture(self):
        import keras
        from patchwork_automaton.neural.model import SeparateActorCritic
        from patchwork_automaton.neural.rustmage_features import RustMageEncoder
        from patchwork_automaton.neural.batching import example_batch
        keras.utils.set_random_seed(191)
        encoder = RustMageEncoder()
        model = SeparateActorCritic(encoder.card_count,encoder.feature_spec,
            actor_width=16,actor_memory=8,actor_blocks=0,critic_width=16,critic_memory=8)
        batch = example_batch(model.memory_size,model.feature_spec)
        model(batch,training=False)
        return model,batch

    def test_direct_routes_frozen_weights_sync_stale_requests_and_replica_failure(self):
        async def run():
            import multiprocessing as mp
            import numpy as np
            import tensorflow as tf
            from patchwork_automaton.neural.batching import input_signature
            from patchwork_automaton.neural.inference_processes import InferenceProcessGroup,weights_digest
            from patchwork_automaton.rl.rustmage_processes import _RemoteInference
            model,batch = self.fixture()
            compiled = tf.function(model.call,input_signature=[input_signature(batch)],autograph=False)
            expected = {k:np.asarray(v) for k,v in compiled(batch).items()}
            pairs = [mp.get_context('spawn').Pipe() for _ in range(2)]
            group = InferenceProcessGroup([p[0] for p in pairs],replicas=2)
            clients = [_RemoteInference(p[1]) for p in pairs]
            async def predict(client,version):
                return await asyncio.to_thread(client.predict,batch,version=version,identity=('test','game'))
            try:
                # A GPU-enabled learner must not leak its CUDA visibility or
                # default device into independently spawned CPU replicas.
                with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'):
                    await group.start(model)
                pids = {s['pid'] for s in group.stats()['replicas']}
                self.assertEqual(len(pids),2)
                self.assertNotIn(os.getpid(),pids)
                for state in group.stats()['replicas']:
                    self.assertEqual(state['device'], '/CPU:0')
                    self.assertEqual(state['cuda_visible_devices'], '-1')
                    self.assertEqual(state['visible_gpus'], [])
                    self.assertTrue(all(value.endswith('/device:CPU:0') for value in state['model_variable_devices']))
                    self.assertTrue(all(value.endswith('/device:CPU:0') for value in state['output_placements'].values()))
                weights = model.get_weights()
                weights[-1] = weights[-1]+.5
                model.set_weights(weights)
                for output in await asyncio.gather(*(predict(c,0) for c in clients)):
                    for key in expected: np.testing.assert_array_equal(output[key],expected[key])
                await group.sync(model,1)
                self.assertEqual(group.stats()['weights_sha256'],weights_digest(model.get_weights()))
                updated = {k:np.asarray(v) for k,v in compiled(batch).items()}
                self.assertFalse(np.array_equal(updated['value'],expected['value']))
                for output in await asyncio.gather(*(predict(c,1) for c in clients)):
                    for key in updated: np.testing.assert_array_equal(output[key],updated[key])
                with self.assertRaisesRegex(RuntimeError,'Stale model version'):
                    await predict(clients[0],0)
                stats = await group.refresh_stats()
                self.assertEqual(stats['requests'],4)
                self.assertEqual(stats['graph_traces'],[1,1])
                self.assertTrue(all(s['version']==1 and s['requests']==2 for s in stats['replicas']))
                group.handles[0][1].terminate()
                await asyncio.to_thread(group.handles[0][1].join)
                with self.assertRaisesRegex(RuntimeError,'exited'):
                    await group.refresh_stats()
                with self.assertRaises((EOFError,BrokenPipeError,ConnectionResetError)):
                    await predict(clients[0],1)
            finally:
                await group.close()
                for _,client in pairs: client.close()
            self.assertTrue(all(not p.is_alive() for _,p in group.handles))
        asyncio.run(run())

    def test_failed_initialization_joins_processes(self):
        async def run():
            import multiprocessing as mp
            from patchwork_automaton.neural.inference_processes import InferenceProcessGroup
            model,_ = self.fixture()
            server,client = mp.get_context('spawn').Pipe()
            group = InferenceProcessGroup([server],replicas=1)
            try:
                with patch.object(model,'get_weights',return_value=[]):
                    with self.assertRaisesRegex(RuntimeError,'Inference process failed'):
                        await group.start(model)
                self.assertTrue(group.closed)
                self.assertTrue(all(not p.is_alive() for _,p in group.handles))
            finally:
                await group.close();client.close()
        asyncio.run(run())

    def test_cancelled_start_drains_control_readers_and_joins_processes(self):
        async def run():
            import multiprocessing as mp
            from patchwork_automaton.neural.inference_processes import InferenceProcessGroup
            model,_ = self.fixture()
            server,client = mp.get_context('spawn').Pipe()
            group = InferenceProcessGroup([server],replicas=1)
            try:
                task = asyncio.create_task(group.start(model))
                while not group.handles: await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError): await task
                self.assertTrue(group.closed)
                self.assertTrue(all(not p.is_alive() for _,p in group.handles))
            finally:
                await group.close();client.close()
        asyncio.run(run())

    @unittest.skipUnless(Path('.local/rustmage/target/release/librustmage.so').is_file(),'Requires RustMage')
    def test_collection_order_round_barrier_and_cleanup_after_callback_failure(self):
        async def run():
            from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector
            from patchwork_automaton.rl.rustmage_experiment import run_games
            from patchwork_automaton.neural.batching import InferenceBatcher
            model,_ = self.fixture()
            local = await InferenceBatcher(model,max_batch=2,wait_ms=0).start()
            collector = ProcessGameCollector(model,workers=2,inference_processes=2)
            options = dict(games=4,seed_base=55,index_base=7,mode='goldfish',max_own_turns=3,max_decisions=150)
            try:
                await collector.start()
                expected = await run_games(workers=2,model=model,batcher=local,**options)
                seen = []
                task = asyncio.create_task(collector.run_games(**options,progress=lambda r:seen.append(r['index'])))
                while not collector.active: await asyncio.sleep(0)
                with self.assertRaisesRegex(RuntimeError,'active'):
                    await collector.sync_weights(model,1)
                with self.assertRaisesRegex(RuntimeError,'active'):
                    await collector.run_games(**options)
                actual = await task
                self.assertEqual(sorted(seen),[7,8,9,10])
                self.assertEqual([r['index'] for r in actual],[7,8,9,10])
                for a,b in zip(expected,actual):
                    self.assertNotEqual(b['status'],'error',b)
                    for key in ('seed','seat','starting_player','status','actions','decisions','final_life'):
                        self.assertEqual(a[key],b[key],key)
                self.assertTrue(all(s['requests']>0 for s in collector.stats()['replicas']))
                await collector.sync_weights(model,1)
                with self.assertRaisesRegex(ValueError,'Stale model version'):
                    await collector.run_games(**options,version=0)
                def fail(row): raise RuntimeError('progress failed')
                with self.assertRaisesRegex(RuntimeError,'progress failed'):
                    await collector.run_games(**options,progress=fail)
                self.assertTrue(collector.broken)
            finally:
                await collector.close();await local.close()
            self.assertTrue(all(not p.is_alive() for _,p in collector.replicas.handles))
        asyncio.run(run())

    def test_new_defaults_legacy_resume_and_total_engine_limit(self):
        from examples.train_rustmage import inference_process_count
        args = SimpleNamespace(inference_processes=None,workers=24,collector='processes')
        self.assertEqual(inference_process_count(args),2)
        self.assertEqual(inference_process_count(args,{}),1)
        self.assertEqual(inference_process_count(args,{'inference_processes':3}),3)
        args.collector = 'threads'
        self.assertEqual(inference_process_count(args),1)
        args.inference_processes = 2
        with self.assertRaisesRegex(ValueError,'process collector'): inference_process_count(args)
        args.collector = 'processes'
        for count in (0,25):
            args.inference_processes = count
            with self.assertRaisesRegex(ValueError,'between one and workers'): inference_process_count(args)
