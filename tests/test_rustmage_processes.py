"""Spawned collection preserves game outcomes and routes only private inference."""
import asyncio
from pathlib import Path
import unittest
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE and Path('.local/rustmage/target/release/librustmage.so').is_file(),
                     'Requires neural extra and RustMage')
class ProcessCollectionTests(unittest.TestCase):
    def test_thread_process_agreement_and_stale_inference_errors(self):
        async def run():
            import keras
            from patchwork_automaton.neural.model import SeparateActorCritic
            from patchwork_automaton.neural.rustmage_features import RustMageEncoder
            from patchwork_automaton.neural.batching import InferenceBatcher
            from patchwork_automaton.rl.rustmage_experiment import run_games
            from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector
            keras.utils.set_random_seed(91)
            encoder=RustMageEncoder()
            model=SeparateActorCritic(encoder.card_count,encoder.feature_spec,
                actor_width=16,actor_memory=8,actor_blocks=0,critic_width=16,critic_memory=8)
            batcher=await InferenceBatcher(model,max_batch=2,wait_ms=2).start()
            collector=ProcessGameCollector(model,batcher,workers=2)
            try:
                for mode in ('baseline','goldfish'):
                    options=dict(games=2,seed_base=45,mode=mode,max_decisions=150,max_own_turns=3)
                    expected=await run_games(workers=2,model=model,batcher=batcher,**options)
                    actual=await collector.run_games(**options)
                    for a,b in zip(expected,actual):
                        self.assertNotEqual(b['status'],'error',b)
                        self.assertGreater(b['worker_pid'], 0)
                        self.assertGreaterEqual(b['worker_inference_seconds'], 0.)
                        self.assertLessEqual(b['worker_inference_seconds'], b['seconds'])
                        if mode == 'baseline': self.assertEqual(b['worker_inference_requests'], 0)
                        else: self.assertGreater(b['worker_inference_requests'], 0)
                        for key in ('seed','index','seat','starting_player','status','actions','decisions','final_life'):
                            self.assertEqual(a[key],b[key],key)
                rows=await collector.run_games(games=2,seed_base=45,mode='goldfish',version=99,max_decisions=150)
                self.assertTrue(all(r['status']=='error' and 'Stale model version' in r['error'] for r in rows))
            finally:
                await collector.close()
                await batcher.close()
        asyncio.run(run())
