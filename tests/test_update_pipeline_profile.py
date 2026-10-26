"""Timing probes must preserve update semantics and whole-game exclusions."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from examples.profile_ppo_update import Probe, instrument, interval_union, select_training
from test_neural import AVAILABLE


class ProfileAccountingTests(unittest.TestCase):
    def test_union_does_not_double_count_concurrent_spans(self):
        self.assertEqual(interval_union([(0,4),(2,7),(9,10)]),8)
        self.assertEqual(interval_union([]),0)
        with self.assertRaises(ValueError):interval_union([(2,1)])

    def test_partition_preserves_order_and_never_admits_holdout(self):
        games=[{'index':i,'trajectories':[{'path':str(i)}]} for i in [3,1,2,0]]
        report={'rounds':[{'round':40,'games':games,'split':{'training_indices':[0,3],'validation_indices':[1,2]}}]}
        _,rows=select_training(report,40)
        self.assertEqual([r['path'] for r in rows],['3','0'])
        report['rounds'][0]['split']['validation_indices']=[0,1,2]
        with self.assertRaises(ValueError):select_training(report,40)

    def test_lock_release_and_span_flush_on_exception(self):
        probe=Probe();lock=threading.Lock()
        with self.assertRaisesRegex(RuntimeError,'boom'):
            with probe.locked(lock):raise RuntimeError('boom')
        self.assertTrue(lock.acquire(blocking=False));lock.release()
        with tempfile.TemporaryDirectory() as directory:
            summary=probe.finish(directory)
            self.assertEqual(summary['spans']['packing_lock_held']['calls'],1)
            self.assertTrue((Path(directory)/'host-spans.json.gz').is_file())


@unittest.skipUnless(AVAILABLE,'Requires neural dependencies')
class ProfileParityTests(unittest.TestCase):
    def test_timing_only_preserves_model_both_adams_rng_and_guards(self):
        import numpy as np
        from test_critic_warmup import CriticWarmupTests
        for cache in (0,16*1024**2):
            with self.subTest(cache=cache),tempfile.TemporaryDirectory() as d:
                root=Path(d);fixture=CriticWarmupTests()
                model,a,paths=fixture.fixture(root/'a',cache_bytes=cache,value_max_steps=2)
                other,b,_=fixture.fixture(root/'b',cache_bytes=cache,value_max_steps=2)
                a.compile();b.compile();before=a.rng.bit_generator.state
                probe=Probe();transformed=instrument(b,probe)
                self.assertIn('_probe.locked',transformed)
                expected=a.update_files(paths);actual=b.update_files(paths)
                for key in ['accepted_steps','rejected_epochs','critic_steps','optimizer_steps','critic_optimizer_steps','early_stop','samples']:
                    self.assertEqual(expected[key],actual[key],key)
                for key in ['loss','exact_kl','max_state_kl','value_loss','entropy']:
                    self.assertAlmostEqual(expected[key],actual[key],places=7,msg=key)
                self.assertEqual(a.rng.bit_generator.state,b.rng.bit_generator.state)
                self.assertNotEqual(before,a.rng.bit_generator.state)
                for x,y in [(model.variables,other.variables),(a.optimizer.variables,b.optimizer.variables),(a.critic_optimizer.variables,b.critic_optimizer.variables)]:
                    for left,right in zip(x,y):np.testing.assert_array_equal(np.asarray(left),np.asarray(right))
                self.assertEqual(sum(p['accepted'] for p in probe.proposals),actual['accepted_steps']//4)
                summary=probe.finish(root)
                for key in ['initial_rollout_scan','batch_pack','packing_lock_wait','record_gather','collate','actor_batch_wait','critic_batch_wait','step_call','critic_step_call','guard_scan']:
                    self.assertGreater(summary['spans'][key]['calls'],0,key)


if __name__=='__main__':unittest.main()
