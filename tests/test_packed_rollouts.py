"""Packing is a storage/execution choice, never a change to the PPO samples."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE,'Requires neural dependencies')
class PackedRolloutTests(unittest.TestCase):
    def samples(self):
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.training import Sample
        rng=np.random.default_rng(891)
        for n,a,s,h in [(2,3,0,0),(4,1,2,3),(1,2,1,1),(3,4,3,2)]:
            values={k:v[0] for k,v in example_batch(5).items() if k not in ('memory_in','advance_memory')}
            for k,v in values.items():
                shape={'entities':(n,v.shape[-1]),'card_ids':(n,),'entity_mask':(n,),
                       'semantics':(n,s,v.shape[-1]),'semantic_mask':(n,s),'relations':(n,n,4),
                       'candidates':(a,v.shape[-1]),'candidate_refs':(a,n),'candidate_mask':(a,),
                       'history':(h,v.shape[-1]),'history_cards':(h,),'history_mask':(h,)}.get(k,v.shape)
                values[k]=(rng.normal(size=shape) if v.dtype.kind=='f' else rng.integers(0,2,size=shape)).astype(v.dtype)
            values['candidate_mask'][:]=True
            yield Sample(values,rng.normal(size=5).astype(np.float32),a%2==0,a-1,.25,np.log(np.full(a,1/a)).astype(np.float32))

    def assert_arrays(self,a,b):
        self.assertEqual(set(a),set(b))
        for key in a:
            self.assertEqual(a[key].dtype,b[key].dtype,key)
            self.assertEqual(a[key].shape,b[key].shape,key)
            np.testing.assert_array_equal(a[key],b[key],err_msg=key)

    def test_ragged_vector_gather_matches_original_padding_empty_axes_and_repeats(self):
        from patchwork_automaton.neural.packed_rollouts import NumericRollout
        from patchwork_automaton.neural.features import collate
        records=list(self.samples());store=NumericRollout()
        try:
            for sample in records:store.append(sample)
            store.finish()
            for indices in ([3,0,2,1,3],[0],[1,2],[2,3]):
                subset=[records[i] for i in indices]
                expected=collate([s.arrays for s in subset]);expected['memory_in']=np.stack([s.memory for s in subset]);expected['advance_memory']=np.asarray([[float(s.advance)] for s in subset],np.float32)
                actual=store.gather(indices);old=actual.pop('_old');actions=actual.pop('_action')[:,0]
                self.assert_arrays(expected,actual)
                ref=np.full(expected['candidate_mask'].shape,-1.e9,np.float32)
                for i,s in enumerate(subset):ref[i,:len(s.log_probs)]=s.log_probs
                np.testing.assert_array_equal(old,ref)
                np.testing.assert_array_equal(actions,[s.action for s in subset])
            for indices in ([],[-1],[4]):
                with self.assertRaises(ValueError):store.gather(indices)
        finally:store.close()
        with self.assertRaises(ValueError):store.gather([0])

    def test_policy_cache_read_only_exact_and_closes_without_views(self):
        from patchwork_automaton.neural.packed_rollouts import PolicyBatchCache
        cache=PolicyBatchCache();batch=({'a':np.arange(6,dtype=np.float32).reshape(2,3),'empty':np.empty((2,0),bool)},np.array([1,0],np.int32),np.ones(2,np.float32))
        self.assertIsNone(cache.get((0,1)))
        cache.put((0,1),batch);cache.finish();actual=cache.get((0,1))
        self.assert_arrays(batch[0],actual[0])
        for a,b in zip(batch[1:],actual[1:]):np.testing.assert_array_equal(a,b)
        self.assertFalse(actual[0]['a'].flags.writeable)
        self.assertEqual(cache.hits,1);del actual;cache.close()

    def test_full_concurrent_update_preserves_model_both_adams_rng_guard_decisions(self):
        from test_critic_warmup import CriticWarmupTests
        for cache_bytes in (0,16*1024**2):
            for prepack,policy in [(False,True),(True,False),(True,True)]:
                with self.subTest(cache=cache_bytes,prepack=prepack,policy=policy),tempfile.TemporaryDirectory() as d:
                    fixture=CriticWarmupTests();m,a,paths=fixture.fixture(Path(d)/'a',cache_bytes=cache_bytes,value_max_steps=2)
                    n,b,_=fixture.fixture(Path(d)/'b',cache_bytes=cache_bytes,value_max_steps=2)
                    b.numerical_prepacking=prepack;b.cache_policy_batches=policy
                    a.guard_audit=b.guard_audit=True
                    a.compile();b.compile();x=a.update_files(paths);y=b.update_files(paths)
                    for key in ['loss','exact_kl','max_state_kl','value_loss','entropy','accepted_steps','rejected_epochs','critic_steps','optimizer_steps','critic_optimizer_steps','early_stop','samples','guard_proposals']:
                        # Paths in auditing are intentionally the same source paths.
                        self.assertEqual(x[key],y[key],key)
                    self.assertEqual(a.rng.bit_generator.state,b.rng.bit_generator.state)
                    for left,right in [(m.variables,n.variables),(a.optimizer.variables,b.optimizer.variables),(a.critic_optimizer.variables,b.critic_optimizer.variables)]:
                        self.assertEqual(len(left),len(right))
                        for u,v in zip(left,right):np.testing.assert_array_equal(np.asarray(u),np.asarray(v))
                    if policy:self.assertGreater(y['pipeline']['policy_cache_hits'],0)
                    if prepack:self.assertGreater(y['pipeline']['numerical_disk_bytes'],0)

    def test_resources_close_when_optimizer_raises(self):
        from test_critic_warmup import CriticWarmupTests
        from patchwork_automaton.neural.packed_rollouts import NumericRollout,PolicyBatchCache
        closed=[];nclose=NumericRollout.close;pclose=PolicyBatchCache.close
        def nc(x):closed.append('numeric');nclose(x)
        def pc(x):closed.append('policy');pclose(x)
        with tempfile.TemporaryDirectory() as d:
            _,trainer,paths=CriticWarmupTests().fixture(d)
            trainer.numerical_prepacking=trainer.cache_policy_batches=True
            with patch.object(NumericRollout,'close',nc),patch.object(PolicyBatchCache,'close',pc),patch.object(trainer,'step',side_effect=RuntimeError('optimizer failed')):
                with self.assertRaisesRegex(RuntimeError,'optimizer failed'):trainer.update_files(paths)
        self.assertCountEqual(closed,['numeric','policy'])


if __name__=='__main__':unittest.main()
