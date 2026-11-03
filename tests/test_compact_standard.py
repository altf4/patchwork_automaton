"""Input parity, role ordering, dataset boundaries and sparse training checks."""
import unittest
import numpy as np
from patchwork_automaton.neural.compact_standard import SPEC,KINDS,ROLE,LOCAL,encode,restore,pack
from examples.experiment_compact_standard import schedule


def fixture(count=3):
    entities=np.zeros((3,SPEC['entities']),np.float32)
    entities[:,0]=[1,2,3];entities[:,15]=[.2,.3,.4]
    semantics=np.zeros((3,1,SPEC['semantics']),np.float32);semantics[:,0,2]=[0,1,2]
    nodes=np.concatenate([np.ones((3,1),np.float32),semantics[:,0],entities],axis=1)
    candidates=np.zeros((count,SPEC['candidates']),np.float32)
    for i in range(count):
        candidates[i,i%KINDS]=1
        candidates[i,KINDS:KINDS+ROLE]=nodes[i%3]
        candidates[i,KINDS+ROLE:KINDS+2*ROLE]=nodes[(i+1)%3]
        candidates[i,-1]=i/4
    relations=np.zeros((3,3,4),np.float32);relations[1,0,0]=1
    return dict(entities=entities,semantics=semantics,context=np.arange(SPEC['context'],dtype=np.float32)/100,
        candidates=candidates,candidate_mask=np.ones(count,bool),relations=relations,
        candidate_refs=np.eye(count,3,dtype=np.float32))


class CodecTests(unittest.TestCase):
    def test_current_state_exact_round_trip_and_role_order(self):
        data=fixture();r=encode(data,action=2,deck=1);back=restore(r)
        for key in data:np.testing.assert_array_equal(back[key],data[key],err_msg=key)
        self.assertEqual(r.action,2);self.assertEqual(r.deck,1)
        self.assertNotEqual(r.roles[0,0],r.roles[0,1])
        self.assertLess(r.nbytes,sum(a.nbytes for a in data.values())/5)

    def test_pack_preserves_boundaries_and_global_example_labels(self):
        records=[encode(fixture(2),action=1),encode(fixture(3),action=2)]
        data=pack(records)
        np.testing.assert_array_equal(data['action_splits'],[0,2,5])
        np.testing.assert_array_equal(data['action'],[1,2])
        self.assertEqual(len(data['board']),data['node_splits'][-1])
        self.assertEqual(len(data['roles']),5)

    def test_balanced_disjoint_whole_games(self):
        rows=schedule(900,12345)
        from collections import Counter
        counts=Counter((tuple(r['decks']),r['partition']) for r in rows)
        self.assertEqual(set(counts.values()),{80,20})
        self.assertEqual(len({r['seed'] for r in rows}),900)
        with self.assertRaises(ValueError):schedule(100,123)

    def test_invalid_action_or_padded_inputs_rejected(self):
        data=fixture()
        with self.assertRaises(ValueError):encode(data,action=3)
        data['candidate_mask'][1]=False
        with self.assertRaises(ValueError):encode(data)


class NeuralTests(unittest.TestCase):
    def test_sparse_batching_segmented_loss_and_learning(self):
        import tensorflow as tf
        import keras
        from patchwork_automaton.neural.compact_standard_model import TensorStore,CompactStandardPolicy,objective
        keras.utils.set_random_seed(24)
        records=[encode(fixture(2),action=1),encode(fixture(3),action=2)]
        with tf.device('/CPU:0'):
            store=TensorStore(pack(records));b=store.batch(tf.constant([1,0],tf.int32))
            np.testing.assert_array_equal(b['labels'].numpy(),[2,4])
            logits=tf.constant([1.,2.,3.,4.,5.])
            loss,correct,_=objective(logits,b)
            expected=[tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[2],logits=[[1.,2.,3.]]).numpy()[0],
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[1],logits=[[4.,5.]]).numpy()[0]]
            np.testing.assert_allclose(loss.numpy(),expected,rtol=1e-6)
            np.testing.assert_array_equal(correct.numpy(),[1,1])
            model=CompactStandardPolicy(16);opt=keras.optimizers.Adam(.003)
            # Interning and computing a row once must preserve the function
            # and gradients compared with recomputing every occurrence.
            expanded=store.batch(tf.constant([1,0],tf.int32),deduplicate=False)
            np.testing.assert_allclose(model(b).numpy(),model(expanded).numpy(),rtol=1e-6,atol=1e-6)
            def gradients(batch):
                with tf.GradientTape() as tape:value=tf.reduce_mean(objective(model(batch),batch)[0])
                return [tf.convert_to_tensor(g).numpy() for g in tape.gradient(value,model.trainable_variables)]
            for left,right in zip(gradients(b),gradients(expanded)):
                np.testing.assert_allclose(left,right,rtol=1e-5,atol=1e-6)
            @tf.function
            def step():
                with tf.GradientTape() as tape:value=tf.reduce_mean(objective(model(b),b)[0])
                gradients=tape.gradient(value,model.trainable_variables)
                opt.apply_gradients(zip(gradients,model.trainable_variables));return value
            before=float(tf.reduce_mean(objective(model(b),b)[0]))
            for _ in range(40):step()
            after=float(tf.reduce_mean(objective(model(b),b)[0]))
            self.assertLess(after,before*.3)


if __name__=='__main__':unittest.main()
