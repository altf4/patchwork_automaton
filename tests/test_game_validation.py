"""Game-level leakage protection and policy/value validation arithmetic."""
from pathlib import Path
import tempfile
import unittest
from test_neural import AVAILABLE
if AVAILABLE:
    import numpy as np
    from patchwork_automaton.neural.validation import split_games, training_normalization, GameValidator


@unittest.skipUnless(AVAILABLE,'Requires neural dependencies')
class SplitTests(unittest.TestCase):
    def test_eighty_twenty_is_stable_and_keeps_both_seats_together(self):
        games=[dict(index=i,seed=700+i,status='win',trajectories=[
            dict(path=f'{i}-p0',samples=3,reward=1.),dict(path=f'{i}-p1',samples=7,reward=-1.)]) for i in range(100)]
        train,validation,meta=split_games(games,seed=91,round_id=3)
        self.assertEqual((len(train),len(validation)),(80,20))
        self.assertEqual((meta['training']['samples'],meta['validation']['samples']),(800,200))
        reversed_split=split_games(list(reversed(games)),seed=91,round_id=3)[2]
        self.assertEqual(meta,reversed_split)
        self.assertFalse(set(meta['training_indices'])&set(meta['validation_indices']))
        self.assertEqual(set(meta['training_indices'])|set(meta['validation_indices']),set(range(100)))
        failed_index=meta['validation_indices'][0]
        games[failed_index].update(status='cutoff',trajectories=[])
        _,_,changed=split_games(games,seed=91,round_id=3)
        self.assertEqual(meta['validation_indices'],changed['validation_indices'])
        self.assertEqual(changed['validation']['terminal_games'],19)
        self.assertEqual(changed['training']['terminal_games'],80)
        duplicated=[dict(train[0]),dict(train[0],index=100)]
        with self.assertRaisesRegex(ValueError,'Trajectory paths'):split_games(duplicated)

    def test_small_and_legacy_batches(self):
        games=[dict(index=i,seed=i,trajectories=[]) for i in range(6)]
        self.assertEqual(len(split_games(games)[1]),2)
        self.assertEqual(len(split_games(games,fraction=0)[0]),6)
        with self.assertRaises(ValueError):split_games(games[:1])
        with self.assertRaises(ValueError):split_games(games,fraction=1)
        with self.assertRaises(ValueError):split_games(games+games)


@unittest.skipUnless(AVAILABLE,'Requires neural dependencies')
class MetricTests(unittest.TestCase):
    def test_replay_resets_each_seat_and_respects_nonadvancing_substeps(self):
        import tensorflow as tf
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.experiment import TrajectoryWriter
        from patchwork_automaton.neural.training import Sample
        class MemoryToy:
            memory_size=1
            feature_spec=dict(entities=1,semantics=1,context=1,candidates=1,history=1)
            def __call__(self,inputs,training=False):
                return dict(logits=tf.zeros_like(inputs['candidates'][:,:,0]),
                    value=inputs['memory_in'],
                    memory_out=inputs['memory_in']+inputs['advance_memory'])
        model=MemoryToy()
        arrays={k:v[0] for k,v in example_batch(1,model.feature_spec).items()
                if k not in ('memory_in','advance_memory')}
        arrays.update(candidates=np.ones((2,1),np.float32),candidate_mask=np.ones(2,bool),
                      candidate_refs=np.zeros((2,2),np.float32))
        with tempfile.TemporaryDirectory() as directory:
            trajectories=[]
            for seat,(advances,reward) in enumerate([([True,False,True],1.),([True,True],-1.)]):
                path=Path(directory)/f'p{seat}.pkl.gz'
                writer=TrajectoryWriter(path)
                try:
                    for advance in advances:
                        writer.write(Sample(arrays,np.array([999.],np.float32),advance,0,0.,
                                            np.log(np.array([.5,.5],np.float32))))
                finally:writer.close()
                trajectories.append(dict(path=str(path),samples=len(advances),reward=reward))
            games=[dict(index=5,seed=9,trajectories=trajectories)]
            validator=GameValidator(model)
            norm=dict(center=0.,scale=1.)
            result=validator.evaluate(games,norm)
            self.assertEqual(result['terminal_games'],1)
            self.assertEqual(result['games'][0]['samples'],5)
            self.assertEqual(result['value_after']['samples'],4)
            self.assertEqual(result['value_after']['mse'],1.5)
            self.assertEqual(result['value_mse']['game_bootstrap_95_interval'],[1.5,1.5])
            frozen=validator.evaluate(games,norm,replay=False)
            self.assertEqual(frozen['value_after']['mse'],998002.)

    def test_proxy_value_metrics_and_read_only_inference(self):
        import tensorflow as tf
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.experiment import TrajectoryWriter
        from patchwork_automaton.neural.training import Sample
        class Toy:
            memory_size=1
            feature_spec=dict(entities=1,semantics=1,context=1,candidates=1,history=1)
            def __init__(self):
                self.preference=tf.Variable(0.,dtype=tf.float32)
                self.estimate=tf.Variable(0.,dtype=tf.float32)
            def __call__(self,inputs,training=False):
                assert not training
                return dict(logits=inputs['candidates'][:,:,0]*self.preference,
                    value=tf.ones_like(inputs['memory_in'])*self.estimate,
                    memory_out=inputs['memory_in']+inputs['advance_memory'])
        model=Toy()
        arrays={k:v[0] for k,v in example_batch(1,model.feature_spec).items()
                if k not in ('memory_in','advance_memory')}
        arrays.update(candidates=np.array([[1.],[0.]],np.float32),candidate_mask=np.ones(2,bool),
                      candidate_refs=np.zeros((2,2),np.float32))
        with tempfile.TemporaryDirectory() as directory:
            games=[]
            for i,reward in enumerate([1.,-1.]):
                path=Path(directory)/f'{i}.pkl.gz'
                writer=TrajectoryWriter(path)
                try:
                    writer.write(Sample(arrays,np.zeros(1,np.float32),True,i,0.,np.log(np.array([.5,.5],np.float32))))
                finally:writer.close()
                games.append(dict(index=i,seed=i,trajectories=[dict(path=str(path),samples=1,reward=reward)]))
            norm=training_normalization(games,1.)
            self.assertEqual((norm['center'],norm['scale']),(0.,1.))
            validate=GameValidator(model)
            initial=validate.evaluate(games,norm)
            self.assertAlmostEqual(initial['policy_surrogate_gain']['mean'],0.,places=7)
            self.assertAlmostEqual(initial['value_after']['mse'],1.)
            model.preference.assign(1.)
            model.estimate.assign(.3)
            changed=validate.evaluate(games,norm)
            self.assertAlmostEqual(changed['policy_surrogate_gain']['mean'],.2,places=6)
            self.assertAlmostEqual(changed['return_surrogate_gain']['mean'],np.tanh(.5),places=6)
            self.assertAlmostEqual(changed['value_after']['mse'],1.09,places=6)
            self.assertAlmostEqual(changed['value_before']['mse'],1.,places=6)
            self.assertAlmostEqual(float(model.preference),1.)
            self.assertAlmostEqual(float(model.estimate),.3,places=6)
            # A held-out outcome cannot change the training normalization.
            training_norm=training_normalization(games[:1],1.)
            games[1]['trajectories'][0]['reward']=0.
            self.assertEqual(training_norm,training_normalization(games[:1],1.))
            empty=validate.evaluate([],norm)
            self.assertIsNone(empty['policy_surrogate_gain'])
            self.assertEqual(empty['value_after']['samples'],0)
            # Forced-only trajectories still support value validation; PPO does
            # not normalize advantages when there are no actual choices.
            forced=dict(arrays,candidates=np.ones((1,1),np.float32),candidate_mask=np.ones(1,bool),
                        candidate_refs=np.zeros((1,2),np.float32))
            path=Path(directory)/'forced.pkl.gz'
            writer=TrajectoryWriter(path)
            try:writer.write(Sample(forced,np.zeros(1,np.float32),True,0,.2,np.zeros(1,np.float32)))
            finally:writer.close()
            forced_games=[dict(index=3,seed=3,trajectories=[dict(path=str(path),samples=1,reward=1.)])]
            forced_norm=training_normalization(forced_games,1.)
            self.assertEqual((forced_norm['training_choices'],forced_norm['center'],forced_norm['scale']),(0,0.,1.))
            self.assertIsNone(validate.evaluate(forced_games,forced_norm)['policy_surrogate_gain'])
