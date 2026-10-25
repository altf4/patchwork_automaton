"""Numeric storage must preserve frozen PPO records and guarded updates."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE,'Install neural dependencies')
class IndexedRolloutTests(unittest.TestCase):
    def records(self):
        from patchwork_automaton.neural.training import Sample
        rng=np.random.default_rng(19)
        for i in range(9):
            yield Sample(dict(entities=rng.normal(size=(i+2,7)).astype(np.float32),
                              card_ids=np.arange(i+2,dtype=np.int32),
                              entity_mask=np.ones(i+2,bool),
                              semantics=rng.normal(size=(i+2,1,3)).astype(np.float32)),
                         rng.normal(size=5).astype(np.float32),i%2==0,i,1./(i+2),
                         None if i==0 else rng.normal(size=i+2).astype(np.float64))

    def assert_records_equal(self,a,b):
        self.assertEqual((a.action,a.value,a.advance),(b.action,b.value,b.advance))
        self.assertEqual(list(a.arrays),list(b.arrays))
        for key in a.arrays:
            self.assertEqual(a.arrays[key].dtype,b.arrays[key].dtype)
            np.testing.assert_array_equal(a.arrays[key],b.arrays[key])
        np.testing.assert_array_equal(a.memory,b.memory)
        if a.log_probs is None:self.assertIsNone(b.log_probs)
        else:
            self.assertEqual(a.log_probs.dtype,b.log_probs.dtype)
            np.testing.assert_array_equal(a.log_probs,b.log_probs)

    def test_roundtrip_random_access_and_records_survive_reader_close(self):
        from patchwork_automaton.neural.indexed_rollouts import IndexedTrajectoryWriter,IndexedTrajectoryReader
        with tempfile.TemporaryDirectory() as directory:
            expected=list(self.records())
            writer=IndexedTrajectoryWriter(Path(directory)/'game.pkl.gz')
            for record in expected:writer.write(record)
            writer.close();writer.close()
            self.assertEqual(writer.count,len(expected))
            with IndexedTrajectoryReader(writer.path) as reader:
                self.assertEqual(len(reader),len(expected))
                actual=[reader.get(i) for i in (8,0,4,4,2)]
                for i in (-1,9):
                    with self.assertRaises(IndexError):reader.get(i)
            for index,record in zip((8,0,4,4,2),actual):
                self.assert_records_equal(expected[index],record)
                self.assertTrue(all(a.flags.owndata for a in record.arrays.values()))
            with self.assertRaises(ValueError):reader.get(0)

    def test_empty_partial_truncated_and_schema_changes(self):
        from patchwork_automaton.neural.indexed_rollouts import IndexedTrajectoryWriter,IndexedTrajectoryReader
        with tempfile.TemporaryDirectory() as directory:
            writer=IndexedTrajectoryWriter(Path(directory)/'empty.pkl.gz');writer.close()
            with IndexedTrajectoryReader(writer.path) as reader:self.assertEqual(list(reader),[])
            writer=IndexedTrajectoryWriter(Path(directory)/'partial.pkl.gz')
            writer.write(next(self.records()));writer.file.flush()
            with self.assertRaises(ValueError):IndexedTrajectoryReader(writer.path)
            writer.close();content=writer.path.read_bytes()
            writer.path.write_bytes(content[:-20])
            with self.assertRaises((ValueError,UnicodeError)):IndexedTrajectoryReader(writer.path)
            writer=IndexedTrajectoryWriter(Path(directory)/'invalid.pkl.gz')
            record=next(self.records());writer.write(record)
            record.arrays=dict(record.arrays,extra=np.ones(1))
            with self.assertRaisesRegex(ValueError,'keys changed'):writer.write(record)
            writer.close()
            with self.assertRaises(ValueError):IndexedTrajectoryReader(writer.path)

    def test_reader_dispatch_and_direct_random_access_avoid_pickle_spill(self):
        from patchwork_automaton.neural.experiment import TrajectoryWriter,read_trajectory
        from patchwork_automaton.neural.rollout_cache import IndexedTrajectories,RolloutCache
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ,PATCHWORK_ROLLOUT_STORAGE='indexed'):
                writer=TrajectoryWriter(Path(directory)/'game.pkl.gz')
                expected=list(self.records())
                for record in expected:writer.write(record)
                writer.close()
            self.assertEqual(writer.path.suffix,'.rlt')
            for a,b in zip(expected,read_trajectory(writer.path),strict=True):self.assert_records_equal(a,b)
            descriptor=dict(path=str(writer.path),samples=len(expected),reward=1.)
            with tempfile.TemporaryFile() as spill:
                indexed=IndexedTrajectories([descriptor],RolloutCache(0),read_trajectory,spill)
                try:
                    for i in (8,1,6,0):self.assert_records_equal(expected[i],indexed.get(0,i))
                    self.assertEqual(indexed.disk_bytes,0)
                    self.assertEqual(spill.tell(),0)
                finally:indexed.close()

    def test_random_access_bounds_open_files_and_closes_evicted_readers(self):
        from patchwork_automaton.neural.indexed_rollouts import IndexedTrajectoryWriter
        from patchwork_automaton.neural.experiment import read_trajectory
        from patchwork_automaton.neural.rollout_cache import IndexedTrajectories,RolloutCache
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile() as spill:
            expected=next(self.records());descriptors=[]
            for i in range(70):
                writer=IndexedTrajectoryWriter(Path(directory)/f'{i}.pkl.gz')
                writer.write(expected);writer.close()
                descriptors.append(dict(path=str(writer.path),samples=1,reward=1.))
            indexed=IndexedTrajectories(descriptors,RolloutCache(0),read_trajectory,spill)
            try:
                self.assert_records_equal(expected,indexed.get(0,0))
                first=indexed.numeric_readers[0]
                for i in range(1,70):self.assert_records_equal(expected,indexed.get(i,0))
                self.assertEqual(len(indexed.numeric_readers),64)
                self.assertIsNone(first.fd)
                readers=list(indexed.numeric_readers.values())
            finally:indexed.close()
            self.assertTrue(all(r.fd is None for r in readers))

    def test_numeric_storage_preserves_concurrent_ppo_rng_backtracking_and_cache_limits(self):
        from test_concurrent_ppo import ConcurrentPPOTests
        from patchwork_automaton.neural.experiment import TrajectoryWriter,read_trajectory
        helper=ConcurrentPPOTests()
        with tempfile.TemporaryDirectory() as directory:
            model,originals,_=helper.fixture(directory)
            other,_,_=helper.fixture(Path(directory)/'other');other.set_weights(model.get_weights())
            numeric=[]
            with patch.dict(os.environ,PATCHWORK_ROLLOUT_STORAGE='indexed'):
                for index,original in enumerate(originals):
                    writer=TrajectoryWriter(Path(directory)/f'numeric-{index}.pkl.gz')
                    for record in read_trajectory(original['path']):writer.write(record)
                    writer.close();numeric.append(dict(original,path=str(writer.path)))
            from patchwork_automaton.neural.validation import GameValidator,training_normalization
            games=[[dict(index=i,seed=20+i,trajectories=[t]) for i,t in enumerate(ts)] for ts in (originals,numeric)]
            norms=[training_normalization(g,1.) for g in games]
            self.assertEqual(*norms)
            validator=GameValidator(model,batch_size=4)
            before=[v.copy() for v in model.get_weights()]
            self.assertEqual(validator.evaluate(games[0],norms[0]),validator.evaluate(games[1],norms[1]))
            for a,b in zip(before,model.get_weights()):np.testing.assert_array_equal(a,b)
            # Force file access and concurrent preparation/measurement. Identical
            # RNG permutations and backtracking decisions must yield identical state.
            trainers=[helper.trainer(m,True,0,prefetch_batches=2,measurement_workers=3) for m in (model,other)]
            results=[]
            for trainer,trajectories in zip(trainers,(originals,numeric)):
                forward=trainer.measure_actor_with_drift;first=[True]
                def reject_once(*args):
                    metrics,drift=forward(*args)
                    if first[0]:first[0]=False;return metrics,np.array([1.,1.])
                    return metrics,drift
                trainer.measure_actor_with_drift=reject_once
                results.append(trainer.update_files(trajectories))
            helper.assert_state_equal(*trainers)
            for key in ('samples','accepted_steps','critic_steps','rejected_epochs','loss','value_loss','exact_kl'):
                self.assertEqual(results[0][key],results[1][key])
            self.assertGreater(results[0]['cache']['spill_bytes'],0)
            self.assertEqual(results[1]['cache']['spill_bytes'],0)
            self.assertEqual(results[1]['cache']['retained_bytes'],0)
