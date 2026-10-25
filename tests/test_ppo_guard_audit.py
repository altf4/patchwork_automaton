"""Opt-in guard telemetry preserves updates and identifies rejected proposals."""
import json
from pathlib import Path
import tempfile
import unittest
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE, 'Requires neural dependencies')
class PPOGuardAuditTests(unittest.TestCase):
    def fixture(self, directory):
        from test_critic_warmup import CriticWarmupTests
        return CriticWarmupTests().fixture(directory,cache_bytes=16*1024**2)

    def test_audit_preserves_model_adam_rng_and_accepted_updates_exactly(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            model,plain,trajectories = self.fixture(Path(directory)/'plain')
            other,audited,_ = self.fixture(Path(directory)/'audited')
            other.set_weights(model.get_weights())
            for name in ('optimizer','critic_optimizer'):
                for left,right in zip(getattr(plain,name).variables,getattr(audited,name).variables):
                    right.assign(np.asarray(left))
            observed = []
            audited.guard_audit = True
            audited.guard_observer = observed.append
            reference = plain.update_files(trajectories)
            result = audited.update_files(trajectories)
            self.assertNotIn('guard_proposals',reference)
            self.assertTrue(observed)
            self.assertEqual(observed,result['guard_proposals'])
            json.dumps(observed,allow_nan=False)
            for key in ('accepted_steps','rejected_epochs','early_stop','optimizer_steps','critic_steps',
                        'loss','policy_loss','value_loss','entropy','exact_kl','max_state_kl'):
                self.assertEqual(result[key],reference[key],key)
            for left,right in zip(model.variables,other.variables):
                np.testing.assert_array_equal(np.asarray(left),np.asarray(right))
            for name in ('optimizer','critic_optimizer'):
                for left,right in zip(getattr(plain,name).variables,getattr(audited,name).variables):
                    np.testing.assert_array_equal(np.asarray(left),np.asarray(right))
            self.assertEqual(plain.rng.bit_generator.state,audited.rng.bit_generator.state)

    def test_quantiles_worst_record_and_immediate_rejection_before_rollback(self):
        import numpy as np
        from patchwork_automaton.neural.experiment import read_trajectory
        from patchwork_automaton.neural.ppo import normalized_advantages
        with tempfile.TemporaryDirectory() as directory:
            model,trainer,trajectories = self.fixture(directory)
            trainer.guard_audit = True
            trainer.epochs = 1
            trainer.backtracks = 0
            trainer.target_kl = .04
            trainer.max_state_kl = .1
            actor = [np.asarray(v).copy() for v in model.actor.variables]
            adam = [np.asarray(v).copy() for v in trainer.optimizer.variables]
            values = np.array([.001,.01,.02,.03,.04,.05,.06,.2])
            trainer.measure_actor_with_drift = lambda *args: (np.zeros(6),np.array([values.mean(),values.max()]))
            cursor = [0]
            def per_state(inputs, old, actions):
                length = len(actions); start = cursor[0]; cursor[0] += length
                chosen = np.exp(np.asarray(old)[np.arange(length),np.asarray(actions)])
                return values[start:start+length],np.ones(length,bool),chosen,chosen*.8
            trainer.actor_state_drift = per_state
            seen = []
            trainer.guard_observer = lambda event: seen.append((event,int(trainer.optimizer.iterations)))
            result = trainer.update_files(trajectories)
            event,counter_during_proposal = seen[0]
            self.assertFalse(event['accepted'])
            self.assertTrue(event['finite'])
            self.assertEqual(event['rejection_reasons'],['mean_kl','max_state_kl'])
            self.assertEqual(event['epoch'],1)
            self.assertEqual(event['attempt'],0)
            self.assertAlmostEqual(event['effective_learning_rate'],trainer.learning_rate)
            self.assertEqual(event['choice_states'],8)
            self.assertEqual(event['states_above_max_state_kl'],1)
            np.testing.assert_allclose(list(event['kl_quantiles'].values()),np.quantile(values,[.5,.95,.99,.999]))
            worst = event['worst_states'][0]
            self.assertEqual(worst['trajectory_path'],trajectories[1]['path'])
            self.assertEqual(worst['sample_offset'],3)
            self.assertEqual(worst['sample_index'],7)
            self.assertEqual(worst['action_index'],1)
            sample = list(read_trajectory(trajectories[1]['path']))[3]
            self.assertAlmostEqual(worst['old_selected_probability'],float(np.exp(sample.log_probs[1])))
            self.assertAlmostEqual(worst['new_selected_probability'],worst['old_selected_probability']*.8)
            samples = [s for t in trajectories for s in read_trajectory(t['path'])]
            advantages = normalized_advantages([1.]*4+[-1.]*4,[s.value for s in samples],np.ones(8,bool),trainer.advantage_scale_floor)
            self.assertEqual(worst['advantage'],float(advantages[-1]))
            self.assertEqual(result['accepted_steps'],0)
            self.assertEqual(result['rejected_epochs'],1)
            self.assertGreater(counter_during_proposal,int(trainer.optimizer.iterations))
            for before,after in zip(actor,model.actor.variables):np.testing.assert_array_equal(before,np.asarray(after))
            for before,after in zip(adam,trainer.optimizer.variables):np.testing.assert_array_equal(before,np.asarray(after))

    def test_reasons_distinguish_mean_max_multiple_and_nonfinite(self):
        import numpy as np
        from patchwork_automaton.neural.streaming_ppo import _guard_status, _json_number
        cases = [([0.],[.01,.05],True,True,[]),
                 ([0.],[.03,.05],False,True,['mean_kl']),
                 ([0.],[.01,.2],False,True,['max_state_kl']),
                 ([0.],[.03,.2],False,True,['mean_kl','max_state_kl']),
                 ([np.nan],[.01,.05],False,False,['nonfinite']),
                 ([np.nan],[.03,.2],False,False,['nonfinite','mean_kl','max_state_kl'])]
        for proposed,drift,accepted,finite,reasons in cases:
            with self.subTest(proposed=proposed,drift=drift):
                self.assertEqual(_guard_status(proposed,drift,.02,.1),(accepted,finite,reasons))
        self.assertIsNone(_json_number(np.nan))
        self.assertIsNone(_json_number(np.inf))

    def test_concurrent_spill_audit_holds_packing_lock_for_every_record_read(self):
        import sys
        import threading
        from types import SimpleNamespace
        from unittest.mock import patch
        from patchwork_automaton.neural import streaming_ppo
        from test_critic_warmup import CriticWarmupTests
        with tempfile.TemporaryDirectory() as directory:
            _,trainer,trajectories = CriticWarmupTests().fixture(directory,cache_bytes=0)
            trainer.guard_audit = True
            trainer.epochs = 1
            class OwnedLock:
                def __init__(self):
                    self.lock, self.owner = threading.Lock(), None
                def __enter__(self):
                    self.lock.acquire()
                    self.owner = threading.get_ident()
                    return self
                def __exit__(self, *args):
                    self.owner = None
                    self.lock.release()
            packing_lock = OwnedLock()
            worst_reads = []
            original = streaming_ppo.IndexedTrajectories
            case = self
            class CheckedIndexedTrajectories(original):
                def get(self, index, position):
                    case.assertEqual(packing_lock.owner,threading.get_ident(),
                                     'Shared trajectory read outside packing lock')
                    if sys._getframe(1).f_code.co_name == 'audit_states':
                        worst_reads.append((index,position))
                    return super().get(index,position)
            with patch.object(streaming_ppo,'threading',SimpleNamespace(Lock=lambda:packing_lock)), \
                 patch.object(streaming_ppo,'IndexedTrajectories',CheckedIndexedTrajectories):
                result = trainer.update_files(trajectories)
            self.assertTrue(trainer.concurrent_updates)
            self.assertGreater(trainer.prefetch_batches,0)
            self.assertGreater(result['cache']['spill_bytes'],0)
            self.assertTrue(worst_reads)
            self.assertEqual(len(worst_reads),sum(len(e['worst_states']) for e in result['guard_proposals']))


if __name__ == '__main__':
    unittest.main()
