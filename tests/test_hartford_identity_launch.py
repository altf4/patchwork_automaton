"""Validate preparation without collecting games or applying optimizer updates."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from examples import train_hartford_ppo as driver


class IdentityLaunchTests(unittest.TestCase):
    def test_transfer_is_explicit_and_hash_pinned(self):
        with self.assertRaisesRegex(ValueError,'requires --initialize-from'):
            driver.configuration(driver.parser().parse_args(['--architecture','compact-identity']))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'policy.keras';p.write_bytes(b'configuration-only')
            sha=hashlib.sha256(p.read_bytes()).hexdigest()
            args=['--architecture','compact-identity','--initialize-from',str(p),'--initialize-from-sha256',sha,
                '--rounds','100','--seed','412610510000','--skip-evaluations']
            c=driver.configuration(driver.parser().parse_args(args))
            self.assertEqual(c['rounds']*c['games_per_round'],100000)
            self.assertEqual(c['architecture'],'compact-identity-standard-v2')
            self.assertEqual(c['initialization'],'transfer_from_compact_v1')
            self.assertTrue(c['use_identity']);self.assertFalse(c['use_rules'])
            self.assertEqual(c['game_timeout'],20)
            self.assertTrue(c['skip_evaluations'])
            self.assertEqual(driver.memory_runtime_policy(c)['input_budget_mib'],512)
            original=driver.configuration(driver.parser().parse_args(['--skip-evaluations']))
            for key in ('learning_rate','value_learning_rate','train_batch','ppo_epochs','entropy_weight',
                    'target_kl','max_state_kl','ppo_clip','validation_fraction','advantage_scale_floor',
                    'value_max_steps','workers','runners','inference_processes','opponent_inference_processes'):
                self.assertEqual(c[key],original[key],key)
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                driver.configuration(driver.parser().parse_args(args))

    def test_legacy_and_default_runs_reject_transfer_arguments(self):
        for architecture in ('legacy','compact'):
            with self.assertRaisesRegex(ValueError,'requires --architecture'):
                driver.configuration(driver.parser().parse_args(['--architecture',architecture,
                    '--initialize-from','unused']))

    def test_cli_initializes_transfer_with_zero_optimizer_steps(self):
        import numpy as np
        from patchwork_automaton.neural.standard_policy import StandardPolicy
        from patchwork_automaton.neural.compact_standard_model import CompactActorCritic
        from patchwork_automaton.neural.compact_identity_model import TensorStore
        from patchwork_automaton.neural.compact_standard import pack as old_pack
        from patchwork_automaton.neural.compact_standard_model import TensorStore as OldStore
        from patchwork_automaton.neural.compact_identity import pack
        from test_compact_identity import record
        import tensorflow as tf
        with tempfile.TemporaryDirectory() as tmp:
            source=CompactActorCritic();source(source.example_inputs())
            path=Path(tmp)/'policy.keras';StandardPolicy(source).save(path)
            digest=driver.sha256(path)
            args=driver.parser().parse_args(['--architecture','compact-identity','--initialize-from',str(path),
                '--initialize-from-sha256',digest,'--skip-evaluations'])
            model,trainer=driver.initialize_random(driver.configuration(args),device='cpu')
            self.assertEqual((int(trainer.optimizer.iterations),int(trainer.critic_optimizer.iterations)),(0,0))
            records=[record(),record(lands=1)]
            old=source(OldStore(old_pack(records)).batch(tf.constant([0,1])))
            new=model(TensorStore(pack(records)).batch(tf.constant([0,1])))
            for key in ('logits','value'):np.testing.assert_array_equal(old[key],new[key])
            self.assertEqual(driver.sha256(path),digest)


if __name__=='__main__':unittest.main()
