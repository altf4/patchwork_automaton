"""Saved-update selection, provenance isolation and guard classification."""
import json
from pathlib import Path
import tempfile
import unittest
from importlib.util import find_spec

from examples.diagnose_rustmage_guard import (
    compare_arrays, exact_checkpoint_root, guard_summary, parser, recorded_partitions,
)


class GuardDiagnosticTests(unittest.TestCase):
    def test_exact_checkpoint_selection_never_changes_source_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'source'
            zero=source/'round-0000';zero.mkdir(parents=True)
            latest=source/'latest.json';latest.write_text('{"checkpoint":"round-0008"}')
            with exact_checkpoint_root(zero) as root:
                self.assertEqual(json.loads((root/'latest.json').read_text()),{'checkpoint':'round-0000'})
                self.assertEqual((root/'round-0000').resolve(),zero.resolve())
                self.assertEqual(latest.read_text(),'{"checkpoint":"round-0008"}')
            self.assertFalse(root.exists())
            self.assertTrue(zero.is_dir())

    def test_partition_uses_recorded_membership_and_relocates_copied_rollouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary);(source/'rollouts').mkdir()
            games=[]
            for i in (2,0,1):
                (source/'rollouts'/f'game-{i}.pkl.gz').write_bytes(b'trusted')
                games.append(dict(index=i,seed=10+i,trajectories=[dict(
                    path=f'/missing/old-project/game-{i}.pkl.gz',samples=1,reward=1.)]))
            report=dict(rounds=[dict(round=1,games=games,split=dict(training_indices=[2,0],validation_indices=[1]))])
            train,heldout,recorded=recorded_partitions(report,source)
            self.assertEqual([g['index'] for g in train],[0,2])
            self.assertEqual([g['index'] for g in heldout],[1])
            self.assertTrue(all(Path(t['path']).is_file() for g in train+heldout for t in g['trajectories']))
            self.assertTrue(report['rounds'][0]['games'][0]['trajectories'][0]['path'].startswith('/missing'))
            with self.assertRaisesRegex(ValueError,'first on-policy'):
                recorded_partitions(report,source,round_id=2)
            recorded['split']['validation_indices']=[0]
            with self.assertRaisesRegex(ValueError,'partition'):
                recorded_partitions(report,source)

    def test_summary_classifies_reasons_without_equating_guard_success_to_strength(self):
        events=[dict(accepted=True,rejection_reasons=[]),
            dict(accepted=False,rejection_reasons=['max_state_kl']),
            dict(accepted=False,rejection_reasons=['mean_kl','max_state_kl']),
            dict(accepted=False,rejection_reasons=['nonfinite'])]
        summary=guard_summary(events)
        self.assertEqual(summary['accepted_proposals'],1)
        self.assertEqual(summary['max_state_vetoes'],2)
        self.assertTrue(summary['max_state_veto_dominates'])
        self.assertFalse(guard_summary([])['max_state_veto_dominates'])

    def test_comparison_exposes_small_gpu_numerical_difference(self):
        import numpy as np
        exact=compare_arrays([np.array([1.],np.float32)],[np.array([1.],np.float32)])
        self.assertTrue(exact['bitwise_equal'])
        drift=compare_arrays([np.array([1.],np.float64)],[np.array([1.+1e-8],np.float64)])
        self.assertFalse(drift['bitwise_equal'])
        self.assertTrue(drift['allclose_1e_7'])
        self.assertGreater(drift['max_absolute_difference'],0)

    def test_cli_defaults_reproduce_saved_guard_limits(self):
        args=parser().parse_args(['--source','pilot','--output','audit'])
        self.assertIsNone(args.max_state_kl)
        self.assertIsNone(args.target_kl)
        self.assertEqual(args.round,1)
        from examples.train_rustmage import parser as training_parser
        training=training_parser().parse_args(['--output','new'])
        self.assertEqual(training.target_kl,.02)
        self.assertEqual(training.max_state_kl,.1)
        self.assertFalse(training.guard_audit)

    @unittest.skipUnless(all(find_spec(n) for n in ('numpy','keras','tensorflow')),'Requires neural dependencies')
    def test_end_to_end_restores_checkpoint_rng_and_matches_original_update(self):
        import keras
        import numpy as np
        import tensorflow as tf
        from examples.diagnose_rustmage_guard import run
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.experiment import make_trainer,save_checkpoint,TrajectoryWriter
        from patchwork_automaton.neural.model import SeparateActorCritic
        from patchwork_automaton.neural.rustmage_features import RustMageEncoder
        from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
        from patchwork_automaton.neural.training import Sample
        keras.utils.set_random_seed(19)
        encoder=RustMageEncoder()
        model=SeparateActorCritic(encoder.card_count,encoder.feature_spec,
            actor_width=8,actor_memory=4,actor_blocks=0,critic_width=8,critic_memory=4)
        batch=example_batch(model.memory_size,encoder.feature_spec)
        for key in ('candidates','candidate_refs','candidate_mask'):
            batch[key]=np.repeat(batch[key],2,axis=1)
        batch['candidates'][:,1,0]=1.
        output=model(batch,training=False)
        logp=np.asarray(tf.nn.log_softmax(output['logits']))[0]
        config=dict(algorithm='ppo',learning_rate=.0001,train_batch=2,ppo_epochs=1,
            seed=19,target_kl=.02,max_state_kl=.1,validation_batch=2,value_max_steps=1)
        trainer=make_trainer(model,config);trainer.compile()
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'pilot';source.mkdir()
            games=[]
            for i in range(3):
                path=source/'rollouts'/f'game-{i}.pkl.gz'
                writer=TrajectoryWriter(path)
                arrays={k:np.asarray(v[0]).copy() for k,v in batch.items() if k not in ('memory_in','advance_memory')}
                writer.write(Sample(arrays,np.asarray(batch['memory_in'][0]).copy(),True,
                    i%2,float(np.asarray(output['value'])[0,0]),logp.copy()))
                writer.close()
                games.append(dict(index=i,seed=10+i,status='win',trajectories=[dict(
                    path=str(path),samples=1,reward=1. if i%2 else -1.)]))
            train=[t for g in games[:2] for t in g['trajectories']]
            trainer.warmup_critic_files(train,max_steps=1)
            save_checkpoint(source/'checkpoints',model,trainer,
                dict(round=0,config=config,critic_warmup={'status':'complete'}),policy_type=RustMagePolicy)
            update=trainer.update_files(train)
            save_checkpoint(source/'checkpoints',model,trainer,dict(round=1,config=config),policy_type=RustMagePolicy)
            report=dict(config=config,rounds=[dict(round=1,games=games,
                split=dict(training_indices=[0,1],validation_indices=[2]),update=update)])
            (source/'report.json').write_text(json.dumps(report))
            result=run(parser().parse_args(['--source',str(source),'--output',str(Path(temporary)/'audit'),
                '--device','cpu','--validation-device','cpu','--save-checkpoint']))
            comparison=result['comparison_to_original_round1']
            self.assertTrue(comparison['actor']['bitwise_equal'])
            self.assertTrue(comparison['actor_optimizer']['bitwise_equal'])
            self.assertTrue(comparison['rng_equal'])
            self.assertEqual(result['initial_optimizer_steps'],0)
            self.assertEqual(result['initial_critic_optimizer_steps'],1)
            self.assertTrue(result['source_unchanged'])
            self.assertTrue(result['inputs_unchanged'])
            self.assertEqual(result['guard_summary']['accepted_proposals'],1)
            self.assertTrue(Path(result['checkpoint']).is_dir())


if __name__ == '__main__':
    unittest.main()
