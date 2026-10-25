"""Warm starts preserve checkpoints while creating an independent experiment."""
import hashlib
import json
import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE, 'Install neural dependencies')
class RustMageTrainingSetupTests(unittest.TestCase):
    def test_warm_start_preserves_weights_and_source_but_resets_adam(self):
        import numpy as np
        from examples.train_rustmage import initialize_model, place_trainer
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.experiment import make_trainer
        from patchwork_automaton.neural.model import SeparateActorCritic
        from patchwork_automaton.neural.rustmage_features import RustMageEncoder
        from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
        encoder = RustMageEncoder()
        model = SeparateActorCritic(encoder.card_count, encoder.feature_spec,
            actor_width=16, actor_memory=8, actor_blocks=0, critic_width=16, critic_memory=8)
        model(example_batch(model.memory_size, model.feature_spec))
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            RustMagePolicy(model).save(source/'policy.keras')
            (source/'state.json').write_text(json.dumps(dict(round=10, config=dict(games_per_round=100))))
            # A warm start intentionally does not import previous Adam state.
            (source/'optimizer.npz').write_bytes(b'previous optimizer must not be loaded')
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            restored, info = initialize_model(SimpleNamespace(initialize_from=source,
                architecture='separate'), encoder)
            for expected, actual in zip(model.get_weights(), restored.get_weights()):
                np.testing.assert_array_equal(actual, expected)
            trainer = make_trainer(restored, dict(algorithm='ppo', learning_rate=.0003,
                train_batch=128, ppo_epochs=2, value_max_steps=125,seed=20261024))
            trainer.compile()
            traces = place_trainer(trainer, 'cpu')
            self.assertTrue(all(count == 1 for count in traces.values()))
            self.assertEqual(int(trainer.optimizer.iterations), 0)
            self.assertEqual(int(trainer.critic_optimizer.iterations), 0)
            self.assertEqual(trainer.rng.bit_generator.state,np.random.default_rng(20266024).bit_generator.state)
            self.assertEqual(info['source_collected_games'], 1000)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()})
            with self.assertRaisesRegex(ValueError, 'architecture'):
                initialize_model(SimpleNamespace(initialize_from=source, architecture='shared'), encoder)

    def test_cli_excludes_ambiguous_resume_initialization(self):
        from examples.train_rustmage import parser
        with self.assertRaises(SystemExit):
            parser().parse_args(['--output', 'unused', '--resume', '--initialize-from', 'source'])

    def test_imitation_warm_start_reports_actual_dataset_partitions(self):
        from examples.train_rustmage import initialize_model
        with tempfile.TemporaryDirectory() as temporary:
            training = Path(temporary)
            source = training/'checkpoints'/'round-0040'
            source.mkdir(parents=True)
            (source/'policy.keras').write_bytes(b'immutable supervised weights')
            (source/'state.json').write_text(json.dumps(dict(round=40,
                config=dict(algorithm='imitation'), round_result=dict(games_seen=10000))))
            (training/'report.json').write_text(json.dumps(dict(rounds=[dict(round=i,
                split=dict(training=dict(attempted_games=200), validation=dict(attempted_games=50)))
                for i in range(1,41)])))
            encoder = SimpleNamespace(card_count=12, feature_spec={'context':87})
            model = SimpleNamespace(separate_value=True, **vars(encoder))
            before = {str(p):p.read_bytes() for p in training.rglob('*') if p.is_file()}
            with patch('examples.train_rustmage.RustMagePolicy.load', return_value=SimpleNamespace(model=model)):
                restored, info = initialize_model(SimpleNamespace(initialize_from=source,
                    architecture='separate'), encoder)
            self.assertIs(restored, model)
            self.assertEqual(info['source_algorithm'], 'imitation')
            self.assertEqual(info['source_collected_games'], 10000)
            self.assertEqual(info['source_optimization_games'], 8000)
            self.assertEqual(info['source_validation_games'], 2000)
            self.assertEqual(before, {str(p):p.read_bytes() for p in training.rglob('*') if p.is_file()})

    def test_resume_warmup_uses_prewarm_until_complete_checkpoint_is_published(self):
        from examples.train_rustmage import resume_checkpoint_root
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.assertEqual(resume_checkpoint_root(output, 1000), output/'prewarm-checkpoints')
            self.assertEqual(resume_checkpoint_root(output, 0), output/'checkpoints')
            prewarm=output/'prewarm-checkpoints'/'round-0000'
            prewarm.mkdir(parents=True)
            (prewarm/'state.json').write_text(json.dumps(dict(round=0,initial_report={'rounds':[]})))
            self.assertEqual(resume_checkpoint_root(output,1000),prewarm.parent)
            self.assertEqual(json.loads((prewarm.parent/'latest.json').read_text()),{'checkpoint':'round-0000'})
            zero = output/'checkpoints'/'round-0000'
            zero.mkdir(parents=True)
            (zero/'state.json').write_text(json.dumps(dict(critic_warmup=dict(status='running'))))
            with self.assertRaisesRegex(ValueError, 'completed critic warm-up'):
                resume_checkpoint_root(output, 1000)
            (zero/'state.json').write_text(json.dumps(dict(critic_warmup=dict(status='complete'))))
            self.assertEqual(resume_checkpoint_root(output,1000), output/'checkpoints')
            self.assertEqual(json.loads((zero.parent/'latest.json').read_text()), {'checkpoint':'round-0000'})

    def test_warmup_freezes_actor_and_adam_and_commits_separate_game_counts(self):
        import numpy as np
        from examples.train_rustmage import critic_warmup
        async def check(output):
            args = SimpleNamespace(output=output, seed=721, critic_warmup_games=10,
                critic_warmup_max_steps=157, train_horizon=60, learner_device='cpu', validation_device='cpu')
            actor = np.array([1.,2.],np.float32)
            adam = np.array([0.,0.],np.float32)
            model = SimpleNamespace(actor=SimpleNamespace(weights=[actor]))
            calls=[]; commits=[]; syncs=[]
            class Trainer:
                optimizer=SimpleNamespace(variables=[adam])
                advantage_scale_floor=1.
                def warmup_critic_files(self, trajectories, *, max_steps):
                    calls.append((trajectories,max_steps))
                    return dict(critic_only=True,critic_steps=157,accepted_actor_steps=0)
            class Validator:
                def evaluate(self,games,normalization):
                    return dict(attempted_games=len(games),recurrent_states='replayed from zero with current weights')
            async def collect(**options):
                self.assertEqual(options['seed_base'],30_000_721)
                self.assertEqual(options['mode'],'selfplay')
                self.assertEqual(options['version'],0)
                self.assertEqual(options['output'],output/'warmup-rollouts'/'attempt-0001')
                return [dict(index=i,seed=options['seed_base']+i,status='win',
                    trajectories=[dict(path=f'game-{i}',samples=2,reward=1. if i%2 else -1.)]) for i in range(10)]
            async def sync(version): syncs.append(version)
            report=dict(config={'critic_warmup_games':10}, rounds=[],training_games_completed=0,
                collected_games_completed=0,optimization_games_completed=0,validation_games_completed=0)
            def save(directory,model,trainer,state,**kwargs):
                commits.append((directory,state))
                self.assertTrue(state['critic_warmup']['actor_unchanged'])
                self.assertTrue(state['critic_warmup']['actor_optimizer_unchanged'])
            with patch('examples.train_rustmage.save_checkpoint',side_effect=save), \
                    patch('examples.train_rustmage.training_normalization',return_value={'center':0.,'scale':1.}) as normalize:
                await critic_warmup(args,model,Trainer(),Validator(),model,report,
                    collect,sync,lambda:0,lambda label:None)
                self.assertEqual(len(normalize.call_args.args[0]),8)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][1],157)
            self.assertEqual(len(calls[0][0]),8)
            warm=report['critic_warmup']
            self.assertEqual(warm['status'],'complete')
            self.assertEqual(warm['split']['training']['attempted_games'],8)
            self.assertEqual(warm['split']['validation']['attempted_games'],2)
            self.assertEqual(warm['generalization_after']['validation']['attempted_games'],2)
            self.assertEqual(commits[0][0],output/'checkpoints')
            self.assertEqual(commits[0][1]['round'],0)
            self.assertEqual(syncs,[0])
            self.assertEqual(report['training_games_completed'],0)
            self.assertEqual(report['collected_games_completed'],0)
            self.assertEqual(report['optimization_games_completed'],0)
            self.assertEqual(report['validation_games_completed'],0)
            self.assertEqual(json.loads((output/'report.json').read_text())['critic_warmup']['status'],'complete')
        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(check(Path(temporary)))

    def test_failed_warmup_does_not_publish_round_zero_and_retry_gets_new_artifacts(self):
        import numpy as np
        from examples.train_rustmage import critic_warmup
        async def check(output):
            args=SimpleNamespace(output=output, seed=1, critic_warmup_games=10,
                critic_warmup_max_steps=None, train_horizon=60, learner_device='cpu',validation_device='cpu')
            actor=np.array([1.],np.float32)
            model=SimpleNamespace(actor=SimpleNamespace(weights=[actor]))
            trainer=SimpleNamespace(optimizer=SimpleNamespace(variables=[]),advantage_scale_floor=1.)
            validator=SimpleNamespace(evaluate=lambda *a:{})
            attempts=[]
            async def collect(**options):
                attempts.append(options['output'])
                return [dict(index=i,seed=i,status='win',trajectories=[dict(path=str(i),samples=1,reward=1.)]) for i in range(10)]
            def update(*a,**kw): actor[0]+=1; return {}
            trainer.warmup_critic_files=update
            report=dict(config={},rounds=[])
            async def sync(version): self.fail('Failed warm-up must not synchronize')
            with patch('examples.train_rustmage.save_checkpoint') as save, \
                    patch('examples.train_rustmage.training_normalization',return_value={'center':0.,'scale':1.}):
                for attempt in range(2):
                    with self.assertRaisesRegex(RuntimeError,'changed actor weights'):
                        await critic_warmup(args,model,trainer,validator,model,report,
                            collect,sync,lambda:0,lambda label:None)
                save.assert_not_called()
            self.assertEqual(attempts,[output/'warmup-rollouts'/'attempt-0001',output/'warmup-rollouts'/'attempt-0002'])
            self.assertEqual(json.loads((output/'report.json').read_text())['critic_warmup']['status'],'running')
        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(check(Path(temporary)))


if __name__ == '__main__':
    unittest.main()
