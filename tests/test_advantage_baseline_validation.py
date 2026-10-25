"""Actor baseline selection must not rewrite value diagnostics or validation data."""
import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE,'Requires neural dependencies')
class AdvantageBaselineValidationTests(unittest.TestCase):
    def test_zero_normalization_and_surrogate_preserve_true_value_diagnostics(self):
        import numpy as np
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.experiment import TrajectoryWriter,read_trajectory
        from patchwork_automaton.neural.training import Sample
        from patchwork_automaton.neural.validation import training_normalization,GameValidator
        class Toy:
            memory_size=1
            feature_spec=dict(entities=1,semantics=1,context=1,candidates=1,history=1)
            def __call__(self,*args,**kwargs):
                raise AssertionError('Test installs a deterministic numeric validator forward')
        model=Toy()
        arrays={k:v[0] for k,v in example_batch(1,model.feature_spec).items()
            if k not in ('memory_in','advance_memory')}
        arrays.update(candidates=np.ones((2,1),np.float32),candidate_mask=np.ones(2,bool),
                      candidate_refs=np.zeros((2,2),np.float32))
        with tempfile.TemporaryDirectory() as temporary:
            games=[]
            for i,(reward,value) in enumerate([(1.,.8),(-1.,-.2)]):
                path=Path(temporary)/f'{i}.pkl.gz'
                writer=TrajectoryWriter(path)
                writer.write(Sample(arrays,np.zeros(1,np.float32),True,i,value,
                    np.log(np.array([.5,.5],np.float32))))
                writer.close()
                games.append(dict(index=i,seed=i,trajectories=[dict(path=str(path),samples=1,reward=reward)]))
            results={}
            for baseline in ('critic','zero'):
                norm=training_normalization(games,1.,advantage_baseline=baseline)
                self.assertEqual(norm['advantage_baseline'],baseline)
                self.assertAlmostEqual(norm['center'], -.3 if baseline=='critic' else 0.,places=6)
                self.assertEqual(norm['scale'],1.)
                validator=GameValidator(model,advantage_baseline=baseline)
                def forward(inputs):
                    count=len(inputs['memory_in'])
                    return (np.full(count,.3,np.float32),
                        np.tile(np.log([.75,.25]),(count,1)),inputs['memory_in'])
                validator.forward=forward
                results[baseline]=validator.evaluate(games,norm)
                self.assertEqual(results[baseline]['advantage_baseline'],baseline)
                self.assertAlmostEqual(results[baseline]['policy_surrogate_gain']['mean'],
                    .1 if baseline=='critic' else .2,places=6)
                with self.assertRaisesRegex(ValueError,'baselines must match'):
                    validator.evaluate(games,dict(norm,advantage_baseline='zero' if baseline=='critic' else 'critic'))
            self.assertEqual(results['critic']['value_before'],results['zero']['value_before'])
            self.assertEqual(results['critic']['value_after'],results['zero']['value_after'])
            self.assertAlmostEqual(results['zero']['value_before']['mse'],.34,places=6)
            self.assertAlmostEqual(results['zero']['value_after']['mse'],1.09,places=6)
            self.assertEqual([next(read_trajectory(g['trajectories'][0]['path'])).value for g in games],[.8,-.2])
            train_norm=training_normalization(games[:1],1.,advantage_baseline='zero')
            games[1]['trajectories'][0]['reward']=0.
            self.assertEqual(train_norm,training_normalization(games[:1],1.,advantage_baseline='zero'))

    def test_cli_defaults_and_legacy_resume_reject_silent_baseline_switch(self):
        from examples.train_rustmage import parser,run_experiment
        default=parser().parse_args(['--output','unused'])
        self.assertEqual(default.advantage_baseline,'critic')
        self.assertIsNone(default.baseline_eval_seed)
        args=parser().parse_args(['--output','unused','--resume','--advantage-baseline','zero',
            '--baseline-eval-seed','45261024'])
        self.assertEqual(args.baseline_eval_seed,45261024)
        config={k:v for k,v in vars(args).items() if k not in ('output','resume','rounds','plot','initialize_from','advantage_baseline')}
        config.update(inference_processes=2,algorithm='ppo',advantage_scale_floor=1.,ppo_clip=.2)
        with patch('examples.train_rustmage.load_checkpoint',return_value=(None,None,dict(round=0,config=config))):
            with self.assertRaisesRegex(ValueError,'same experiment configuration'):
                asyncio.run(run_experiment(args))

    def test_checkpoint_restores_zero_mode_and_legacy_defaults_to_critic(self):
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.experiment import make_trainer,save_checkpoint,load_checkpoint
        from patchwork_automaton.neural.model import SeparateActorCritic
        from patchwork_automaton.neural.rustmage_features import RustMageEncoder
        from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
        encoder=RustMageEncoder()
        model=SeparateActorCritic(encoder.card_count,encoder.feature_spec,
            actor_width=8,actor_memory=4,actor_blocks=0,critic_width=8,critic_memory=4)
        model(example_batch(model.memory_size,model.feature_spec))
        config=dict(algorithm='ppo',learning_rate=.0001,train_batch=2,ppo_epochs=1,
                    advantage_baseline='zero')
        trainer=make_trainer(model,config);trainer.compile()
        with tempfile.TemporaryDirectory() as temporary:
            save_checkpoint(temporary,model,trainer,dict(round=0,config=config),policy_type=RustMagePolicy)
            _,restored,state=load_checkpoint(temporary)
            self.assertEqual(restored.advantage_baseline,'zero')
            self.assertEqual(state['config']['advantage_baseline'],'zero')
        legacy=dict(config);legacy.pop('advantage_baseline')
        self.assertEqual(make_trainer(model,legacy).advantage_baseline,'critic')


if __name__ == '__main__':
    unittest.main()
