"""Recipe flags remain explicit, checkpointed and compatible with old defaults."""
import asyncio
from pathlib import Path
from unittest.mock import patch
import unittest
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE,'Requires neural dependencies')
class TrainingRecipeConfigurationTests(unittest.TestCase):
    def test_defaults_and_factory_forward_explicit_recipes(self):
        from examples.train_rustmage import parser
        from patchwork_automaton.neural.experiment import make_trainer
        default=parser().parse_args(['--output','unused'])
        self.assertEqual(default.entropy_weight,.01)
        self.assertEqual(default.critic_update_scope,'all')
        self.assertEqual(default.critic_unroll,1)
        args=parser().parse_args(['--output','unused','--entropy-weight','0',
            '--critic-update-scope','head','--critic-unroll','8'])
        config=dict(vars(args),algorithm='ppo')
        with patch('patchwork_automaton.neural.streaming_ppo.StreamingPPOTrainer') as constructor:
            make_trainer(object(),config)
            self.assertEqual(constructor.call_args.kwargs['entropy_weight'],0.)
            self.assertEqual(constructor.call_args.kwargs['critic_update_scope'],'head')
            self.assertEqual(constructor.call_args.kwargs['critic_unroll'],8)
            for key in ('entropy_weight','critic_update_scope','critic_unroll'):
                config.pop(key)
            make_trainer(object(),config)
            self.assertEqual(constructor.call_args.kwargs['entropy_weight'],.01)
            self.assertEqual(constructor.call_args.kwargs['critic_update_scope'],'all')
            self.assertEqual(constructor.call_args.kwargs['critic_unroll'],1)

    def test_legacy_resume_accepts_defaults_but_rejects_recipe_switch(self):
        from examples.train_rustmage import parser,run_experiment
        class ReachedRuntime(Exception): pass
        args=parser().parse_args(['--output','unused','--resume'])
        config={k:v for k,v in vars(args).items() if k not in
            ('output','resume','rounds','plot','initialize_from','entropy_weight','critic_update_scope','critic_unroll')}
        config.update(inference_processes=2,algorithm='ppo',advantage_scale_floor=1.,ppo_clip=.2)
        state=dict(round=0,config=config,initial_report=dict(rounds=[],evaluations=[]))
        with patch('examples.train_rustmage.load_checkpoint',return_value=(None,None,state)), \
                patch('examples.train_rustmage.assert_model_device',side_effect=ReachedRuntime):
            with self.assertRaises(ReachedRuntime): asyncio.run(run_experiment(args))
        for flags in (['--entropy-weight','0'],['--critic-update-scope','head'],['--critic-unroll','8']):
            changed=parser().parse_args(['--output','unused','--resume',*flags])
            with patch('examples.train_rustmage.load_checkpoint',return_value=(None,None,state)):
                with self.assertRaisesRegex(ValueError,'same experiment configuration'):
                    asyncio.run(run_experiment(changed))

    def test_invalid_recipes_fail_before_output_creation(self):
        from examples.train_rustmage import parser,run_experiment
        for flags,reason in [(['--entropy-weight','-1'],'Entropy'),
                (['--entropy-weight','nan'],'Entropy'),(['--critic-unroll','0'],'unroll'),
                (['--architecture','shared','--critic-update-scope','head'],'separate architecture'),
                (['--architecture','shared','--critic-unroll','8'],'separate architecture')]:
            args=parser().parse_args(['--output','unused',*flags])
            with self.subTest(flags=flags),self.assertRaisesRegex(ValueError,reason):
                asyncio.run(run_experiment(args))


if __name__ == '__main__':
    unittest.main()
