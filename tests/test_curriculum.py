"""Teacher solutions execute real primitive choices and get terminal rewards."""
import unittest
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE,'Install neural dependencies with uv')
class CurriculumTests(unittest.TestCase):
    def test_context_calibration_accepts_a_fresh_unbuilt_policy(self):
        import numpy as np
        from patchwork_automaton.neural.policy import untrained_policy
        from patchwork_automaton.neural.calibration import calibrate_context
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.features import CONTEXT_WIDTH
        model=untrained_policy(seed=3,width=16,memory_size=16,blocks=0,fact_readout=True).model
        calibrated=calibrate_context(model,np.zeros((4,CONTEXT_WIDTH),np.float32))
        batch=example_batch(16)
        np.testing.assert_allclose(model(batch)['logits'],calibrated(batch)['logits'],atol=1.e-5)

    def test_context_calibration_preserves_logits_value_and_memory(self):
        import numpy as np
        from patchwork_automaton.neural.calibration import calibrate_context
        from patchwork_automaton.neural.policy import untrained_policy
        from patchwork_automaton.neural.features import Encoder,collate
        from patchwork_automaton.neural.curriculum import TASKS,make_episode
        from patchwork_automaton.core import Response
        rng=np.random.default_rng(123);items=[]
        for task in TASKS:
            episode=make_episode(task,rng)
            while not episode.terminal:
                items.append(Encoder().encode(episode.observation()).arrays)
                episode.step(Response(sorted(episode.oracle_options())[0]))
        model=untrained_policy(seed=5,width=16,memory_size=16,blocks=1,
            fact_readout=True,player_readout=True,stable_heads=True).model
        batch=collate(items);batch.update(memory_in=rng.normal(size=(len(items),16)).astype(np.float32),
            advance_memory=np.ones((len(items),1),np.float32))
        expected=model(batch)
        calibrated=calibrate_context(model,np.stack([a['context'] for a in items]))
        for key,value in expected.items():np.testing.assert_allclose(value,calibrated(batch)[key],atol=2.e-5)
        # Recalibrating an already standardized model also preserves its function.
        twice=calibrate_context(calibrated,np.stack([a['context'] for a in items])+0.1)
        for key,value in expected.items():np.testing.assert_allclose(value,twice(batch)[key],atol=2.e-5)

    def test_supervised_balancing_equalizes_prompt_groups_without_changing_rewards(self):
        import numpy as np
        from examples.train_curriculum import demonstration_weights
        examples=[({},None,None,'attack')]*12+[({},None,None,'keep')]*3
        weights=demonstration_weights(examples,True)
        self.assertAlmostEqual(float(weights[:12].sum()),float(weights[12:].sum()))
        self.assertAlmostEqual(float(weights.mean()),1.)
        np.testing.assert_array_equal(demonstration_weights(examples,False),np.ones(15))
        examples=[({},None,None,('combat','cast'))]*12+[({},None,None,('combat','attack'))]*3+[({},None,None,('opening','keep'))]*3
        weights=demonstration_weights(examples,True)
        self.assertAlmostEqual(float(weights[:15].sum()),float(weights[15:].sum()))
        self.assertAlmostEqual(float(weights[:12].sum()),float(weights[12:15].sum()))
        examples=[({},None,None,('recovery','priority',('boolean',)))]*20+[({},None,None,('recovery','priority',('uuid',)))]*2
        weights=demonstration_weights(examples,True)
        self.assertAlmostEqual(float(weights[:20].sum()),float(weights[20:].sum()),places=5)

    def test_all_five_teachers_complete_with_stable_native_objects(self):
        import numpy as np
        from patchwork_automaton.neural.curriculum import TASKS,make_episode
        from patchwork_automaton.neural.features import Encoder
        from patchwork_automaton.core import Response
        from patchwork_automaton.baselines.modern_burn import LAND_ORDER
        from patchwork_automaton.burn_semantics import CARD_REGISTRY
        from patchwork_automaton.semantics import AbilityKind,CostKind
        stages=set();long_opening=False
        for heldout in (False,True):
            for task in TASKS:
                for seed in range(18):
                    episode=make_episode(task,np.random.default_rng(seed),heldout=heldout)
                    sideboard=None
                    while not episode.terminal:
                        self.assertLess(episode.steps,32)
                        obs=episode.observation();self.assertEqual(obs,episode.observation())
                        self.assertNotIn('oracle',obs.native)
                        for obj in obs.objects.values():
                            if obj.zone in ('hand','battlefield','stack'):
                                self.assertEqual('LAND' in obj.properties['cardTypes'],obj.name in LAND_ORDER,obj.name)
                                self.assertEqual('CREATURE' in obj.properties['cardTypes'],obj.name in
                                    ('Goblin Guide','Monastery Swiftspear','Eidolon of the Great Revel'),obj.name)
                                self.assertEqual('ENCHANTMENT' in obj.properties['cardTypes'],obj.name=='Eidolon of the Great Revel')
                                spells=[a for a in CARD_REGISTRY[obj.name].abilities if a.kind==AbilityKind.SPELL]
                                mv=int(sum(c.quantity.value for c in spells[0].costs if c.kind==CostKind.MANA)) if spells else 0
                                self.assertEqual(obj.properties['manaValue'],mv,obj.name)
                        current={o.id for o in obs.objects.values() if o.zone=='sideboard'}
                        self.assertEqual(len(current),15)
                        if sideboard is not None:self.assertEqual(current,sideboard)
                        sideboard=current
                        Encoder().encode(obs)
                        offered={o.id for o in obs.decision.options}
                        self.assertTrue(episode.oracle_options())
                        self.assertFalse(episode.oracle_options()-offered,(task,seed,episode.oracle_options(),offered))
                        if task!='openings':stages.add((task,episode.stage))
                        reward=episode.step(Response(sorted(episode.oracle_options())[0]))
                        self.assertEqual(reward,1. if episode.terminal else None)
                    long_opening|=task=='openings' and episode.steps>3
                    self.assertEqual(episode.reward,1.,(task,seed))
        self.assertTrue(long_opening)
        self.assertIn(('creatures_and_attacking','attack'),stages)
        self.assertIn(('lands_and_casting','shock'),stages)
        self.assertIn(('mana_colors','ability'),stages)

    def test_lethal_requires_resolution_and_self_burn_can_lose(self):
        import numpy as np
        from patchwork_automaton.neural.curriculum import make_episode
        from patchwork_automaton.core import Response
        for seed in range(20):
            episode=make_episode('lethal',np.random.default_rng(seed))
            initial=episode.enemy_life
            while not episode.terminal:
                response=Response(sorted(episode.oracle_options())[0])
                episode.step(response)
            self.assertLessEqual(episode.enemy_life,0)
            self.assertEqual(len(episode.graveyard),1 if initial<=3 else 2)

    def test_passing_before_land_or_before_attack_fails(self):
        import numpy as np
        from patchwork_automaton.neural.curriculum import make_episode
        from patchwork_automaton.core import Response
        for seed in range(20):
            episode=make_episode('creatures_and_attacking',np.random.default_rng(seed))
            self.assertEqual(episode.step(Response('no')),-1.)
            episode=make_episode('creatures_and_attacking',np.random.default_rng(seed))
            while episode.stage!='attack':episode.step(Response(sorted(episode.oracle_options())[0]))
            self.assertEqual(episode.step(Response('no')),-1.)

    def test_wrong_color_and_cancel_fail_without_intermediate_rewards(self):
        import numpy as np
        from patchwork_automaton.neural.curriculum import make_episode
        from patchwork_automaton.core import Response
        episode=make_episode('mana_colors',np.random.default_rng(2))
        self.assertIsNone(episode.step(Response('object:'+episode.land)))
        self.assertIsNone(episode.step(Response('object:'+(episode.white if episode.wanted=='red' else episode.red))))
        self.assertEqual({o.id for o in episode.observation().decision.options},
            {'no','mana:WHITE' if episode.wanted=='red' else 'mana:RED'})
        self.assertEqual(episode.step(Response('no')),-1.)
