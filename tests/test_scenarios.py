"""Diagnostic environment validity, counterexamples and input isolation."""
import unittest
from dataclasses import replace
from test_neural import AVAILABLE

@unittest.skipUnless(AVAILABLE,'Install neural dependencies with uv')
class ScenarioTests(unittest.TestCase):
    def test_rewards_agree_with_mana_requirement_and_combat_continuation(self):
        import numpy as np
        from patchwork_automaton.neural.scenarios import TACTICAL_TASKS,sample
        from patchwork_automaton.burn_semantics import CARD_REGISTRY
        from patchwork_automaton.semantics import Color
        for distribution in ('legacy','diverse'):
            for heldout in (False,True):
                rng=np.random.default_rng(13)
                variants=set()
                for task in TACTICAL_TASKS:
                    for _ in range(30):
                        s=sample(task,rng,heldout=heldout,distribution=distribution);obs=s.observation;variants.add(s.variant)
                        me=next(p for p in obs.players if p.id==obs.actor)
                        opp=next(p for p in obs.players if p.id!=obs.actor)
                        for o in obs.decision.options:
                            expected=False
                            if task=='mana' and 'semantic_ability' in o.constraints:
                                card=next(c for c in CARD_REGISTRY.values() if c.id==o.constraints['semantic_card'])
                                a=next(a for a in card.abilities if a.id==o.constraints['semantic_ability'])
                                color=a.effects[0].color
                                expected=(color==Color.RED and me.native['manaPool']['white']==1 or
                                          color==Color.WHITE and me.native['manaPool']['red']==1)
                            elif task=='bolt_target':expected=o.id=='object:'+opp.id.value and opp.life<=3
                            elif task=='charm_mode':
                                creature=next(o for o in obs.objects.values() if o.name=='Monastery Swiftspear')
                                # Successfully casting Charm adds one prowess power.
                                combat=(int(creature.properties['power'])+1) if creature.properties['attacking'] else 0
                                ability=o.constraints.get('semantic_ability')
                                damage=4+combat if ability==901 else 2*combat if ability==903 else combat if ability==902 else 0
                                expected=damage>=opp.life
                            self.assertEqual(s.reward(o.id)==1,expected,(task,s.variant,o))
                self.assertEqual(len(variants),5)

    def test_oracle_not_encoded_and_candidate_order_is_not_the_answer(self):
        import numpy as np
        from patchwork_automaton.neural.scenarios import TASKS,sample
        from patchwork_automaton.neural.features import Encoder
        encoder=Encoder();rng=np.random.default_rng(14)
        for task in TASKS:
            winning_positions=set()
            for _ in range(20):
                s=sample(task,rng)
                winning_positions.add(next(i for i,o in enumerate(s.observation.decision.options) if o.id in s.winning_options))
                wrong=replace(s,winning_options=frozenset())
                for k,v in encoder.encode(s.observation).arrays.items():
                    np.testing.assert_array_equal(v,encoder.encode(wrong.observation).arrays[k])
            self.assertGreater(len(winning_positions),1)

    def test_mana_counterfactual_changes_numeric_state_but_not_candidate_identity(self):
        import numpy as np
        from patchwork_automaton.neural.scenarios import sample
        from patchwork_automaton.neural.features import Encoder
        s=sample('mana',np.random.default_rng(22));obs=s.observation
        before=Encoder().encode(obs).arrays
        me=next(p for p in obs.players if p.id==obs.actor)
        pool=dict(me.native['manaPool']);pool['red'],pool['white']=pool['white'],pool['red']
        swapped=replace(me,native={**me.native,'manaPool':pool})
        changed=replace(obs,players=tuple(swapped if p.id==me.id else p for p in obs.players))
        after=Encoder().encode(changed).arrays
        self.assertEqual(np.count_nonzero(before['entities']!=after['entities']),2)
        np.testing.assert_array_equal(before['candidates'],after['candidates'])
        np.testing.assert_array_equal(before['candidate_refs'],after['candidate_refs'])

    def test_stable_heads_roundtrip_and_candidate_permutation(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from patchwork_automaton.neural.scenarios import sample
        from patchwork_automaton.neural.policy import NeuralPolicy,untrained_policy
        from examples.train_scenarios import batch_for
        policy=untrained_policy(seed=9,width=16,memory_size=16,blocks=1,
                                candidate_interactions=True,player_readout=True,stable_heads=True,standardize_inputs=True,isolate_value=True,fact_readout=True)
        s=sample('mana',np.random.default_rng(3))
        inputs=batch_for([s],policy.encoder,16)
        policy.model.entity_standardization.adapt(inputs['entities'][0])
        policy.model.candidate_standardization.adapt(inputs['candidates'][0])
        policy.model.context_standardization.adapt(inputs['context'])
        original=policy.model(inputs)
        perm=np.arange(inputs['candidates'].shape[1])[::-1]
        shuffled={k:(v[:,perm] if k in ('candidates','candidate_refs','candidate_mask') else v) for k,v in inputs.items()}
        np.testing.assert_allclose(np.asarray(policy.model(shuffled)['logits']),np.asarray(original['logits'])[:,perm],atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.keras';policy.save(path)
            restored=NeuralPolicy.load(path).model
            self.assertTrue(restored.stable_heads)
            self.assertTrue(restored.standardize_inputs)
            self.assertTrue(restored.isolate_value)
            self.assertTrue(restored.fact_readout)
            for key in ('logits','value','memory_out'):
                np.testing.assert_allclose(np.asarray(restored(inputs)[key]),np.asarray(original[key]),atol=1e-6)

    def test_noisy_terminal_outcomes_preserve_expected_action_preference(self):
        import numpy as np
        from patchwork_automaton.neural.scenarios import sample,sample_rewards
        scenario=sample('mana',np.random.default_rng(7))
        options=scenario.observation.decision.options
        correct=next(i for i,o in enumerate(options) if o.id in scenario.winning_options)
        wrong=next(i for i,o in enumerate(options) if o.id not in scenario.winning_options)
        n=10000;scenarios=[scenario]*n
        for action,expected in ((correct,.8),(wrong,.2)):
            rewards=sample_rewards(scenarios,[action]*n,np.random.default_rng(99),.2)
            self.assertLess(abs(float(np.mean(rewards==1))-expected),.02)
            self.assertTrue(set(rewards)<= {-1.,1.})
        self.assertEqual(sample_rewards([scenario],[correct],np.random.default_rng(99))[0],1.)

    def test_diverse_training_retains_identical_heldout_states(self):
        import numpy as np
        from patchwork_automaton.neural.scenarios import sample,TACTICAL_TASKS
        left=np.random.default_rng(123);right=np.random.default_rng(123)
        for task in TACTICAL_TASKS:
            for _ in range(30):
                self.assertEqual(sample(task,left,heldout=True),sample(task,right,heldout=True,distribution='diverse'))
        rng=np.random.default_rng(321);life=set();powers={'direct_damage':set(),'double_strike':set()};self_lower=False
        for _ in range(200):
            s=sample('charm_mode',rng,distribution='diverse')
            me=next(p for p in s.observation.players if p.id==s.observation.actor)
            life.add(me.life)
            creature=next(o for o in s.observation.objects.values() if o.name=='Monastery Swiftspear')
            powers[s.variant].add(int(creature.properties['power']))
            b=sample('bolt_target',rng,distribution='diverse')
            own=next(p for p in b.observation.players if p.id==b.observation.actor)
            opponent=next(p for p in b.observation.players if p.id!=b.observation.actor)
            self_lower |= own.life<opponent.life
        self.assertTrue(any(n>20 for n in life))
        self.assertTrue(any(n<6 for n in life))
        self.assertFalse(life & {17,18,19,20})
        self.assertTrue(powers['direct_damage'] & powers['double_strike'])
        self.assertFalse((powers['direct_damage']|powers['double_strike']) & {2,5})
        self.assertTrue(self_lower)

    def test_identity_mask_preserves_semantics_and_source_consistency(self):
        import numpy as np
        from patchwork_automaton.neural.scenarios import sample,TASKS
        from patchwork_automaton.neural.features import Encoder,ABILITY_ID_SLICE
        from patchwork_automaton.neural.augmentation import mask_identities
        from examples.train_scenarios import batch_for
        rng=np.random.default_rng(30)
        inputs=batch_for([sample(task,rng) for task in TASKS],Encoder(),16)
        original={k:v.copy() for k,v in inputs.items()}
        masked=mask_identities(inputs,rng,1.)
        np.testing.assert_array_equal(masked['card_ids'],0)
        for key in inputs:np.testing.assert_array_equal(inputs[key],original[key])
        for key in inputs.keys()-{'card_ids','candidates'}:
            np.testing.assert_array_equal(masked[key],inputs[key])
        np.testing.assert_array_equal(masked['candidates'][...,:ABILITY_ID_SLICE.start],inputs['candidates'][...,:ABILITY_ID_SLICE.start])
        np.testing.assert_array_equal(masked['candidates'][...,ABILITY_ID_SLICE.stop:],inputs['candidates'][...,ABILITY_ID_SLICE.stop:])
        identities=masked['candidates'][...,ABILITY_ID_SLICE]
        sourced=np.any(inputs['candidate_refs'],axis=-1)&inputs['candidate_mask']
        np.testing.assert_array_equal(identities[sourced,0],1.)
        np.testing.assert_array_equal(identities[sourced,1:],0.)

    def test_isolated_value_loss_does_not_update_policy_features(self):
        import numpy as np
        import tensorflow as tf
        from patchwork_automaton.neural.scenarios import sample
        from patchwork_automaton.neural.policy import untrained_policy
        from examples.train_scenarios import batch_for
        policy=untrained_policy(seed=8,width=16,memory_size=16,blocks=1,stable_heads=True,isolate_value=True)
        inputs=batch_for([sample('mana',np.random.default_rng(8))],policy.encoder,16)
        model=policy.model
        with tf.GradientTape() as tape:
            value=model(inputs)['value']
            loss=tf.reduce_mean(tf.square(value-1.))
        gradients=tape.gradient(loss,model.trainable_variables)
        critic_ids={id(v) for v in model.value_head.trainable_variables}
        self.assertTrue(any(g is not None for g,v in zip(gradients,model.trainable_variables) if id(v) in critic_ids))
        for gradient,variable in zip(gradients,model.trainable_variables):
            if id(variable) not in critic_ids:self.assertIsNone(gradient)
