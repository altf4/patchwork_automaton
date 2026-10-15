"""Reviewed Burn mechanics and their actual numerical model representation."""
import json
from dataclasses import replace
from importlib.resources import files
import unittest

from patchwork_automaton.burn_semantics import CARD_REGISTRY as CARDS
from patchwork_automaton.semantics import (
    INITIAL_CARDS, AbilityKind as A, EffectKind as E, CostKind as C,
    Color, Keyword as K, QuantityKind as Q, Recipient as R,
    ObjectFilter as F, Condition as D, RuleTag as G, Coverage,
    UNKNOWN_CARD, encode_card, lookup_visible,
)
from patchwork_automaton.core import Handle, ObjectView


class BurnSemanticsTests(unittest.TestCase):
    def test_whole_deck_has_reviewed_distinct_identities_and_numeric_annotations(self):
        deck=json.loads(files('patchwork_automaton').joinpath('data/modern-burn-2026-10-04.json').read_text())
        names=[c['name'].replace('Wear/Tear','Wear // Tear') for side in ('mainboard','sideboard') for c in deck[side]]
        self.assertEqual(len(names),26)
        self.assertEqual(len({CARDS[n].id for n in names}),26)
        def numeric(value):
            if isinstance(value,tuple):
                for v in value:numeric(v)
            else:self.assertIs(type(value),int)
        for name in names:
            card=CARDS[name]
            self.assertNotEqual(card.coverage,Coverage.UNKNOWN)
            self.assertTrue(card.abilities)
            for value in encode_card(card).values():numeric(value)
        for name,card in INITIAL_CARDS.items():self.assertEqual(CARDS[name],card)

    def test_searing_blaze_preserves_conditional_damage_and_self_target_legality(self):
        a=CARDS['Searing Blaze'].abilities[0]
        self.assertEqual([e.quantity.kind for e in a.effects],[Q.LANDFALL_DAMAGE]*2)
        self.assertEqual([e.recipient for e in a.effects],[R.TARGET]*2)
        self.assertEqual(a.effects[0].object_filter,F.PLAYER|F.PLANESWALKER)
        self.assertEqual(a.effects[1].object_filter,F.CREATURE)
        self.assertIn(G.SAME_CONTROLLER_AS_FIRST_TARGET,a.effects[1].tags)
        self.assertEqual(a.costs[0].quantity.value,2)
        self.assertEqual(a.costs[0].color,Color.RED)

    def test_modes_costs_and_order_are_not_flattened(self):
        charm=CARDS['Boros Charm']
        self.assertEqual([a.mode for a in charm.abilities],[1,2,3])
        self.assertEqual(charm.abilities[0].effects[0].quantity.value,4)
        self.assertEqual(charm.abilities[1].effects[0].recipient,R.YOUR_PERMANENTS)
        self.assertEqual(charm.abilities[2].effects[0].keyword,K.DOUBLE_STRIKE)
        crack=CARDS['Skullcrack'].abilities[0]
        self.assertEqual([e.kind for e in crack.effects],[E.CANT_GAIN_LIFE,E.DAMAGE_CANT_BE_PREVENTED,E.DAMAGE])
        skewer=CARDS['Skewer the Critics'].abilities
        self.assertEqual(len(skewer),2)
        self.assertEqual(skewer[1].condition,D.OPPONENT_LOST_LIFE)
        self.assertIn(G.ALTERNATIVE_CAST_COST,skewer[1].tags)
        self.assertEqual(sum(c.quantity.value for c in skewer[0].costs),3)
        self.assertEqual(sum(c.quantity.value for c in skewer[1].costs),1)

    def test_mana_payment_damage_and_threshold_are_distinct(self):
        ring=CARDS['Barbarian Ring'].abilities
        self.assertEqual([e.kind for e in ring[0].effects],[E.ADD_MANA,E.DAMAGE])
        self.assertNotIn(C.PAY_LIFE,[c.kind for c in ring[0].costs])
        self.assertIn(G.SEVEN_CARDS_IN_YOUR_GRAVEYARD,ring[1].tags)
        canyon=CARDS['Sunbaked Canyon'].abilities
        for a in canyon[:2]:
            self.assertEqual(len(a.effects),1)
            self.assertIn(C.PAY_LIFE,[c.kind for c in a.costs])
        self.assertEqual({a.effects[0].color for a in canyon[:2]},{Color.RED,Color.WHITE})
        self.assertEqual(canyon[2].effects[0].kind,E.DRAW)
        self.assertIn(C.SACRIFICE,[c.kind for c in canyon[2].costs])

    def test_triggers_and_conditional_effects_keep_qualifiers(self):
        swift=CARDS['Monastery Swiftspear']
        self.assertEqual(swift.keywords,(K.HASTE,K.PROWESS))
        self.assertEqual(swift.abilities[1].condition,D.YOU_CAST_NONCREATURE)
        eidolon=CARDS['Eidolon of the Great Revel'].abilities[1]
        self.assertEqual(eidolon.effects[0].recipient,R.SPELL_CASTER)
        self.assertEqual(eidolon.condition_threshold,3)
        self.assertEqual(eidolon.condition,D.SPELL_MANA_VALUE_AT_MOST)
        self.assertIn(G.IF_REVEALED_CARD_IS_LAND,CARDS['Goblin Guide'].abilities[1].effects[1].tags)
        self.assertIn(G.IF_TARGET_WAS_NONBASIC,CARDS['Molten Rain'].abilities[0].effects[1].tags)
        self.assertIn(G.IF_TARGET_DIES_THIS_TURN,CARDS['Searing Blood'].abilities[0].effects[1].tags)

    def test_split_card_aliases_and_fuse_cost(self):
        self.assertIs(CARDS['Wear/Tear'],CARDS['Wear // Tear'])
        self.assertEqual(CARDS['Wear'].id,CARDS['Tear'].id)
        self.assertEqual(CARDS['Tear'].abilities[0].costs[0].color,Color.WHITE)
        fused=CARDS['Wear // Tear'].abilities[2]
        self.assertEqual(sum(c.quantity.value for c in fused.costs),3)
        self.assertIn(G.FUSE_FROM_HAND,fused.tags)
        self.assertEqual([e.object_filter for e in fused.effects],[F.ARTIFACT,F.ENCHANTMENT])

    def test_default_lookup_includes_burn_but_never_hidden_identity(self):
        obj=ObjectView(Handle('xmage','object','guide'),'Goblin Guide','hand')
        self.assertEqual(lookup_visible(obj),CARDS['Goblin Guide'])
        self.assertIs(lookup_visible(replace(obj,properties={'faceDown':True})),UNKNOWN_CARD)


from test_neural import AVAILABLE, observation

@unittest.skipUnless(AVAILABLE,'Install neural dependencies with uv')
class BurnTensorTests(unittest.TestCase):
    def test_annotations_reach_model_and_rule_tags_change_tensors(self):
        import numpy as np
        from patchwork_automaton.neural.features import Encoder, semantic_rows, SEMANTIC_WIDTH, FEATURE_VERSION
        from patchwork_automaton.neural.policy import untrained_policy
        obs=observation()
        objects={Handle('xmage','object',name):ObjectView(Handle('xmage','object',name),name,'battlefield',obs.actor)
                 for name in CARDS}
        obs=replace(obs,objects=objects)
        encoded=Encoder().encode(obs)
        self.assertEqual(Encoder().card_count,29)
        self.assertTrue(np.all(encoded.arrays['card_ids'][2:]>0))
        self.assertEqual(FEATURE_VERSION,3)
        a=CARDS['Searing Blaze'].abilities[0]
        plain=replace(CARDS['Searing Blaze'],abilities=(replace(a,effects=(a.effects[0],replace(a.effects[1],tags=()))),))
        self.assertFalse(np.array_equal(semantic_rows(CARDS['Searing Blaze']),semantic_rows(plain)))
        for name,card in CARDS.items():
            self.assertTrue(all(len(row)==SEMANTIC_WIDTH for row in semantic_rows(card)),name)
        policy=untrained_policy(seed=5,width=16,memory_size=16,blocks=1)
        result=policy.act(obs)
        self.assertTrue(np.isfinite(result.value))
        self.assertIn(result.response.option,[o.id for o in obs.decision.options])
