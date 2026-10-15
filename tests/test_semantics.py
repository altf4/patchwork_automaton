import unittest

from patchwork_automaton.core import Handle, ObjectView
from patchwork_automaton.semantics import (
    Ability, AbilityKind, CardSemantics, Color, Cost, CostKind, Coverage,
    Effect, EffectKind, INITIAL_CARDS, Keyword, ObjectFilter, Quantity,
    QuantityKind, Recipient, UNKNOWN_CARD, encode_card, lookup_visible,
)


class SemanticTests(unittest.TestCase):
    def test_bolt_and_mountain_preserve_different_costs_and_effects(self):
        bolt = encode_card(INITIAL_CARDS['Lightning Bolt'])
        mountain = encode_card(INITIAL_CARDS['Mountain'])
        self.assertEqual(bolt['effects'][0][1:6],
                         (EffectKind.DAMAGE, QuantityKind.CONSTANT, 3,
                          Recipient.TARGET, ObjectFilter.ANY_TARGET))
        self.assertEqual(bolt['costs'][0][1:5],
                         (CostKind.MANA, QuantityKind.CONSTANT, 1, Color.RED))
        self.assertEqual(mountain['costs'][0][1], CostKind.TAP_SOURCE)
        self.assertEqual(mountain['effects'][0][1], EffectKind.ADD_MANA)
        self.assertEqual(mountain['effects'][0][6], Color.RED)

    def test_hidden_identity_does_not_leak_through_registry(self):
        handle = Handle('xmage', 'object', 'example')
        for key in ('hideInfo', 'faceDown', 'face_down'):
            obj = ObjectView(handle, 'Lightning Bolt', 'hand', properties={key: True})
            self.assertIs(lookup_visible(obj), UNKNOWN_CARD)
        self.assertIs(lookup_visible(ObjectView(handle, None, 'hand')), UNKNOWN_CARD)
        self.assertIs(lookup_visible(ObjectView(handle, 'Unannotated card', 'hand')), UNKNOWN_CARD)
        self.assertEqual(lookup_visible(ObjectView(handle, 'Mountain', 'hand')).id, 1)

    def test_unknown_is_distinct_from_verified_absence(self):
        unknown = encode_card(UNKNOWN_CARD)
        known = encode_card(INITIAL_CARDS['Mountain'])
        self.assertEqual(unknown['keywords'], known['keywords'])
        self.assertEqual(unknown['keyword_coverage'], Coverage.UNKNOWN)
        self.assertEqual(known['keyword_coverage'], Coverage.COMPLETE)

    def test_grouping_modes_effect_order_and_no_freeform_features(self):
        draw = Effect(EffectKind.DRAW, Quantity(QuantityKind.CONSTANT, 2), Recipient.SELF_PLAYER)
        discard = Effect(EffectKind.DISCARD, Quantity(QuantityKind.CONSTANT, 1), Recipient.SELF_PLAYER)
        card = CardSemantics(3, Coverage.PARTIAL, [Keyword.FLYING], Coverage.PARTIAL,
            [Ability(10, AbilityKind.ACTIVATED, [draw, discard], mode=1),
             Ability(11, AbilityKind.ACTIVATED, [draw], mode=2)])
        features = encode_card(card)
        self.assertEqual([row[0] for row in features['effects']], [0, 0, 1])
        self.assertEqual([row[1] for row in features['effects']],
                         [EffectKind.DRAW, EffectKind.DISCARD, EffectKind.DRAW])
        self.assertEqual([row[-1] for row in features['abilities']], [1, 2])
        def check(value):
            if isinstance(value, tuple):
                for item in value: check(item)
            else: self.assertIs(type(value), int)
        for value in features.values(): check(value)
        self.assertIsInstance(card.abilities, tuple)
        self.assertIsInstance(card.abilities[0].effects, tuple)

    def test_generic_mana_is_not_produced_mana(self):
        with self.assertRaises(ValueError):
            Effect(EffectKind.ADD_MANA, color=Color.GENERIC)
        self.assertEqual(Cost(CostKind.MANA, color=Color.GENERIC).color, Color.GENERIC)

    def test_duplicate_ability_identity_rejected(self):
        ability = Ability(1, AbilityKind.SPELL)
        with self.assertRaises(ValueError):
            CardSemantics(1, abilities=(ability, ability))
