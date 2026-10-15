"""Numerical policy invariants; no server or network access required."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('numpy', 'tensorflow', 'keras'))
if AVAILABLE:
    import numpy as np
    import keras
    import tensorflow as tf
    from patchwork_automaton.neural.features import Encoder, collate
    from patchwork_automaton.neural.policy import NeuralPolicy, untrained_policy, digit_options, distribution_bounds
    from patchwork_automaton.neural.history import History
    from patchwork_automaton.semantics import INITIAL_CARDS
from patchwork_automaton.core import (Handle, DecisionToken, Decision, Option, Observation, ObjectView, PlayerView)
from patchwork_automaton.errors import StaleDecision


def observation(generation=1):
    me = Handle('xmage', 'player', 'me')
    opponent = Handle('xmage', 'player', 'opponent')
    game = Handle('xmage', 'game', 'game')
    land = Handle('xmage', 'object', 'land')
    spell = Handle('xmage', 'object', 'bolt')
    players = (PlayerView(me, 'Alice', 20, 1, 53), PlayerView(opponent, 'Bob', 20, 7, 53))
    objects = {land: ObjectView(land, 'Mountain', 'battlefield', me, {'tapped': False, 'cardTypes': ['LAND']}),
               spell: ObjectView(spell, 'Lightning Bolt', 'hand', me, {'manaValue': 1, 'cardTypes': ['INSTANT']})}
    decision = Decision(DecisionToken('xmage', 'session', game, me, generation), 'GAME_SELECT',
        'Display only', (Option('object:bolt', 'uuid', 'Lightning Bolt', {'legality': 'advertised'}),
                         Option('no', 'boolean', 'Pass')))
    return Observation('xmage', game, me, generation, players, objects, decision,
        native={'canPlayObjects': {'objects': {'bolt': {'basicCastAbilities': [{'id': 'ability', 'value': 'display'}]}}}},
        turn=1, phase='precombat_main', active_player=me, priority_player=me)


@unittest.skipUnless(AVAILABLE, 'Install the neural extra with uv')
class EncoderTests(unittest.TestCase):
    def test_history_retains_transient_reveals_without_merging_seats(self):
        history = History(INITIAL_CARDS)
        obs = observation()
        history.observe(obs)
        history.take(obs)
        revealed = Handle('xmage', 'object', 'revealed')
        visible = replace(obs, revision=2, objects={**obs.objects,
            revealed: ObjectView(revealed, 'Raging Goblin', 'revealed')})
        history.observe(visible)
        history.observe(replace(obs, revision=3))
        events = history.take(obs)
        self.assertEqual(events['history_cards'].tolist(), [4, 4])
        other = replace(obs, actor=obs.players[1].id)
        self.assertFalse(history.take(other)['history_mask'].any())
        history.reset()
        self.assertFalse(history.take(obs)['history_mask'].any())

    def test_text_does_not_enter_model_features(self):
        obs = observation()
        changed = replace(obs, decision=replace(obs.decision, message='Completely different prose',
            options=tuple(replace(o, label='different') for o in obs.decision.options)),
            players=tuple(replace(p, name='different') for p in obs.players))
        before, after = Encoder().encode(obs), Encoder().encode(changed)
        for key in before.arrays:
            np.testing.assert_array_equal(before.arrays[key], after.arrays[key])
            self.assertIn(before.arrays[key].dtype.kind, 'bif')

    def test_hidden_raw_card_data_never_becomes_features(self):
        obs = observation()
        handle = next(h for h in obs.objects if h.value == 'bolt')
        first = replace(obs, objects={handle: ObjectView(handle, 'Lightning Bolt', 'hand',
            obs.actor, {'faceDown': True, 'power': 99, 'color': {'red': True}, 'rules': ['private']})})
        second = replace(first, objects={handle: replace(first.objects[handle], name='Mountain',
            properties={'faceDown': True, 'power': 1, 'color': {'blue': True}})})
        for key, value in Encoder().encode(first).arrays.items():
            np.testing.assert_array_equal(value, Encoder().encode(second).arrays[key])
        self.assertEqual(Encoder().encode(first).arrays['card_ids'][-1], 0)

    def test_padding_preserves_all_candidates_and_entities(self):
        obs = observation()
        small = Encoder().encode(obs).arrays
        many = replace(obs, decision=replace(obs.decision,
            options=tuple(Option(f'choice:{i}', 'string', 'label') for i in range(513))))
        large = Encoder().encode(many).arrays
        batch = collate([small, large])
        self.assertEqual(batch['candidate_mask'].sum(axis=1).tolist(), [2, 513])
        self.assertEqual(batch['candidate_refs'].shape, (2, 513, 4))

    def test_integer_prefix_masks_cover_exactly_the_allowed_range(self):
        for maximum in (0, 7, 10, 27, 103):
            prefixes = [0]
            for remaining in reversed(range(len(str(maximum)))):
                prefixes = [p * 10 + d for p in prefixes for d in digit_options(p, remaining, maximum)]
            self.assertEqual(prefixes, list(range(maximum + 1)))

    def test_distribution_bounds_preserve_feasibility(self):
        entries = ({'min': 1, 'max': 3}, {'min': 2, 'max': 5}, {'min': 0, 'max': 2})
        prefixes = [()]
        for _ in entries:
            extended = []
            for p in prefixes:
                low, high = distribution_bounds(entries, 6, 6, p)
                extended.extend((*p, n) for n in range(low, high + 1))
            prefixes = extended
        self.assertTrue(prefixes)
        self.assertTrue(all(sum(p) == 6 for p in prefixes))


@unittest.skipUnless(AVAILABLE, 'Install the neural extra with uv')
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = untrained_policy(seed=3, width=32, memory_size=32, blocks=1)
        cls.arrays = cls.policy.encoder.encode(observation()).arrays
        cls.batch = collate([cls.arrays])
        cls.batch['memory_in'] = np.zeros((1, 32), np.float32)
        cls.batch['advance_memory'] = np.ones((1, 1), np.float32)
        cls.policy.model(cls.batch)

    def test_masks_and_padding_do_not_change_valid_scores(self):
        base = self.policy.model(self.batch)
        larger = dict(self.arrays)
        larger['candidates'] = np.concatenate([larger['candidates']] * 3)
        larger['candidate_refs'] = np.concatenate([larger['candidate_refs']] * 3)
        larger['candidate_mask'] = np.ones(6, bool)
        batch = collate([self.arrays, larger])
        batch['memory_in'] = np.zeros((2, 32), np.float32)
        batch['advance_memory'] = np.ones((2, 1), np.float32)
        output = self.policy.model(batch)
        np.testing.assert_allclose(np.asarray(output['logits'])[0, :2], np.asarray(base['logits'])[0], atol=1.e-6)
        self.assertTrue((np.asarray(output['logits'])[0, 2:] < -1.e8).all())

    def test_gradients_reach_encoder_memory_policy_and_value(self):
        with tf.GradientTape() as tape:
            output = self.policy.model(self.batch, training=True)
            loss = -tf.nn.log_softmax(output['logits'])[0, 0] + tf.square(output['value'][0, 0] - 1)
        gradients = tape.gradient(loss, self.policy.model.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))
        self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients))

    def test_internal_arguments_do_not_advance_memory(self):
        batch = dict(self.batch, memory_in=np.ones((1, 32), np.float32),
                     advance_memory=np.zeros((1, 1), np.float32))
        np.testing.assert_array_equal(np.asarray(self.policy.model(batch)['memory_out']), batch['memory_in'])

    def test_checkpoint_roundtrip_and_vocabulary_check(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.keras'
            self.policy.save(path)
            loaded = NeuralPolicy.load(path)
            np.testing.assert_allclose(np.asarray(self.policy.model(self.batch)['logits']),
                                       np.asarray(loaded.model(self.batch)['logits']), atol=1.e-6)
            with self.assertRaises(ValueError):
                NeuralPolicy.load(path, encoder=Encoder({}))

    def test_actual_policy_selects_bounded_integer_and_distribution(self):
        policy = NeuralPolicy(self.policy.model, seed=5)
        obs = observation()
        choices = ((Option('amount', 'integer', 'X', {'minimum': -7, 'maximum': 2147483647}),),
                   (Option('amounts', 'distribution', 'damage', {'minimum': 4, 'maximum': 4,
                     'entries': ({'min': 1, 'max': 3}, {'min': 0, 'max': 3})}),))
        for generation, options in enumerate(choices, 1):
            current = replace(obs, decision=replace(obs.decision,
                token=replace(obs.decision.token, generation=generation), options=options))
            record = policy.act(current)
            if generation == 1:
                self.assertLessEqual(record.response.value, 2147483647)
                self.assertGreaterEqual(record.response.value, -7)
            else:
                self.assertEqual(sum(record.response.value), 4)
            self.assertGreater(len(record.selections), 1)
            with self.assertRaises(StaleDecision): policy.act(current)
        policy.reset()
        self.assertFalse(policy.states)


if __name__ == '__main__': unittest.main()
