"""Independent Python/Rust equivalence, private history, and allocation lifetime."""
import gc
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from patchwork_automaton.backends.rustmage import RustMageMatch, action_options, domain_action, _plain
from patchwork_automaton.baselines.foundations import foundations_deck, FoundationsAggro
from patchwork_automaton.core import Decision
from patchwork_automaton.errors import EventOverflow, UnsupportedInteraction
from patchwork_automaton.neural.features import python_collate
from patchwork_automaton.neural.history import History
from patchwork_automaton.neural.native_pipeline import NativeObservation, NativeHistory, collate
from patchwork_automaton.neural.rustmage_features import RustMageEncoder, expand_options
from patchwork_automaton.neural.rustmage_observation import learning_observation

AVAILABLE = (Path('.local/observation-target/release/libpatchwork_observation_pipeline.so').is_file()
             and Path('.local/rustmage/target/release/librustmage.so').is_file())


@unittest.skipUnless(AVAILABLE, 'Build RustMage and scripts/build-observation-pipeline.py')
class NativePipelineTests(unittest.TestCase):
    def equal_arrays(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertEqual(left[key].dtype, right[key].dtype, key)
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)

    def test_complete_games_match_reference_features_actions_history_and_hashes(self):
        from patchwork_automaton.rl.rustmage_experiment import visible_state_fingerprint
        encoder = RustMageEncoder()
        observed, decisions, samples = set(), 0, []
        with patch.dict(os.environ, PATCHWORK_OBSERVATION_PIPELINE='rust'):
            for seed in (19, 31, 721, 722):
                history, reference = NativeHistory(), History(encoder.registry)
                with RustMageMatch(foundations_deck('red'), seed=seed) as match:
                    for _ in range(1500):
                        pairs = []
                        for viewer in ('P0', 'P1'):
                            actual = learning_observation(match, viewer)
                            self.assertIsNone(actual.native._full)
                            history.observe(actual)
                            self.assertIsNone(actual.native._full)
                            raw = actual.native.full()
                            fake = SimpleNamespace(_raw=lambda _: raw, session=match.session, handle=match.handle)
                            expected = learning_observation(fake, viewer)
                            reference.observe(expected)
                            self.assertEqual(actual.native.fingerprint(), visible_state_fingerprint(raw))
                            pairs.append((actual, expected))
                        if pairs[0][0].terminal: break
                        actual, expected = next(pair for pair in pairs if pair[0].decision is not None)
                        choices = expand_options(expected.decision)
                        self.assertEqual(actual.native.native_choices(), choices)
                        observed.add(actual.decision.kind)
                        encoded = encoder.encode(actual)
                        self.equal_arrays(encoded.arrays, encoder.encode(expected).arrays)
                        if len(choices) > 1:
                            ah, eh = history.take(actual), reference.take(expected)
                            self.equal_arrays(ah, eh)
                            encoded.arrays.update(ah)
                            if len(samples) < 10 or decisions % 71 == 0:
                                samples.append(encoded.arrays)
                        response = FoundationsAggro().choose(expected)
                        option = next(o for o in expected.decision.options if o.id == response.option)
                        raw = match._raw(actual.actor.value)
                        if option.kind == 'finite':
                            updated = match.native_game.step(raw, int(option.id.split(':')[1]))
                        else:
                            updated = match.native_game.act(raw, domain_action(option, response.value))
                        match._after_action(updated, actual.actor.value, raw['revision'])
                        decisions += 1
                    else: self.fail('Reference game failed to terminate')
        self.assertGreater(decisions, 100)
        self.assertTrue({'priority', 'attackers', 'targets', 'pay_costs'} <= observed, observed)
        self.equal_arrays(collate(samples), python_collate(samples))
        # Arrays survive the match, observations, native encoder output and history.
        snapshots = [{k: v.copy() for k, v in sample.items()} for sample in samples]
        gc.collect()
        for sample, expected in zip(samples, snapshots): self.equal_arrays(sample, expected)

    def test_domains_order_bounds_deduplication_and_concession(self):
        cases = [
            dict(actions=[dict(kind='PayCosts', payment=[1,0,0,1,0,0])],
                 mana_payment=dict(required=[0,0,0,1,0,0], available=[2,0,0,3,0,0], generic=1)),
            dict(actions=[dict(kind='AssignDamage', target={'Player':'P1'}, amount=2)],
                 damage_assignment=dict(target={'Player':'P1'}, minimum=0, maximum=4)),
            dict(actions=[], mana_lot_spend=dict(minimum=0, maximum=100, values=[0,100])),
            dict(actions=[dict(kind='Concede'), dict(kind='Pass'), dict(kind='Pass')]),
        ]
        for raw in cases:
            native = NativeObservation.from_raw(raw)
            for allow in (False, True):
                expected = expand_options(Decision(None, 'priority', '', action_options(raw)), allow_concede=allow)
                self.assertEqual(native.native_choices(allow), expected)
        for raw in [dict(actions=[], damage_assignment=dict(minimum=0, maximum=999999)),
                    dict(actions=[], x_choice=dict(intervals=[])), dict(actions=[])]:
            with self.assertRaises(UnsupportedInteraction): NativeObservation.from_raw(raw).native_choices()

    def test_stochastic_choices_match_reference_without_materializing_python_objects(self):
        encoder, rng = RustMageEncoder(), np.random.default_rng(753)
        native, reference = NativeHistory(), History(encoder.registry)
        choices_seen = set()
        with patch.dict(os.environ, PATCHWORK_OBSERVATION_PIPELINE='rust'):
            with RustMageMatch(foundations_deck('red'), seed=753) as match:
                for _ in range(1000):
                    pairs = []
                    for viewer in ('P0', 'P1'):
                        actual = learning_observation(match, viewer)
                        native.observe(actual)
                        self.assertIsNone(actual.native._full)
                        raw = actual.native.full()
                        fake = SimpleNamespace(_raw=lambda _: raw, session=match.session, handle=match.handle)
                        expected = learning_observation(fake, viewer)
                        reference.observe(expected)
                        pairs.append((actual, expected))
                    if pairs[0][0].terminal: break
                    actual, expected = next(p for p in pairs if p[0].decision is not None)
                    choices = expand_options(expected.decision)
                    self.assertEqual(actual.native.native_choices(), choices)
                    self.equal_arrays(encoder.encode(actual).arrays, encoder.encode(expected).arrays)
                    if len(choices) > 1: self.equal_arrays(native.take(actual), reference.take(expected))
                    response, action = choices[int(rng.integers(len(choices)))]
                    choices_seen.add(action['kind'])
                    raw = match._raw(actual.actor.value)
                    updated = match.native_game.act(raw, _plain(action))
                    match._after_action(updated, actual.actor.value, raw['revision'])
        self.assertTrue({'Block', 'AssignDamage', 'AnnounceCast', 'ChooseTarget'} <= choices_seen, choices_seen)

    def test_errors_and_live_buffer_lifetime(self):
        with patch.dict(os.environ, PATCHWORK_OBSERVATION_PIPELINE='rust'):
            with RustMageMatch(foundations_deck('red'), seed=39) as match:
                obs = learning_observation(match, 'P0')
                with self.assertRaises(EventOverflow): NativeHistory(limit=1).observe(obs)
                sample = RustMageEncoder().encode(obs).arrays
                expected = {k: v.copy() for k, v in sample.items()}
                retained = sample['entities'][2:, :3]
                batch = collate([sample])
                no_choices = dict(sample, candidate_mask=np.zeros_like(sample['candidate_mask']))
                with self.assertRaisesRegex(ValueError, 'no allowed candidate'): collate([no_choices])
                with self.assertRaises(ValueError): collate([])
                bad = dict(sample, entities=sample['entities'][:, :-1])
                with self.assertRaisesRegex(ValueError, 'mix feature schemas'): collate([sample, bad])
                bad = dict(sample, candidate_refs=np.zeros((1,1), np.float32))
                with self.assertRaisesRegex(ValueError, 'axes'): collate([bad])
                raw = dict(obs.native.full(), actions=[dict(kind='ChooseTarget', target={'Player': 'P9'})])
                with self.assertRaisesRegex(UnsupportedInteraction, 'Unknown player target'):
                    NativeObservation.from_raw(raw).encoded(obs.decision.token)
        del sample, obs, match
        gc.collect()
        np.testing.assert_array_equal(retained, expected['entities'][2:, :3])
        self.equal_arrays(batch, python_collate([expected]))

    def test_cache_charges_rust_buffers_even_when_only_a_small_slice_survives(self):
        from patchwork_automaton.neural.rollout_cache import retained_bytes, RolloutCache
        with patch.dict(os.environ, PATCHWORK_OBSERVATION_PIPELINE='rust'):
            with RustMageMatch(foundations_deck('red')) as match:
                sample = RustMageEncoder().encode(learning_observation(match, 'P0')).arrays
        base = sample['entity_mask']
        while isinstance(base, np.ndarray): base = base.base
        allocated = base.obj._allocation_owner.native_nbytes
        self.assertGreaterEqual(allocated, sum(v.nbytes for v in sample.values()))
        self.assertGreaterEqual(retained_bytes(sample['entity_mask'][:1]), allocated)
        self.assertGreaterEqual(retained_bytes(sample), allocated)
        self.assertLess(retained_bytes(sample), allocated * 2)  # Bundle charged once.
        cache = RolloutCache(allocated - 1)
        self.assertFalse(cache.put('slice', sample['entity_mask'][:1]))


if __name__ == '__main__': unittest.main()
