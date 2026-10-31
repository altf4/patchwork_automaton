"""Production Standard playback dispatch and matched native game coverage."""
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from patchwork_automaton.neural.standard_features import StandardEncoder
from patchwork_automaton.rl import standard_imitation

LIBRARY = Path('.local/direct-standard/target/release/libpatchwork_direct_standard.so')


class DispatchTests(unittest.TestCase):
    def test_direct_is_default_and_reference_mode_is_explicit(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('PATCHWORK_STANDARD_PLAYBACK', None)
            with patch('patchwork_automaton.rl.standard_direct.play', return_value={'status': 'win'}) as direct:
                self.assertEqual(standard_imitation.play_standard_game({})['playback_pipeline'], 'direct')
                direct.assert_called_once()
            os.environ['PATCHWORK_STANDARD_PLAYBACK'] = 'python'
            with patch.object(standard_imitation, '_play_standard_game_python', return_value={'status': 'win'}) as reference:
                self.assertEqual(standard_imitation.play_standard_game({})['playback_pipeline'], 'python')
                reference.assert_called_once()
            os.environ['PATCHWORK_STANDARD_PLAYBACK'] = 'unrecognized'
            with self.assertRaisesRegex(ValueError, 'must be direct or python'):
                standard_imitation.playback_mode()

    def test_rejects_mismatched_model_schema(self):
        from patchwork_automaton.neural.direct_standard import validate_model
        with self.assertRaisesRegex(ValueError, 'feature schema'):
            validate_model(SimpleNamespace(feature_spec={'entities': 154}))
        validate_model(SimpleNamespace(feature_spec=StandardEncoder.feature_spec))

    def test_missing_library_fails_instead_of_falling_back(self):
        from patchwork_automaton.neural.direct_standard import library
        try:
            library.cache_clear()
            with patch.dict(os.environ, {'DIRECT_STANDARD_LIBRARY': '/missing/direct-standard.so'}):
                with self.assertRaisesRegex(FileNotFoundError, 'build-direct-standard'):
                    library()
        finally:
            library.cache_clear()


class FixedInference:
    def predict(self, batch, *, version, identity, unbatched):
        count = len(batch['candidate_mask'])
        return dict(logits=np.zeros((1, count), np.float32),
                    value=np.zeros((1, 1), np.float32),
                    memory_out=np.zeros((1, 4), np.float32))


@unittest.skipUnless(LIBRARY.is_file(), 'Requires the guarded direct Standard build')
class MatchedPlaybackTests(unittest.TestCase):
    def test_scripted_actions_reject_nonactor_foreign_and_stale_views(self):
        import ctypes as C
        import json
        from patchwork_automaton.baselines.standard_decks import standard_deck
        from patchwork_automaton.neural.direct_standard import library, string
        lib = library()
        deck = standard_deck('mono-green-landfall')
        cards = [c.name for c in deck.cards for _ in range(c.quantity)]
        config = json.dumps(dict(decks=[cards, cards], seed=22,
                                 starting_player='P0', deck_rules='Constructed')).encode()
        sessions = [lib.ds_new(config), lib.ds_new(config)]
        views = [lib.ds_observe(sessions[0], p) for p in (0, 1)]
        try:
            count = C.c_size_t()
            self.assertTrue(lib.ds_prepare(views[0], False, C.byref(count)))
            action = string(lib, lib.ds_choice_debug_json(views[0], 0)).encode()
            self.assertFalse(lib.ds_apply_script(sessions[0], views[1], action))
            self.assertFalse(lib.ds_apply_script(sessions[1], views[0], action))
            self.assertTrue(lib.ds_apply_script(sessions[0], views[0], action), lib.ds_error())
            self.assertFalse(lib.ds_apply_script(sessions[0], views[0], action))
        finally:
            for view in views: lib.ds_view_free(view)
            for session in sessions: lib.ds_free(session)

    def assert_samples_match(self, left, right, ppo):
        if ppo:
            from patchwork_automaton.neural.rollout_records import read_trajectory
            reader = read_trajectory
        else:
            from patchwork_automaton.neural.demonstrations import read_demonstration
            reader = read_demonstration
        a = list(reader(left['path'] if ppo else left))
        b = list(reader(right['path'] if ppo else right))
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertEqual(x.action, y.action)
            self.assertEqual(set(x.arrays), set(y.arrays))
            for key in x.arrays:
                np.testing.assert_array_equal(x.arrays[key], y.arrays[key], err_msg=key)
            if ppo:
                np.testing.assert_array_equal(x.memory, y.memory)
                np.testing.assert_array_equal(x.log_probs, y.log_probs)
                self.assertEqual(x.value, y.value)
            else:
                self.assertEqual((x.action_kind, x.decision_kind), (y.action_kind, y.decision_kind))

    def check_game(self, game, *, neural=False, two_neural=False):
        model = SimpleNamespace(feature_spec=StandardEncoder.feature_spec, memory_size=4)
        options = {}
        if neural:
            options.update(model=model, batcher=FixedInference(), learner_seat=0, record_ppo=True)
        if two_neural:
            game = dict(game, opponent_kind='frozen_neural', opponent_version=7, opponent_checkpoint_sha256='a' * 64)
            options.update(opponent_model=model, opponent_batcher=FixedInference(), opponent_version=7, opponent_sha256='a' * 64)
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for mode in ('python', 'direct'):
                out = Path(tmp) / mode; out.mkdir()
                with patch.dict(os.environ, {'PATCHWORK_STANDARD_PLAYBACK': mode}):
                    rows.append(standard_imitation.play_standard_game(game, output=out, **options))
            for row in rows:
                self.assertIn(row['status'], ('win', 'loss', 'draw'), row.get('error'))
            a, b = rows
            for key in ('status', 'winner_seat', 'decisions', 'turn', 'final_life', 'actions'):
                self.assertEqual(a.get(key), b.get(key), key)
            self.assertEqual(len(a['trajectories']), len(b['trajectories']))
            self.assertGreater(len(a['trajectories']), 0)
            for x, y in zip(a['trajectories'], b['trajectories']):
                self.assertEqual((x['player'], x['samples']), (y['player'], y['samples']))
                self.assert_samples_match(x, y, neural)

    def test_scripted_games_and_demonstrations_match(self):
        for i, decks in enumerate((['mono-green-landfall', 'izzet-prowess'],
                                   ['izzet-prowess', 'bw-lifegain'],
                                   ['bw-lifegain', 'mono-green-landfall'])):
            with self.subTest(decks=decks):
                self.check_game(dict(index=i, seed=1622610310000+i, decks=decks, starting_player=f'P{i%2}'))

    def test_neural_against_scripted_matches(self):
        self.check_game(dict(index=10, seed=1622610310010,
            decks=['mono-green-landfall', 'izzet-prowess'], starting_player='P1'), neural=True)

    def test_neural_self_play_matches(self):
        self.check_game(dict(index=11, seed=1622610310011,
            decks=['bw-lifegain', 'izzet-prowess'], starting_player='P0'), neural=True, two_neural=True)

    def test_neural_self_play_does_not_export_json_views(self):
        from patchwork_automaton.neural.direct_standard import library
        model = SimpleNamespace(feature_spec=StandardEncoder.feature_spec, memory_size=4)
        game = dict(index=12, seed=1622610310012, decks=['bw-lifegain', 'izzet-prowess'],
                    starting_player='P0', opponent_kind='frozen_neural',
                    opponent_version=7, opponent_checkpoint_sha256='a' * 64)
        with patch.dict(os.environ, {'PATCHWORK_STANDARD_PLAYBACK': 'direct'}), \
                patch.object(library(), 'ds_debug_json', side_effect=AssertionError('JSON view exported')):
            row = standard_imitation.play_standard_game(game, model=model, batcher=FixedInference(),
                learner_seat=0, opponent_model=model, opponent_batcher=FixedInference(),
                opponent_version=7, opponent_sha256='a' * 64)
        self.assertIn(row['status'], ('win', 'loss', 'draw'), row.get('error'))


if __name__ == '__main__':
    unittest.main()
