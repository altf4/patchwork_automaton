"""An explicit ten-deck catalog leaves previous experiments reproducible."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from patchwork_automaton.baselines.standard_decks import DECK_CATALOGS, HARTFORD_DECKS, STANDARD_DECKS, standard_deck
from patchwork_automaton.rl import standard_imitation


class HartfordDeckTests(unittest.TestCase):
    def test_catalogs_preserve_previous_three_and_exact_ten(self):
        self.assertEqual(tuple(STANDARD_DECKS), ('mono-green-landfall', 'izzet-prowess', 'bw-lifegain'))
        self.assertEqual(len(HARTFORD_DECKS), 10)
        self.assertEqual(set(DECK_CATALOGS), {'standard', 'hartford'})
        for deck_id in HARTFORD_DECKS:
            deck = standard_deck(deck_id, catalog='hartford')
            self.assertEqual(sum(c.quantity for c in deck.cards), 60)
            self.assertFalse(any(c.sideboard for c in deck.cards))
            full = standard_deck(deck_id, include_sideboard=True, catalog='hartford')
            self.assertEqual(sum(c.quantity for c in full.cards if c.sideboard), 15)
        with self.assertRaises(ValueError):
            standard_deck('dimir-midrange')
        with self.assertRaises(ValueError):
            standard_deck('dimir-midrange', catalog='typo')
        with self.assertRaises(TypeError):
            HARTFORD_DECKS['dimir-midrange']['mainboard'][0]['quantity'] = 1

    def test_python_worker_uses_game_catalog_for_both_decks(self):
        seen = []
        class Match:
            def __init__(self, *decks, **options):
                seen.extend(decks)
            def __enter__(self):
                return self
            def __exit__(self, *error):
                pass
        def observation(match, seat):
            return SimpleNamespace(turn=1, terminal=True, players=[SimpleNamespace(life=20)] * 2,
                                   outcome=SimpleNamespace(won=seat == 'P0', draw=False))
        game = dict(index=0, seed=10, decks=['dimir-midrange', 'bant-airbending'],
                    starting_player='P0', deck_catalog='hartford')
        with patch.object(standard_imitation, 'RustMageMatch', Match), \
             patch.object(standard_imitation, 'learning_observation', observation):
            row = standard_imitation._play_standard_game_python(game,
                opponent_policy_factory=lambda deck: SimpleNamespace())
        self.assertEqual(row['status'], 'win', row)
        self.assertEqual(seen, [standard_deck(d, catalog='hartford') for d in game['decks']])


if __name__ == '__main__':
    unittest.main()
