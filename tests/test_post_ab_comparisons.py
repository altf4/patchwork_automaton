"""Exact 10K panel sizes, matched seeds, seat balance and immutable result handoffs."""
from collections import Counter
import importlib.util
from pathlib import Path
import unittest

from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS, STANDARD_DECKS

SCRIPT = Path(__file__).resolve().parents[1] / 'docs/benchmarks/hartford-identity-opponents-ab-2026-11-03/post-ab-comparisons/evaluate.py'
SPEC = importlib.util.spec_from_file_location('post_ab', SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class PanelScheduleTests(unittest.TestCase):
    def test_exact_panels_balance_and_pairing(self):
        config = dict(seeds=dict(scripted=452611030000, neural=462611030000),
                      models={'reference': {'sha256': 'a' * 64}})
        ids = set()
        for number, kind in enumerate(('scripted', 'neural')):
            decks = tuple(STANDARD_DECKS if kind == 'scripted' else HARTFORD_DECKS)
            panels = []
            for model_number in range(2):
                panel = dict(kind=kind, offset=(number * 2 + model_number) * 10000)
                rows = evaluation.schedule(panel, config, decks)
                self.assertEqual(len(rows), 10000)
                self.assertEqual(len({r['seed'] for r in rows}), 10000)
                self.assertTrue(ids.isdisjoint(r['index'] for r in rows))
                ids.update(r['index'] for r in rows)
                self.assertEqual(set(Counter((r['learner_seat'], r['starting_player']) for r in rows).values()), {2500})
                cells = Counter((r['learner_deck'], r['opponent_deck']) for r in rows)
                self.assertEqual(set(cells.values()), {1108, 1112} if kind == 'scripted' else {100})
                panels.append(sorted((r['seed'], r['decks'], r['learner_seat'], r['starting_player']) for r in rows))
            self.assertEqual(panels[0], panels[1])
        self.assertEqual(len(ids), 40000)


if __name__ == '__main__':
    unittest.main()
