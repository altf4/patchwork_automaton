"""Endpoint schedule and durable-result validation; no models or games run."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS
from patchwork_automaton.rl.hartford_evaluation import summarize


SCRIPT = Path(__file__).resolve().parents[1] / 'docs/benchmarks/hartford-identity-opponents-ab-2026-11-03/evaluate.py'
spec = importlib.util.spec_from_file_location('opponent_ab_evaluate', SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def outcome(planned, result='win'):
    row = dict(planned, trajectories=[], actions=None)
    if result == 'draw':
        return dict(row, status='draw', winner_seat=None, draw=True, learner_won=False)
    winner = planned['learner_seat'] if result == 'win' else 1 - planned['learner_seat']
    return dict(row, status='win' if winner == 0 else 'loss', winner_seat=winner,
                draw=False, learner_won=result == 'win')


class EndpointEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(seed=432611030000)
        self.planned = evaluation.schedule(self.config, HARTFORD_DECKS, 'a' * 64)

    def test_balanced_unique_deterministic_schedule(self):
        from collections import Counter
        self.assertEqual(len(self.planned), 10000)
        self.assertEqual({r['index'] for r in self.planned}, set(range(10000)))
        self.assertEqual({r['seed'] for r in self.planned}, set(range(432611030000, 432611040000)))
        counts = Counter((r['learner_deck'], r['opponent_deck'], r['learner_seat'],
                          r['starting_player']) for r in self.planned)
        self.assertEqual(len(counts), 400)
        self.assertEqual(set(counts.values()), {25})
        self.assertEqual(self.planned, evaluation.schedule(self.config, HARTFORD_DECKS, 'a' * 64))
        self.assertTrue(all(r['deck_catalog'] == 'hartford' for r in self.planned))

    def test_extension_has_twenty_thousand_fresh_balanced_games(self):
        from collections import Counter
        extension = evaluation.schedule(dict(seed=432611040000, games=20000), HARTFORD_DECKS, 'a' * 64)
        self.assertEqual(len(extension), 20000)
        self.assertEqual({r['index'] for r in extension}, set(range(10000, 30000)))
        self.assertEqual({r['seed'] for r in extension}, set(range(432611040000, 432611060000)))
        self.assertTrue({r['seed'] for r in self.planned}.isdisjoint(r['seed'] for r in extension))
        counts = Counter((r['learner_deck'], r['opponent_deck'], r['learner_seat'], r['starting_player'])
                         for r in self.planned + extension)
        self.assertEqual(len(counts), 400)
        self.assertEqual(set(counts.values()), {75})

    def test_original_schedule_keeps_its_audited_hash(self):
        expected = '5500e9a7449bcccf9fa6180cf636d6581611f8070ab069c6eb65bd18303a23d2'
        opponent = 'dfa899c49a05c85e81722080cd783f7fc8bdbc785042cd5fac1b5ee98b652610'
        planned = evaluation.schedule(self.config, HARTFORD_DECKS, opponent)
        self.assertEqual(evaluation.value_sha(planned), expected)

    def test_resume_recovers_exact_successes_and_blocks_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'games').mkdir()
            selected = self.planned[:3]
            for planned in selected:
                row = outcome(planned)
                (root / 'games' / f"game-{row['index']}.json").write_text(json.dumps(
                    dict(row=row, sha256=evaluation.value_sha(row))))
            rows = evaluation.load_results(root, self.planned)
            self.assertEqual(set(rows), {r['index'] for r in selected})
            remaining = [r for r in self.planned if r['index'] not in rows]
            self.assertEqual(len(remaining), 9997)
            self.assertTrue(set(rows).isdisjoint(r['index'] for r in remaining))
            path = root / 'games' / f"game-{selected[0]['index']}.json"
            envelope = json.loads(path.read_text())
            envelope['row']['seed'] += 1
            path.write_text(json.dumps(envelope))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                evaluation.load_results(root, self.planned)

    def test_valid_checksum_cannot_hide_wrong_schedule_or_duplicate_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'games').mkdir()
            row = outcome(self.planned[0])
            row['opponent_checkpoint_sha256'] = 'b' * 64
            path = root / 'games' / f"game-{row['index']}.json"
            path.write_text(json.dumps(dict(row=row, sha256=evaluation.value_sha(row))))
            with self.assertRaisesRegex(ValueError, 'pinned schedule'):
                evaluation.load_results(root, self.planned)
            row = outcome(self.planned[0])
            envelope = dict(row=row, sha256=evaluation.value_sha(row))
            path.write_text(json.dumps(envelope))
            (root / 'games' / 'duplicate.json').write_text(json.dumps(envelope))
            with self.assertRaisesRegex(ValueError, 'Foreign or duplicate'):
                evaluation.load_results(root, self.planned)

    def test_preserved_failure_blocks_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'failures').mkdir()
            failure = root / 'failures/game-1.json'
            failure.write_text('{"error":"original diagnostic"}')
            before = failure.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Unresolved evaluation failures'):
                evaluation.load_results(root, self.planned)
            self.assertEqual(failure.read_bytes(), before)

    def test_adjudication_diagnostics_must_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'games').mkdir()
            row = outcome(self.planned[0], 'draw')
            diagnostic = root / 'diagnostics' / f"game-{row['index']:06d}"
            diagnostic.mkdir(parents=True)
            (diagnostic / 'adjudication.json').write_text(json.dumps(row))
            row.update(adjudicated=True, diagnostics=str(diagnostic))
            row['diagnostic_hashes'] = evaluation.diagnostic_hashes(root, row)
            (root / 'games' / f"game-{row['index']}.json").write_text(json.dumps(
                dict(row=row, sha256=evaluation.value_sha(row))))
            self.assertEqual(len(evaluation.load_results(root, self.planned)), 1)
            (diagnostic / 'adjudication.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'diagnostics changed'):
                evaluation.load_results(root, self.planned)

    def test_score_counts_draws_as_half_and_uses_focal_seat(self):
        rows = [outcome(self.planned[0], 'win'), outcome(self.planned[1], 'loss'),
                outcome(self.planned[2], 'draw')]
        result = summarize(rows, HARTFORD_DECKS)
        self.assertEqual((result['wins'], result['losses'], result['draws']), (1, 1, 1))
        self.assertEqual(result['score'], .5)
        self.assertEqual(result['win_rate'], 1 / 3)

    def test_training_audit_is_bound_to_both_completed_branch_models(self):
        launch = {'arms': {name: {'output': '/runs/' + name} for name in ('control', 'diverse')}}
        models = {name: {'sha256': digest * 64} for name, digest in [('control', 'a'), ('diverse', 'b')]}
        audited = dict(passed=True, games=100000, arms={name: dict(passed=True, games=50000,
            checkpoints=51, partitions={'training': 40000, 'validation': 10000}, evaluation_games=0,
            output=launch['arms'][name]['output'], final_policy_sha256=models[name]['sha256'])
            for name in ('control', 'diverse')})
        evaluation.validate_training_audit(audited, launch, models)
        for field, wrong in [('output', '/runs/foreign'), ('final_policy_sha256', 'c' * 64),
                             ('games', 49999), ('evaluation_games', 4000),
                             ('partitions', {'training': 50000, 'validation': 0})]:
            changed = deepcopy(audited)
            changed['arms']['diverse'][field] = wrong
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'does not match'):
                evaluation.validate_training_audit(changed, launch, models)
        changed = deepcopy(audited)
        changed['arms'].pop('control')
        with self.assertRaisesRegex(ValueError, 'Both completed'):
            evaluation.validate_training_audit(changed, launch, models)


if __name__ == '__main__':
    unittest.main()
