"""Cheap contracts for the fixed Standard evaluation arena."""

from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from examples.evaluate_standard_arena import (arena_cells, prepare_output,
    read_rows, scheduled_games, validate_row)
from patchwork_automaton.rl.standard_arena_report import summarize_arena


LEARNER = 'mono-green-landfall'
OPPONENT = 'izzet-prowess'


def result_row(job, status, *, reason=None):
    game = job['game']
    learner_seat = game['learner_seat']
    winner = 0 if status == 'win' else 1 if status == 'loss' else None
    row = dict(game, status=status, winner_seat=winner,
               learner_won=winner == learner_seat, draw=status == 'draw',
               trajectories=[], behavior_version=0)
    if reason is not None:
        row['reason'] = reason
    return row


def report_row(index, status, *, learner_deck=LEARNER, cell_id='cell-0',
               learner_seat=0, starting_player='P0', reason=None):
    winner = 0 if status == 'win' else 1 if status == 'loss' else None
    row = dict(index=index, seed=1000 + index, cell_id=cell_id,
               learner_label='candidate', learner_deck=learner_deck,
               opponent_label='tactical', opponent_deck=OPPONENT,
               learner_seat=learner_seat, starting_player=starting_player,
               status=status, winner_seat=winner, learner_won=winner == learner_seat,
               draw=status == 'draw', trajectories=[])
    if reason is not None:
        row['reason'] = reason
    return row


class ArenaScheduleTests(unittest.TestCase):
    def test_mirror_panel_is_small_and_matrix_expands_cross_decks(self):
        decks = (LEARNER, OPPONENT)
        styles = ('aggro', 'tactical', 'checkpoint')
        mirrors = arena_cells(decks, decks, styles, 'mirrors')
        matrix = arena_cells(decks, decks, styles, 'matrix')
        self.assertEqual(len(mirrors), 2 * len(styles))
        self.assertEqual(len(matrix), 4 * len(styles))
        self.assertTrue(all(learner == opponent for learner, opponent, _ in mirrors))
        self.assertEqual(set(matrix), {(learner, opponent, style)
            for learner in decks for opponent in decks for style in styles})

    def test_four_roles_are_balanced_and_schedule_is_reproducible(self):
        kwargs = dict(cell_index=2, games_per_cell=12, seed_base=902611000000)
        first = list(scheduled_games(LEARNER, OPPONENT, 'tactical', **kwargs))
        second = list(scheduled_games(LEARNER, OPPONENT, 'tactical', **kwargs))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(len({job['index'] for job in first}), 12)
        self.assertEqual(len({job['seed'] for job in first}), 12)
        self.assertEqual({job['game']['partition'] for job in first}, {'evaluation'})
        self.assertEqual({job['game']['learner_deck'] for job in first}, {LEARNER})
        self.assertEqual({job['game']['opponent_deck'] for job in first}, {OPPONENT})
        roles = Counter((job['game']['learner_seat'],
                         job['game']['starting_player']) for job in first)
        self.assertEqual(set(roles.values()), {3})
        self.assertEqual(len(roles), 4)
        for job in first:
            game = job['game']
            self.assertEqual(game['decks'][game['learner_seat']], LEARNER)
            self.assertEqual(game['decks'][1 - game['learner_seat']], OPPONENT)

    def test_bad_cell_size_is_rejected(self):
        for size in (0, 1, 2, 3, 5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                list(scheduled_games(LEARNER, OPPONENT, 'tactical',
                    cell_index=0, games_per_cell=size, seed_base=1))

    def test_checkpoint_opponent_identity_is_required_and_stable(self):
        digest = 'a1' * 32
        kwargs = dict(cell_index=1, games_per_cell=4, seed_base=100,
                      opponent_sha256=digest)
        jobs = list(scheduled_games(LEARNER, OPPONENT, 'checkpoint', **kwargs))
        self.assertEqual({job['game']['opponent_checkpoint_sha256'] for job in jobs}, {digest})
        self.assertEqual({job['game']['opponent_version'] for job in jobs}, {0})
        self.assertEqual({job['game']['opponent_kind'] for job in jobs}, {'frozen_neural'})
        row = result_row(jobs[0], 'win')
        self.assertEqual(validate_row(row, jobs[0]), row)
        with self.assertRaises(ValueError):
            validate_row(dict(row, opponent_checkpoint_sha256='b2' * 32), jobs[0])
        with self.assertRaises(ValueError):
            list(scheduled_games(LEARNER, OPPONENT, 'checkpoint',
                cell_index=1, games_per_cell=4, seed_base=100))
        with self.assertRaises(ValueError):
            list(scheduled_games(LEARNER, OPPONENT, 'tactical', **kwargs))

    def test_row_validation_rejects_training_data_and_false_results(self):
        jobs = list(scheduled_games(LEARNER, OPPONENT, 'tactical',
            cell_index=0, games_per_cell=4, seed_base=1))
        job = jobs[0]
        row = result_row(job, 'win')
        self.assertEqual(validate_row(row, job), row)
        # Native status is P0-relative even when the focal learner is P1.
        self.assertEqual(validate_row(result_row(jobs[1], 'win'), jobs[1])['learner_won'], False)
        self.assertEqual(validate_row(result_row(jobs[1], 'loss'), jobs[1])['learner_won'], True)
        with self.assertRaises(ValueError):
            validate_row(dict(row, trajectories=[{'action': 0}]), job)
        with self.assertRaises(ValueError):
            validate_row(dict(row, learner_won=False), job)
        with self.assertRaises(ValueError):
            validate_row(dict(row, seed=999), job)


class ArenaReportTests(unittest.TestCase):
    def test_natural_completion_excludes_adjudication_and_unresolved_attempts(self):
        rows = [
            report_row(0, 'win', starting_player='P0'),
            report_row(1, 'loss', starting_player='P1'),
            report_row(2, 'draw', starting_player='P0'),
            dict(report_row(3, 'draw', starting_player='P1', reason='turn_limit'),
                 adjudicated=True),
            report_row(4, 'cutoff', starting_player='P0'),
            report_row(5, 'error', starting_player='P1'),
        ]
        summary = summarize_arena(rows)
        aggregate = summary['aggregate']
        self.assertEqual((aggregate['attempted'], aggregate['completed'],
                          aggregate['natural_completed'], aggregate['adjudicated_draws']),
                         (6, 4, 3, 1))
        self.assertEqual(aggregate['completion_rate'], .5)
        low, high = aggregate['completion_rate_wilson_95_interval']
        self.assertAlmostEqual(low, .1876163, places=6)
        self.assertAlmostEqual(high, .8123837, places=6)
        self.assertEqual((aggregate['on_play']['natural_completed'],
                          aggregate['on_play']['attempted'],
                          aggregate['on_draw']['natural_completed'],
                          aggregate['on_draw']['attempted']), (2, 3, 1, 3))
        for group in (summary['entrants'][0], summary['cells'][0]):
            self.assertEqual(group['natural_completed'], 3)
            self.assertEqual(group['completion_rate'], .5)
            self.assertEqual(group['completion_rate_wilson_95_interval'],
                             aggregate['completion_rate_wilson_95_interval'])
        self.assertEqual(aggregate['win_rate'], .25)
        self.assertEqual(aggregate['game_score'], .5)

    def test_empty_completion_rate_is_undefined(self):
        aggregate = summarize_arena([])['aggregate']
        self.assertEqual(aggregate['natural_completed'], 0)
        self.assertIsNone(aggregate['completion_rate'])
        self.assertIsNone(aggregate['completion_rate_wilson_95_interval'])

    def test_attempts_outcomes_and_play_draw_splits(self):
        rows = [
            report_row(0, 'win', learner_seat=0, starting_player='P0'),
            report_row(1, 'win', learner_seat=1, starting_player='P0'),
            report_row(2, 'draw', learner_seat=1, starting_player='P1', reason='turn_limit'),
            report_row(3, 'cutoff', learner_seat=0, starting_player='P1', reason='action_limit'),
            report_row(4, 'error', learner_seat=0, starting_player='P0', reason='engine_error'),
        ]
        summary = summarize_arena(rows)
        aggregate = summary['aggregate']
        self.assertEqual(summary['schema'], 1)
        self.assertEqual((aggregate['attempted'], aggregate['completed'], aggregate['wins'],
                          aggregate['losses'], aggregate['draws'], aggregate['cutoffs'],
                          aggregate['errors']), (5, 3, 1, 1, 1, 1, 1))
        self.assertAlmostEqual(aggregate['win_rate'], 1 / 3)
        self.assertAlmostEqual(aggregate['game_score'], .5)
        low, high = aggregate['win_rate_wilson_95_interval']
        self.assertLessEqual(low, 1 / 3)
        self.assertGreaterEqual(high, 1 / 3)
        self.assertEqual(aggregate['on_play']['attempted'], 3)
        self.assertEqual(aggregate['on_draw']['attempted'], 2)
        self.assertEqual(aggregate['draw_reasons'], {'turn_limit': 1})
        self.assertEqual(aggregate['cutoff_reasons'], {'action_limit': 1})
        self.assertEqual(aggregate['error_reasons'], {'engine_error': 1})
        self.assertEqual(len(summary['entrants']), 1)
        self.assertEqual(len(summary['cells']), 1)

    def test_groups_entrants_and_rejects_unknown_results(self):
        rows = [report_row(0, 'win'),
                report_row(1, 'loss', learner_deck='bw-lifegain', cell_id='cell-1')]
        summary = summarize_arena(rows)
        self.assertEqual(summary['aggregate']['attempted'], 2)
        self.assertEqual(len(summary['entrants']), 2)
        self.assertEqual(len(summary['cells']), 2)
        with self.assertRaises(ValueError):
            summarize_arena([report_row(2, 'abandoned')])

    def test_summary_refuses_inconsistent_native_winner(self):
        won_by_p0 = report_row(0, 'win', learner_seat=0)
        for change in (dict(winner_seat=1), dict(learner_won=False),
                       dict(draw=True)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize_arena([dict(won_by_p0, **change)])
        draw = report_row(1, 'draw')
        with self.assertRaises(ValueError):
            summarize_arena([dict(draw, winner_seat=0)])


class ArenaOutputTests(unittest.TestCase):
    def test_resume_discards_only_a_torn_final_append(self):
        jobs = list(scheduled_games(LEARNER, OPPONENT, 'tactical',
            cell_index=0, games_per_cell=4, seed_base=100))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'games.jsonl'
            first = result_row(jobs[0], 'draw')
            complete = (json.dumps(first) + '\n').encode()
            path.write_bytes(complete + b'{"index":1')
            self.assertEqual(read_rows(path, jobs), {jobs[0]['index']: first})
            self.assertEqual(path.read_bytes(), complete)
            path.write_bytes(complete + b'{bad json}\n')
            with self.assertRaises(json.JSONDecodeError):
                read_rows(path, jobs)

    def test_existing_output_needs_exact_manifest_and_explicit_resume(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / 'arena'
            manifest = {'schema': 1, 'checkpoint_sha256': 'a1' * 32}
            self.assertEqual(prepare_output(output, manifest, resume=False), output.resolve())
            self.assertTrue((output / 'manifest.json').is_file())
            with self.assertRaises(FileExistsError):
                prepare_output(output, manifest, resume=False)
            self.assertEqual(prepare_output(output, manifest, resume=True), output.resolve())
            with self.assertRaises(ValueError):
                prepare_output(output, dict(manifest, checkpoint_sha256='b2' * 32),
                               resume=True)
            with self.assertRaises(FileNotFoundError):
                prepare_output(Path(temporary) / 'missing', manifest, resume=True)
            self.assertIn('a1' * 32, (output / 'manifest.json').read_text())


if __name__ == '__main__':
    unittest.main()
