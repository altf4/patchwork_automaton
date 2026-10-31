"""Complete teacher labels, transferable action features and whole-game splits."""
from collections import Counter
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import time

import numpy as np

from patchwork_automaton.backends.rustmage import action_options, RustMageMatch
from patchwork_automaton.baselines.standard_decks import standard_deck
from patchwork_automaton.neural.rustmage_observation import learning_observation
from patchwork_automaton.core import Decision
from patchwork_automaton.errors import UnsupportedInteraction
from patchwork_automaton.neural.demonstrations import Demonstration, DemonstrationWriter, read_demonstration
from patchwork_automaton.neural.standard_features import StandardEncoder, expand_standard_options
from patchwork_automaton.rl.standard_imitation import (
    play_standard_game, timed_standard_game, _play_standard_game_python)
from examples.collect_standard_demonstrations import schedule
from examples.train_standard_imitation import audit_datasets, deck_games, partition_games, evaluation_summary


class StandardImitationTests(unittest.TestCase):
    def test_split_balances_every_pairing_seat_and_start_without_game_leakage(self):
        games = list(schedule(2000, 100000))
        self.assertEqual(len(games), 12000)
        self.assertEqual(len({g['seed'] for g in games}), 12000)
        self.assertEqual(Counter(g['partition'] for g in games), dict(training=9600, validation=2400))
        strata = Counter((tuple(g['pairing']), g['index'] % 4, g['partition']) for g in games)
        self.assertEqual(len(strata), 6 * 4 * 2)
        for (_, _, partition), count in strata.items():
            self.assertEqual(count, 400 if partition == 'training' else 100)
        for g in games:
            self.assertEqual(g['starting_player'], f"P{(g['index'] // 2) % 2}")
        next_stage = list(schedule(2000, 112000, index_base=12000))
        self.assertFalse({g['seed'] for g in games} & {g['seed'] for g in next_stage})

    def test_shared_training_partition_keeps_both_seats_and_all_three_decks(self):
        games = list(schedule(2000, 100000))
        for game in games:
            game['trajectories'] = [dict(player=p, deck=deck) for p, deck in enumerate(game['decks'])]
        training, validation = (partition_games(games, part) for part in ('training', 'validation'))
        self.assertEqual((len(training), len(validation)), (9600, 2400))
        self.assertEqual(Counter(t['deck'] for g in training for t in g['trajectories']),
            {'mono-green-landfall': 6400, 'izzet-prowess': 6400, 'bw-lifegain': 6400})
        self.assertTrue(all(len(g['trajectories']) == 2 for g in training + validation))
        self.assertFalse({g['seed'] for g in training} & {g['seed'] for g in validation})

    def test_controller_uses_one_checkpoint_stream_across_decks_and_stages(self):
        from examples.run_standard_imitation import run, parser
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            args = parser().parse_args(['--output', str(output), '--games-per-stage', '120',
                '--stages', '2', '--dataset-root', str(Path(directory) / 'existing-data')])
            with patch('examples.run_standard_imitation.source_hashes', return_value={}), patch(
                    'examples.run_standard_imitation.subprocess.run') as commands:
                run(args)
            training = [c.args[0] for c in commands.call_args_list if 'examples.train_standard_imitation' in c.args[0]]
            self.assertEqual(len(training), 2)
            self.assertEqual({c[c.index('--output') + 1] for c in training}, {str(output / 'agent')})
            self.assertTrue(all('--deck' not in c for c in training))
            self.assertTrue(all(c[c.index('--learner-device') + 1] == 'gpu' for c in training))
            collection = [c for c in commands.call_args_list if 'examples.collect_standard_demonstrations' in c.args[0]]
            self.assertTrue(all(c.kwargs['env']['CUDA_VISIBLE_DEVICES'] == '-1' for c in collection))
            self.assertTrue(all(c.args[0][c.args[0].index('--runners') + 1] == '2' for c in collection))
            self.assertEqual([len(c[c.index('--datasets') + 1:c.index('--output')]) for c in training], [1,2])
            report = json.loads((output / 'experiment.json').read_text())
            self.assertEqual(report['policy_scope'], 'shared')
            self.assertTrue(all('agent' in point and 'agents' not in point for point in report['stages']))

    def test_multicolor_payments_enumerate_feasible_choices_without_cartesian_guard(self):
        decision = Decision(None, 'pay_costs', '', action_options(dict(actions=[],
            mana_payment=dict(required=[0] * 6, available=[100] * 6, generic=1))))
        choices = expand_standard_options(decision)
        self.assertEqual(len(choices), 6)
        self.assertTrue(all(sum(action['payment']) == 1 for _, action in choices))
        huge = Decision(None, 'x', '', action_options(dict(actions=[],
            x_choice=dict(intervals=[dict(minimum=0, maximum=100000)]))))
        with self.assertRaises(UnsupportedInteraction):
            expand_standard_options(huge)

    def test_color_choice_production_and_library_destination_have_distinct_features(self):
        encoder = StandardEncoder()
        self.assertNotEqual(encoder.action_extras(dict(kind='ChooseEntryValue', value={'BasicLandType': 'Forest'})),
                            encoder.action_extras(dict(kind='ChooseEntryValue', value={'BasicLandType': 'Swamp'})))
        self.assertNotEqual(encoder.action_extras(dict(kind='ChooseMana', production=[1, 0, 0, 0, 0, 0])),
                            encoder.action_extras(dict(kind='ChooseMana', production=[0, 1, 0, 0, 0, 0])))
        self.assertNotEqual(encoder.action_extras(dict(kind='ArrangeLibrary', destination='Top')),
                            encoder.action_extras(dict(kind='ArrangeLibrary', destination='Bottom')))
        self.assertNotEqual(encoder.entity_row({'entry_choice': {'BasicLandType': 'Forest'}}, 0),
                            encoder.entity_row({'entry_choice': {'BasicLandType': 'Swamp'}}, 0))
        self.assertEqual(encoder.action_roles(dict(kind='ChooseExplore', object=dict(card=13, generation=2), graveyard=True))['card'], 13)
        self.assertNotEqual(encoder.action_extras(dict(kind='BeginFaceUp', cost=dict(colored=[0,0,1,0,0,0], generic=1))),
                            encoder.action_extras(dict(kind='BeginFaceUp', cost=dict(colored=[0,0,0,1,0,0], generic=1))))

    def test_unresolved_evaluation_does_not_disappear_from_win_rate(self):
        result = evaluation_summary([dict(status='win', learner_won=True),
            dict(status='loss', learner_won=False), dict(status='cutoff'), dict(status='error')])
        self.assertEqual(result['attempted'], 4)
        self.assertEqual(result['win_rate'], .25)
        self.assertEqual((result['cutoffs'], result['errors']), (1, 1))

    @unittest.skipUnless(Path('.local/rustmage/target/release/librustmage.so').exists(), 'Requires RustMage')
    def test_specific_optional_search_alias_uses_existing_optional_choice_columns(self):
        encoder = StandardEncoder()
        with RustMageMatch(standard_deck('mono-green-landfall'), seed=100) as match:
            obs = next(v for p in (0,1) if (v := learning_observation(match, f'P{p}')).decision is not None)
            choices = encoder.expand_options(obs.decision)
            for alias, decision in encoder.decision_aliases.items():
                left = encoder.encode(obs._replace(native=dict(obs.native, decision=alias)), choices=choices).arrays
                right = encoder.encode(obs._replace(native=dict(obs.native, decision=decision)), choices=choices).arrays
                for key in left:
                    np.testing.assert_array_equal(left[key], right[key], err_msg=key)

    def test_turn_50_can_finish_but_turn_51_is_adjudicated_draw(self):
        game = next(schedule(20, 1000))
        for turn in (50, 51):
            observations = [SimpleNamespace(turn=turn, players=[SimpleNamespace(life=20)] * 2,
                terminal=True, outcome=SimpleNamespace(won=p == 0, draw=False)) for p in range(2)]
            with patch('patchwork_automaton.rl.standard_imitation.RustMageMatch'), patch(
                    'patchwork_automaton.rl.standard_imitation.learning_observation', side_effect=observations):
                row = _play_standard_game_python(game)
            self.assertEqual(row['status'], 'win' if turn == 50 else 'draw')
            if turn == 51:
                self.assertTrue(row['adjudicated'])
                self.assertEqual(row['reason'], 'turn_limit')
                self.assertIsNone(row['winner_seat'])

    def test_frozen_opponent_uses_separate_neural_policy_and_records_only_learner(self):
        game=dict(index=77,seed=123456,decks=['mono-green-landfall','izzet-prowess'],
            starting_player='P1',opponent_kind='frozen_neural',
            opponent_checkpoint_sha256='a'*64,opponent_version=200)
        players=[SimpleNamespace(life=20),SimpleNamespace(life=20)]
        option=SimpleNamespace(id='option:0',kind='finite',constraints={'action':{'kind':'Pass'}})
        prompt=SimpleNamespace(options=[option])
        first=[SimpleNamespace(turn=1,players=players,terminal=False,decision=None),
               SimpleNamespace(turn=1,players=players,terminal=False,decision=prompt,
                   actor=SimpleNamespace(value='P1'))]
        last=[SimpleNamespace(turn=1,players=players,terminal=True,
                  outcome=SimpleNamespace(won=False,draw=False)),
              SimpleNamespace(turn=1,players=players,terminal=True,
                  outcome=SimpleNamespace(won=True,draw=False))]
        match=MagicMock()
        match.__enter__.return_value=match
        match._raw.return_value={'revision':0}
        learner_policy,opponent_policy=MagicMock(),MagicMock()
        opponent_policy.act.return_value.response=SimpleNamespace(option='option:0')
        with tempfile.TemporaryDirectory() as directory:
            writer=SimpleNamespace(path=Path(directory)/'learner.pkl.gz',count=1,
                value_sum=0.,value_square_sum=0.,close=lambda:None)
            with patch('patchwork_automaton.rl.standard_imitation.RustMageMatch',return_value=match), patch(
                    'patchwork_automaton.rl.standard_imitation.learning_observation',side_effect=first+last), patch(
                    'patchwork_automaton.neural.standard_inference.StandardInferencePolicy',
                    side_effect=[learner_policy,opponent_policy]) as make_policy, patch(
                    'patchwork_automaton.neural.rollout_records.TrajectoryWriter',return_value=writer) as make_writer:
                row=_play_standard_game_python(game,output=directory,model=object(),batcher=object(),
                    learner_seat=0,record_ppo=True,version=200,
                    opponent_model=object(),opponent_batcher=object(),
                    opponent_version=200,opponent_sha256='a'*64)
        self.assertEqual(row['status'],'loss')
        self.assertEqual(len(row['trajectories']),1)
        self.assertEqual(row['trajectories'][0]['player'],0)
        self.assertEqual(make_writer.call_count,1)
        self.assertEqual(make_policy.call_count,2)
        self.assertIs(make_policy.call_args_list[0].kwargs['sink'],writer)
        self.assertIsNone(make_policy.call_args_list[1].kwargs['sink'])
        opponent_policy.act.assert_called_once()
        opponent_policy.choose.assert_not_called()

    def test_scripted_opponent_factory_is_opt_in(self):
        game=dict(index=78,seed=123457,decks=['mono-green-landfall','izzet-prowess'],
            starting_player='P0')
        views=[SimpleNamespace(turn=1,players=[SimpleNamespace(life=20)]*2,
            terminal=True,outcome=SimpleNamespace(won=p==0,draw=False)) for p in (0,1)]
        factory=MagicMock()
        with patch('patchwork_automaton.rl.standard_imitation.RustMageMatch'), patch(
                'patchwork_automaton.rl.standard_imitation.learning_observation',side_effect=views):
            row=_play_standard_game_python(game,opponent_policy_factory=factory)
        self.assertEqual(row['status'],'win')
        self.assertEqual([call.args[0] for call in factory.call_args_list],game['decks'])

    def test_hard_wall_clock_kills_blocked_game_and_logs_draw(self):
        import sys
        if 'tensorflow' in sys.modules:
            # Other suites may have initialized TF before loading this module.
            # Match production: isolate the supervisor in a TF-free process.
            import subprocess
            subprocess.run([sys.executable, '-m', 'unittest', self.id()],
                check=True, capture_output=True, timeout=15)
            return
        # A blocked native call cannot cooperate with checks inside the game loop.
        with tempfile.TemporaryDirectory() as directory:
            game = next(schedule(20, 1000))
            started = time.monotonic()
            def blocked(game, **options):
                for player in (0, 1):
                    writer = DemonstrationWriter(Path(options['output']) / f"game-{game['index']:06d}-p{player}.pkl.gz")
                    writer.write(Demonstration(dict(candidate_mask=np.ones(2, bool)), 0, 'Keep', 'mulligan'))
                    writer.file.flush()
                time.sleep(30)
            with patch('patchwork_automaton.rl.standard_imitation.play_standard_game', side_effect=blocked):
                row = timed_standard_game(game, diagnostics=directory, output=Path(directory) / 'trajectories', game_timeout=.15)
            self.assertLess(time.monotonic() - started, 3)
            self.assertEqual((row['status'], row['reason']), ('draw', 'wall_clock_timeout'))
            self.assertTrue(row['adjudicated'])
            self.assertEqual(row['seed'], game['seed'])
            self.assertEqual(len(row['trajectories']), 2)
            self.assertTrue(all(len(read_demonstration(t)) == 1 and t['partial'] for t in row['trajectories']))
            self.assertTrue((Path(row['diagnostics']) / 'adjudication.json').exists())
            self.assertIn('_isolated_game', (Path(row['diagnostics']) / 'python-stack.txt').read_text())

    def test_draw_counts_in_evaluation_denominator_and_half_point_score(self):
        result = evaluation_summary([dict(status='win', learner_won=True),
            dict(status='draw', adjudicated=True), dict(status='loss', learner_won=False)])
        self.assertEqual(result['win_rate'], 1/3)
        self.assertEqual(result['score_rate'], .5)
        self.assertEqual(result['adjudicated_draws'], 1)

    @unittest.skipUnless(Path('.local/rustmage/target/release/librustmage.so').exists(), 'Requires RustMage')
    def test_manifested_creature_face_up_menu_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            row = play_standard_game(dict(index=2802, pairing=['mono-green-landfall','izzet-prowess'],
                decks=['mono-green-landfall','izzet-prowess'], seed=302610272802,
                starting_player='P1', partition='training'), output=directory)
            self.assertIn(row['status'], ('win', 'loss', 'draw'), row.get('error'))
            column = StandardEncoder.actions.index('BeginFaceUp')
            self.assertTrue(any(record.arrays['candidates'][:, column].any()
                for trajectory in row['trajectories'] for record in read_demonstration(trajectory)))

    @unittest.skipUnless(Path('.local/rustmage/target/release/librustmage.so').exists(), 'Requires RustMage')
    def test_each_pairing_records_complete_games_and_all_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            for game in list(schedule(20, 402610270000))[::20]:
                row = play_standard_game(game, output=directory)
                self.assertIn(row['status'], ('win', 'loss', 'draw'), row.get('error'))
                self.assertEqual(len(row['trajectories']), 2)
                for trajectory in row['trajectories']:
                    self.assertNotIn('reward', trajectory)
                    records = read_demonstration(trajectory)
                    self.assertGreater(len(records), 0)
                    for record in records:
                        self.assertEqual(record.arrays['candidates'].shape[-1], StandardEncoder.feature_spec['candidates'])
                        self.assertTrue(np.isfinite(record.arrays['candidates']).all())


if __name__ == '__main__':
    unittest.main()
