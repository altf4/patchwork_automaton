"""Synthetic duel routing and lifecycle checks; no engine games or GPU work."""
import asyncio
from contextlib import nullcontext
from copy import deepcopy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

from test_neural import AVAILABLE

if AVAILABLE:
    from examples.evaluate_rustmage_head_to_head import config_for, summarize_duels
    from patchwork_automaton.rl import rustmage_duels, rustmage_experiment
    from patchwork_automaton.neural import rustmage_policy


def synthetic_rows(statuses, seed=470261026):
    rows = []
    for index, status in enumerate(statuses):
        seat = index % 2
        row = dict(index=index, seed=seed+index, seat=seat,
            starting_player=f'P{(index//2)%2}', mode='versus_neural',
            trajectories=[], status=status)
        if status in ('win', 'loss', 'draw'):
            row['winner'] = None if status == 'draw' else seat if status == 'win' else 1-seat
        rows.append(row)
    return rows


@unittest.skipUnless(AVAILABLE, 'Install the neural extra with uv')
class DuelRoutingTests(unittest.TestCase):
    def test_both_seats_use_their_frozen_model_batcher_version_and_seed(self):
        focal, opponent, focal_batcher, opponent_batcher = (object() for _ in range(4))
        calls = []
        def policy(model, **kwargs):
            calls.append((model, kwargs))
            return SimpleNamespace()
        # Policy construction is what this fixture examines. The synthetic
        # match boundary always raises before a real RustMage engine exists.
        forbidden_engine = Mock(side_effect=RuntimeError('synthetic engine boundary'))
        with patch.object(rustmage_policy, 'RustMagePolicy', policy), \
             patch.object(rustmage_experiment, 'RustMageMatch', forbidden_engine), \
             patch.object(rustmage_experiment, 'foundations_deck', return_value='synthetic deck'):
            for index in range(4):
                calls.clear()
                row = rustmage_experiment.play_game(seed=470261026+index,
                    index=index, mode='versus_neural', model=focal, batcher=focal_batcher,
                    version=17, opponent_model=opponent, opponent_batcher=opponent_batcher,
                    opponent_version=29, greedy=False)
                self.assertEqual(row['status'], 'error')
                self.assertIn('synthetic engine boundary', row['error'])
                self.assertEqual(row['trajectories'], [])
                self.assertEqual(row['seat'], index % 2)
                self.assertEqual(row['starting_player'], f'P{(index//2)%2}')
                self.assertEqual(len(calls), 2)
                for player, (model, kwargs) in enumerate(calls):
                    is_focal = player == index % 2
                    self.assertIs(model, focal if is_focal else opponent)
                    self.assertIs(kwargs['batcher'], focal_batcher if is_focal else opponent_batcher)
                    self.assertEqual(kwargs['version'], 17 if is_focal else 29)
                    self.assertEqual(kwargs['seed'], (470261026+index)*2+player)
                    self.assertIsNone(kwargs['sink'])
                    self.assertFalse(kwargs['greedy'])
        self.assertEqual(forbidden_engine.call_count, 4)

    def test_selfplay_preserves_the_original_model_batcher_version_for_both_players(self):
        model, batcher = object(), object()
        calls = []
        def policy(selected, **kwargs):
            calls.append((selected, kwargs));return SimpleNamespace()
        forbidden_engine = Mock(side_effect=RuntimeError('synthetic engine boundary'))
        with patch.object(rustmage_policy, 'RustMagePolicy', policy), \
             patch.object(rustmage_experiment, 'RustMageMatch', forbidden_engine), \
             patch.object(rustmage_experiment, 'foundations_deck', return_value='synthetic deck'):
            for index in range(4):
                calls.clear()
                row = rustmage_experiment.play_game(seed=100+index, index=index,
                    mode='selfplay', model=model, batcher=batcher, version=7)
                self.assertEqual(row['status'], 'error')
                self.assertIn('synthetic engine boundary', row['error'])
                self.assertEqual(len(calls), 2)
                for player, (selected, kwargs) in enumerate(calls):
                    self.assertIs(selected, model)
                    self.assertIs(kwargs['batcher'], batcher)
                    self.assertEqual(kwargs['version'], 7)
                    self.assertEqual(kwargs['seed'], (100+index)*2+player)
                    self.assertIsNone(kwargs['sink'])
        self.assertEqual(forbidden_engine.call_count, 4)

    def test_active_baseline_preserves_one_focal_neural_policy_and_one_expert(self):
        model, batcher = object(), object()
        calls = []
        def policy(selected, **kwargs):
            calls.append((selected, kwargs));return SimpleNamespace()
        forbidden_engine = Mock(side_effect=RuntimeError('synthetic engine boundary'))
        with patch.object(rustmage_policy, 'RustMagePolicy', policy), \
             patch.object(rustmage_experiment, 'RustMageMatch', forbidden_engine), \
             patch.object(rustmage_experiment, 'FoundationsAggro') as expert, \
             patch.object(rustmage_experiment, 'foundations_deck', return_value='synthetic deck'):
            for index in range(4):
                calls.clear();expert.reset_mock()
                row = rustmage_experiment.play_game(seed=200+index, index=index,
                    mode='versus_baseline', model=model, batcher=batcher, version=11)
                self.assertEqual(row['status'], 'error')
                self.assertIn('synthetic engine boundary', row['error'])
                expert.assert_called_once_with()
                self.assertEqual(len(calls), 1)
                selected, kwargs = calls[0]
                self.assertIs(selected, model)
                self.assertIs(kwargs['batcher'], batcher)
                self.assertEqual(kwargs['version'], 11)
                self.assertEqual(kwargs['seed'], (200+index)*2+index%2)
                self.assertIsNone(kwargs['sink'])
        self.assertEqual(forbidden_engine.call_count, 4)

    def test_recording_and_missing_opponent_fail_before_engine_or_writer(self):
        with patch.object(rustmage_experiment, 'RustMageMatch') as engine, \
             patch.object(rustmage_policy, 'RustMagePolicy') as policy:
            for changes in (dict(opponent_model=None), dict(opponent_batcher=None),
                            dict(output='forbidden-trajectory-path')):
                options = dict(seed=1, index=0, mode='versus_neural', model=object(),
                    batcher=object(), opponent_model=object(), opponent_batcher=object())
                options.update(changes)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    rustmage_experiment.play_game(**options)
            engine.assert_not_called()
            policy.assert_not_called()


@unittest.skipUnless(AVAILABLE, 'Install the neural extra with uv')
class DuelSummaryTests(unittest.TestCase):
    def test_config_is_balanced_frozen_sampled_neural_evaluation(self):
        config = config_for(games=8, seed=470261026)
        self.assertNotIn('mode', config['collection'])
        self.assertNotIn('output', config['collection'])
        self.assertFalse(config['collection']['greedy'])
        self.assertEqual(config['mode'], 'versus_neural')
        self.assertEqual(config['cohort']['attempted_games'], 8)
        self.assertFalse(config['cohort']['replace_failed_attempts'])
        self.assertFalse(config['cohort']['training_trajectories'])
        for games in (0, 1, True):
            with self.subTest(games=games), self.assertRaises(ValueError):
                config_for(games=games, seed=470261026)

    def test_all_attempts_including_errors_and_cutoffs_remain_in_denominator(self):
        rows = synthetic_rows(['win', 'loss', 'draw', 'cutoff', 'error', 'win', 'loss', 'draw'])
        result = summarize_duels(rows, config_for(games=8, seed=470261026))
        self.assertEqual(result['attempted_games'], 8)
        self.assertEqual(result['terminal_games'], 6)
        self.assertEqual(result['wins'], 2)
        self.assertEqual(result['losses'], 2)
        self.assertEqual(result['draws'], 2)
        self.assertEqual(result['cutoffs'], 1)
        self.assertEqual(result['errors'], 1)
        self.assertEqual(result['unresolved_attempts'], 2)
        self.assertEqual(result['win_rate_all_attempts'], .25)
        self.assertTrue(result['seat_start_balanced'])
        self.assertEqual({r['attempted_games'] for r in result['seat_start_strata'].values()}, {2})
        for row in rows:
            if row['status'] in ('cutoff', 'error'):
                self.assertNotIn('winner', row)
                self.assertNotIn('terminal_rewards', row)

    def test_missing_reordered_wrong_schedule_recording_and_outcomes_are_rejected(self):
        rows = synthetic_rows(['win', 'loss', 'draw', 'cutoff'])
        config = config_for(games=4, seed=470261026)
        invalid = [rows[:-1], rows[::-1]]
        for key, value in (('index', 99), ('seed', 88), ('seat', 1),
                           ('starting_player', 'P1'), ('mode', 'versus_baseline'),
                           ('trajectories', [{'path':'forbidden'}]), ('status', 'unknown'),
                           ('winner', 1)):
            broken = deepcopy(rows);broken[0][key] = value;invalid.append(broken)
        wrong_draw = deepcopy(rows);wrong_draw[2]['winner'] = 0;invalid.append(wrong_draw)
        wrong_loss = deepcopy(rows);wrong_loss[1]['winner'] = 1;invalid.append(wrong_loss)
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                summarize_duels(values, config)


class FakeConnection:
    def __init__(self, name):
        self.name = name
        self.closed = False
    def close(self):
        self.closed = True


class FakeContext:
    def __init__(self):
        self.pairs = []
    def Pipe(self):
        index = len(self.pairs)
        pair = (FakeConnection(f'parent-{index}'), FakeConnection(f'worker-{index}'))
        self.pairs.append(pair)
        return pair
    def Value(self, kind, value):
        return SimpleNamespace(value=value, get_lock=nullcontext)


class FakeInferenceGroup:
    def __init__(self, connections, **kwargs):
        self.connections = connections
        self.kwargs = kwargs
        self.start = AsyncMock()
        self.close = AsyncMock()
        self.refresh_stats = AsyncMock()
        self.stats = Mock(return_value={'synthetic': True})


@unittest.skipUnless(AVAILABLE, 'Install the neural extra with uv')
class DuelLifecycleTests(unittest.TestCase):
    def make_collector(self, workers=3):
        context = FakeContext()
        pool = Mock()
        groups = []
        def group(*args, **kwargs):
            value = FakeInferenceGroup(*args, **kwargs);groups.append(value);return value
        models = tuple(SimpleNamespace(card_count=10+i, memory_size=2+i,
                        feature_spec={'synthetic': i}) for i in range(2))
        with patch.object(rustmage_duels.mp, 'get_context', return_value=context), \
             patch.object(rustmage_duels, 'ProcessPoolExecutor', return_value=pool), \
             patch.object(rustmage_duels, 'InferenceProcessGroup', side_effect=group):
            collector = rustmage_duels.DuelProcessCollector(*models, workers=workers,
                                                           replicas_per_policy=2)
        return collector, pool, groups, context, models

    def test_two_independent_cpu_groups_receive_only_their_own_policy(self):
        async def check():
            collector, pool, groups, context, models = self.make_collector()
            try:
                await collector.start()
                self.assertTrue(collector.started)
                for i, group in enumerate(groups):
                    group.start.assert_awaited_once_with(models[i], version=0)
                    self.assertEqual(group.kwargs, dict(replicas=2, wait_ms=0, device='cpu'))
                self.assertTrue(set(map(id, groups[0].connections)).isdisjoint(
                                set(map(id, groups[1].connections))))
                self.assertEqual(len(context.pairs), 6)
            finally:
                await collector.close()
            pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
            for group in groups:group.close.assert_awaited_once()
            self.assertTrue(all(c.closed for pair in context.pairs for c in pair))
            await collector.close()
            pool.shutdown.assert_called_once()
        asyncio.run(check())

    def test_failure_starting_second_group_closes_both_and_all_pipes(self):
        async def check():
            collector, pool, groups, context, _ = self.make_collector()
            groups[1].start.side_effect = RuntimeError('synthetic replica start failure')
            with self.assertRaisesRegex(RuntimeError, 'synthetic replica start failure'):
                await collector.start()
            self.assertTrue(collector.broken)
            self.assertTrue(collector.closed)
            for group in groups:group.close.assert_awaited_once()
            self.assertTrue(all(c.closed for pair in context.pairs for c in pair))
            pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        asyncio.run(check())

    def test_failure_closing_first_group_still_closes_other_group_and_pipes(self):
        async def check():
            collector, pool, groups, context, _ = self.make_collector()
            groups[0].close.side_effect = RuntimeError('synthetic close failure')
            with self.assertRaisesRegex(RuntimeError, 'synthetic close failure'):
                await collector.close()
            for group in groups:group.close.assert_awaited_once()
            self.assertTrue(all(c.closed for pair in context.pairs for c in pair))
            pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        asyncio.run(check())

    def test_pool_shutdown_failure_still_closes_both_groups_and_pipes(self):
        async def check():
            collector, pool, groups, context, _ = self.make_collector()
            pool.shutdown.side_effect = RuntimeError('synthetic pool shutdown failure')
            with self.assertRaisesRegex(RuntimeError, 'synthetic pool shutdown failure'):
                await collector.close()
            for group in groups:group.close.assert_awaited_once()
            self.assertTrue(all(c.closed for pair in context.pairs for c in pair))
        asyncio.run(check())

    def test_mocked_collection_preserves_every_scheduled_attempt_and_refreshes_both(self):
        async def check():
            collector, _, groups, _, _ = self.make_collector()
            await collector.start()
            submitted, progress = [], []
            statuses = ('win', 'error', 'cutoff', 'loss')
            async def submit(pool, function, options):
                submitted.append((pool, function, options))
                index = options['index']
                return synthetic_rows(statuses)[index]
            loop = SimpleNamespace(run_in_executor=submit)
            try:
                with patch.object(rustmage_duels.asyncio, 'get_running_loop', return_value=loop):
                    rows = await collector.run_games(games=4, seed_base=470261026,
                        index_base=0, max_decisions=12, max_own_turns=3,
                        repeat_limit=2, greedy=False, progress=progress.append)
                self.assertEqual([row['status'] for row in rows], list(statuses))
                self.assertEqual(rows, progress)
                for index, (pool, function, options) in enumerate(submitted):
                    self.assertIs(pool, collector.pool)
                    self.assertIs(function, rustmage_duels._run_duel_game)
                    self.assertEqual(options, dict(seed=470261026+index, index=index,
                        max_decisions=12, max_own_turns=3, repeat_limit=2, greedy=False))
                for group in groups:group.refresh_stats.assert_awaited_once()
                self.assertFalse(collector.active)
                self.assertFalse(collector.broken)
                with self.assertRaisesRegex(ValueError, 'Only evaluation limits'):
                    await collector.run_games(games=4, seed_base=470261026,
                                               output='forbidden')
            finally:
                await collector.close()
        asyncio.run(check())

    def test_mocked_worker_exception_marks_collector_broken_after_settling_submissions(self):
        async def check():
            collector, _, groups, _, _ = self.make_collector()
            await collector.start()
            settled = []
            async def submit(pool, function, options):
                settled.append(options['index'])
                if options['index'] == 1:
                    raise RuntimeError('synthetic worker failure')
                return synthetic_rows(('win', 'loss', 'draw', 'cutoff'))[options['index']]
            loop = SimpleNamespace(run_in_executor=submit)
            try:
                with patch.object(rustmage_duels.asyncio, 'get_running_loop', return_value=loop):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic worker failure'):
                        await collector.run_games(games=4, seed_base=470261026)
                self.assertEqual(settled, [0, 1, 2, 3])
                self.assertTrue(collector.broken)
                self.assertFalse(collector.active)
                for group in groups:group.refresh_stats.assert_not_awaited()
                with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                    await collector.run_games(games=4, seed_base=470261026)
            finally:
                await collector.close()
            for group in groups:group.close.assert_awaited_once()
        asyncio.run(check())

    def test_worker_keeps_two_private_distinct_pipes_and_closes_other_workers(self):
        roles = [[FakeConnection(f'{r}-{i}') for i in range(3)] for r in range(2)]
        counter = SimpleNamespace(value=1, get_lock=nullcontext)
        metadata = [dict(card_count=10+r, memory_size=3+r, feature_spec={'role':r})
                    for r in range(2)]
        rustmage_duels._initialize_duel_worker(roles, counter, metadata)
        self.assertEqual(counter.value, 2)
        for role, batcher in zip(roles, rustmage_duels._duel_batchers):
            self.assertIs(batcher.connection, role[1])
            self.assertFalse(role[1].closed)
            self.assertTrue(role[0].closed)
            self.assertTrue(role[2].closed)
        self.assertIsNot(rustmage_duels._duel_batchers[0].connection,
                         rustmage_duels._duel_batchers[1].connection)
        self.assertEqual([m.feature_spec for m in rustmage_duels._duel_models],
                         [{'role':0}, {'role':1}])
        with self.assertRaisesRegex(RuntimeError, 'Unexpected replacement duel worker'):
            rustmage_duels._initialize_duel_worker(roles,
                SimpleNamespace(value=3, get_lock=nullcontext), metadata)


if __name__ == '__main__':
    unittest.main()
