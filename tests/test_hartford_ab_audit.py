"""Exercise the final A/B auditor against 50,000 synthetic durable outcomes."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples import train_hartford_ppo as driver


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('hartford_ab_training_audit', ROOT /
    'docs/benchmarks/hartford-identity-opponents-ab-2026-11-03/audit_training.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')


class CompletionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='hartford-ab-audit-test-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.output = cls.root / 'control'
        cls.project = cls.root / 'project'
        for name in ('src/example.py', 'examples/example.py', 'scripts/build-direct-standard.py'):
            path = cls.project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# Synthetic audit fixture; no executable experiment.\n')
        engine = cls.root / 'engine.so'
        engine.write_bytes(b'synthetic-engine-pin')
        cls.engine = dict(library=str(engine), library_sha256=audit.sha(engine))
        parent = cls.root / 'parent/checkpoints/round-0100'
        parent.mkdir(parents=True)
        for name in driver.CHECKPOINT_FILES:
            if name != 'state.json':
                (parent / name).write_bytes(('parent-' + name).encode())
        cls.parent_state = dict(round=100, optimizer_steps=1234, critic_optimizer_steps=15700,
            trainer_rng_state=dict(bit_generator='PCG64', state=dict(state=123456789, inc=13579)))
        write(parent / 'state.json', cls.parent_state)
        cls.config = dict(rounds=50, seed=412610910000, skip_evaluations=True,
            opponent_recipe='current', learning_rate=.0001,
            fork=dict(parent_checkpoint=str(parent), parent_hashes=driver.checkpoint_hashes(parent),
                external_opponents={str(v): dict(hashes={'policy.keras': audit.sha(parent / 'policy.keras')})
                    for v in (1097, 1098, 1099)}))
        source_hashes = driver.source_manifest(cls.project)
        write(cls.output / 'manifest.json', dict(config=cls.config,
            pins=dict(source=source_hashes, readiness=dict(engine=cls.engine))))
        checkpoints = {}
        for number in range(51):
            folder = cls.output / 'checkpoints' / f'round-{number:04d}'
            folder.mkdir(parents=True)
            for name in driver.CHECKPOINT_FILES:
                if name != 'state.json':
                    content = (parent / name).read_bytes() if number == 0 else f'{number}:{name}'.encode()
                    (folder / name).write_bytes(content)
            state = dict(cls.parent_state, round=number, config=cls.config)
            if number:
                state.update(optimizer_steps=1234 + number, critic_optimizer_steps=15700 + number)
            write(folder / 'state.json', state)
            checkpoints[str(number)] = driver.checkpoint_hashes(folder)
        rounds = []
        for number in range(1, 51):
            planned = driver.planned_training(cls.config, number)
            rows = []
            for game in planned:
                version = game['opponent_version']
                external = cls.config['fork']['external_opponents'].get(str(version))
                game['opponent_checkpoint_sha256'] = (external['hashes']['policy.keras'] if external
                    else checkpoints[str(version)]['policy.keras'])
                rows.append(dict(game, status='draw', draw=True, adjudicated=True,
                    reason='wall_clock_timeout', learner_won=False, winner_seat=None,
                    trajectories=[]))
            games = cls.output / 'rounds' / f'round-{number:04d}' / 'games.json'
            write(games, rows)
            result = dict(round=number, games_file=str(games), games_sha256=audit.sha(games),
                training_games=800, validation_games=200)
            rounds.append(result)
            folder = cls.output / 'checkpoints' / f'round-{number:04d}'
            state = audit.read(folder / 'state.json')
            state['round_result'] = result
            write(folder / 'state.json', state)
            checkpoints[str(number)] = driver.checkpoint_hashes(folder)
        write(cls.output / 'checkpoints/latest.json', {'checkpoint': 'round-0050'})
        cls.report = dict(status='complete', config=cls.config, completed_games=50000,
            rounds=rounds, checkpoints=checkpoints, evaluations={}, evaluation_games=0)
        write(cls.output / 'report.json', cls.report)
        cls.launch = dict(project=str(cls.project), source_hashes=source_hashes, engine=cls.engine,
            engine_files={str(engine): audit.sha(engine)},
            preserved_files={str(parent / name): digest for name, digest in driver.checkpoint_hashes(parent).items()},
            arms=dict(control=dict(output=str(cls.output), config=cls.config,
                initial_checkpoint_hashes=checkpoints['0'])))

    @contextmanager
    def restore(self, *paths):
        before = {path: path.read_bytes() for path in paths}
        try:
            yield
        finally:
            for path, content in before.items():
                path.write_bytes(content)

    def call_audit(self, launch=None):
        with patch.object(audit, 'finite_checkpoint', return_value=1) as finite:
            result = audit.audit_arm(launch or self.launch, 'control')
            self.assertEqual(finite.call_count, 51)
        return result

    def test_complete_audit_counts_all_games_and_checkpoints_without_writing(self):
        before = {str(path): audit.sha(path) for path in self.root.rglob('*') if path.is_file()}
        count = audit.verify_static(self.launch)
        result = self.call_audit()
        self.assertTrue(result['passed'])
        self.assertEqual((result['games'], result['checkpoints'], result['evaluation_games']), (50000, 51, 0))
        self.assertEqual(result['partitions'], {'training': 40000, 'validation': 10000})
        self.assertEqual(result['outcomes'], {'draw': 50000})
        self.assertEqual(result['initial_optimizer_steps'], 1234)
        self.assertEqual(result['initial_critic_optimizer_steps'], 15700)
        self.assertEqual(count, 9)
        self.assertEqual(before, {str(path): audit.sha(path) for path in self.root.rglob('*') if path.is_file()})

    def test_reject_changed_arm_configuration(self):
        path = self.output / 'report.json'
        with self.restore(path):
            report = audit.read(path)
            report['config']['learning_rate'] = .01
            write(path, report)
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                self.call_audit()

    def test_reject_changed_game_file_even_if_json_still_valid(self):
        path = self.output / 'rounds/round-0001/games.json'
        with self.restore(path):
            path.write_bytes(path.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'summary changed'):
                self.call_audit()

    def test_reject_rehashed_schedule_corruption(self):
        games = self.output / 'rounds/round-0001/games.json'
        state_path = self.output / 'checkpoints/round-0001/state.json'
        report_path = self.output / 'report.json'
        with self.restore(games, state_path, report_path):
            rows = audit.read(games)
            rows[0]['seed'] += 999999
            write(games, rows)
            report = audit.read(report_path)
            report['rounds'][0]['games_sha256'] = audit.sha(games)
            state = audit.read(state_path)
            state['round_result'] = report['rounds'][0]
            write(state_path, state)
            report['checkpoints']['1'] = driver.checkpoint_hashes(state_path.parent)
            write(report_path, report)
            with self.assertRaisesRegex(ValueError, 'schedule or frozen policy'):
                self.call_audit()

    def test_reject_rehashed_initial_rng_or_optimizer_reset(self):
        state_path = self.output / 'checkpoints/round-0000/state.json'
        report_path = self.output / 'report.json'
        for key, reset in (('optimizer_steps', 0), ('critic_optimizer_steps', 0),
                ('trainer_rng_state', {'reset': True})):
            with self.subTest(key=key), self.restore(state_path, report_path):
                state = audit.read(state_path)
                state[key] = reset
                write(state_path, state)
                report = audit.read(report_path)
                report['checkpoints']['0'] = driver.checkpoint_hashes(state_path.parent)
                write(report_path, report)
                launch = deepcopy(self.launch)
                launch['arms']['control']['initial_checkpoint_hashes'] = report['checkpoints']['0']
                with self.assertRaisesRegex(ValueError, 'Fork reset training state: ' + key):
                    self.call_audit(launch)

    def test_reject_rehashed_checkpoint_configuration_drift(self):
        state_path = self.output / 'checkpoints/round-0050/state.json'
        report_path = self.output / 'report.json'
        with self.restore(state_path, report_path):
            state = audit.read(state_path)
            state['config']['learning_rate'] = .01
            write(state_path, state)
            report = audit.read(report_path)
            report['checkpoints']['50'] = driver.checkpoint_hashes(state_path.parent)
            write(report_path, report)
            with self.assertRaisesRegex(ValueError, 'Checkpoint metadata differs'):
                self.call_audit()

    def test_static_integrity_rejects_changed_engine_and_source(self):
        for path in (Path(self.engine['library']), self.project / 'src/example.py'):
            with self.subTest(path=path.name), self.restore(path):
                path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Preserved artifact changed'):
                    audit.verify_static(self.launch)


if __name__ == '__main__':
    unittest.main()
