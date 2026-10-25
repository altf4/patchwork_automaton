"""Backups/plots read completed JSON bytes, never a model or raw rollout."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from examples import backup_rustmage_pooled_study as backup
from examples import plot_rustmage_pooled_study as plots


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def zip_bytes():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('config.json', '{}')
    return stream.getvalue()


class PooledStudyArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)/'study'
        self.root.mkdir()
        self.config = {'games_per_round':1000, 'seed':42}
        for round_id in range(3):
            self.checkpoint(round_id)
        put_json(self.root/'training/checkpoints/latest.json', {'checkpoint':'round-0002'})
        self.report = dict(config=self.config, rounds=[{'round':1}, {'round':2}], status='running')
        put_json(self.root/'training/report.json', self.report)
        put_json(self.root/'study.json', {'status':'running', 'run_id':'fixed'})
        put_json(self.root/'initialization.json', {'actor_rl_updates_at_start':0})
        for name in ('training/rollouts/raw.pkl.gz', 'exchange/accepted/raw.pkl.gz', 'training/warmup-rollouts/raw'):
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'RAW DATA MUST NOT LEAVE')

    def checkpoint(self, round_id):
        root = self.root/f'training/checkpoints/round-{round_id:04d}'
        root.mkdir(parents=True, exist_ok=True)
        for name in ('policy.keras', 'optimizer.npz', 'critic-optimizer.npz'):
            (root/name).write_bytes(zip_bytes())
        put_json(root/'policy.manifest.json', {'backend':'rustmage'})
        put_json(root/'state.json', dict(round=round_id, config=self.config,
                                      round_result={'round':round_id}))

    def tearDown(self):
        self.temporary.cleanup()

    def test_full_and_incremental_selection_hash_manifest_and_raw_exclusion(self):
        members, manifest = backup.capture_study(self.root)
        self.assertEqual(manifest['included_checkpoint_rounds'], [0, 1, 2])
        self.assertEqual(manifest['latest_round'], 2)
        self.assertFalse(manifest['raw_rollouts_included'])
        self.assertTrue(all('rollouts' not in name and not name.startswith('exchange/') for name in members))
        self.assertEqual({entry['path'] for entry in manifest['members']}, set(members)-{'backup-manifest.json'})
        stream = io.BytesIO()
        backup.write_backup(members, stream)
        with tarfile.open(fileobj=io.BytesIO(stream.getvalue()), mode='r:gz') as archive:
            actual = {entry.name:archive.extractfile(entry).read() for entry in archive}
            self.assertTrue(all(entry.isfile() and not entry.name.startswith('/') for entry in archive.getmembers()))
        for record in manifest['members']:
            self.assertEqual(len(actual[record['path']]), record['size'])
            self.assertEqual(hashlib.sha256(actual[record['path']]).hexdigest(), record['sha256'])
            if record['path'].endswith('.json'):
                json.loads(actual[record['path']])
        increment, header = backup.capture_study(self.root, after_round=1)
        self.assertEqual(header['included_checkpoint_rounds'], [2])
        self.assertFalse(any('/round-0001/' in name or '/round-0000/' in name for name in increment))
        self.assertEqual(json.loads(increment['training/checkpoints/latest.json'])['checkpoint'], 'round-0002')

    def test_mutable_json_snapshot_is_not_reopened_while_streaming(self):
        members, _ = backup.capture_study(self.root)
        put_json(self.root/'study.json', {'status':'complete', 'different':'later'})
        put_json(self.root/'training/report.json', {'corrupted':'after capture'})
        stream = io.BytesIO()
        backup.write_backup(members, stream)
        with tarfile.open(fileobj=io.BytesIO(stream.getvalue()), mode='r:gz') as archive:
            self.assertEqual(json.load(archive.extractfile('study.json'))['status'], 'running')
            self.assertEqual(json.load(archive.extractfile('training/report.json')), self.report)

    def test_bad_json_or_unrecoverable_report_fails_before_stdout_header(self):
        (self.root/'monitor-0004.json').write_text('{partial JSON')
        output = io.BytesIO()
        fake_stdout = types.SimpleNamespace(buffer=output)
        with patch.object(backup.sys, 'stdout', fake_stdout):
            with self.assertRaises(json.JSONDecodeError):
                backup.main(['--study-root', str(self.root)])
        self.assertEqual(output.getvalue(), b'')
        (self.root/'monitor-0004.json').unlink()
        put_json(self.root/'training/report.json', dict(self.report, rounds=[]))
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            backup.capture_study(self.root)

    def test_one_round_lag_is_explicitly_recoverable_and_larger_gap_rejected(self):
        put_json(self.root/'training/report.json', dict(self.report, rounds=[{'round':1}]))
        _, manifest = backup.capture_study(self.root)
        self.assertTrue(manifest['report_recoverable_from_checkpoint'])
        self.assertEqual(manifest['report_last_round'], 1)
        state = self.root/'training/checkpoints/round-0002/state.json'
        put_json(state, dict(round=2, config=self.config))
        with self.assertRaisesRegex(ValueError, 'round_result'):
            backup.capture_study(self.root)

    def test_latest_advancement_retries_whole_capture_before_output(self):
        latest = self.root/'training/checkpoints/latest.json'
        put_json(latest, {'checkpoint':'round-0001'})
        put_json(self.root/'training/report.json', dict(self.report, rounds=[{'round':1}]))
        read, latest_reads = backup._read, [0]
        def changing_read(path, maximum):
            if Path(path) == latest:
                latest_reads[0] += 1
                if latest_reads[0] == 2:
                    put_json(latest, {'checkpoint':'round-0002'})
                    put_json(self.root/'training/report.json', self.report)
            return read(path, maximum)
        with patch.object(backup, '_read', side_effect=changing_read):
            _, manifest = backup.capture_study(self.root)
        self.assertEqual(manifest['latest_round'], 2)
        self.assertEqual(manifest['report_last_round'], 2)
        self.assertEqual(latest_reads[0], 4)

    def test_unsafe_checkpoint_symlinks_and_bounded_log_tails(self):
        put_json(self.root/'training/checkpoints/latest.json', {'checkpoint':'../escape'})
        with self.assertRaisesRegex(ValueError, 'checkpoint name'):
            backup.capture_study(self.root)
        put_json(self.root/'training/checkpoints/latest.json', {'checkpoint':'round-0002'})
        (self.root/'train-through-0004.log').write_bytes(b'0123456789')
        members, _ = backup.capture_study(self.root, include_logs=True, max_log_bytes=4)
        self.assertEqual(members['log-tails/train-through-0004.log'], b'6789')
        policy = self.root/'training/checkpoints/round-0002/policy.keras'
        policy.unlink()
        policy.symlink_to(self.root/'training/checkpoints/round-0001/policy.keras')
        with self.assertRaisesRegex(ValueError, 'regular file'):
            backup.capture_study(self.root)

    def monitor(self, round_id, *, seed=101):
        name = 'imitation' if round_id == 0 else f'{round_id:04d}'
        put_json(self.root/f'monitor-{name}.json', dict(status='complete',
            checkpoint_sha256='a'*64, config={'cohort':{'seed_base':seed}},
            summary=dict(attempted_games=100, win_rate_all_attempts=.51,
                         win_rate_all_attempts_wilson_95_interval=[.41, .61]),
            games=[{}]*100))

    def test_plot_extracts_whole_game_current_memory_and_seals_final_until_complete(self):
        self.monitor(0)
        self.monitor(4)
        metric = dict(value_after={'mse':.7, 'explained_variance':.2},
            value_before={'mse':.8}, value_mse={'game_bootstrap_95_interval':[.6, .8]}, terminal_games=200)
        put_json(self.root/'training/report.json', dict(self.report, rounds=[
            dict(round=1, generalization={'training':metric, 'validation':metric},
                 update={'value_loss':.1})]))
        paired = dict(recorded_attempt_win_rate_change=.012,
                      recorded_attempt_win_rate_change_bootstrap_95_interval=[.002, .022])
        put_json(self.root/'study.json', dict(status='running', paired_comparisons={'primary-40k-minus-4k':paired}))
        data = plots.collect_points(self.root)
        self.assertEqual([p['collected_games'] for p in data['monitoring']], [0, 4000])
        self.assertEqual(data['critic'][0]['validation_mse'], .7)
        self.assertEqual(data['critic'][0]['validation_explained_variance'], .2)
        self.assertIsNone(data['primary_independent_confirmation'])
        put_json(self.root/'study.json', dict(status='complete', paired_comparisons={'primary-40k-minus-4k':paired}))
        self.assertEqual(plots.collect_points(self.root)['primary_independent_confirmation']['ci95'], [.002, .022])
        self.monitor(10, seed=102)
        with self.assertRaisesRegex(ValueError, 'share'):
            plots.collect_points(self.root)


if __name__ == '__main__':
    unittest.main()
