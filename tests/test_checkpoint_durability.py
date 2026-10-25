"""Checkpoint publication waits for durable file data and directory metadata."""
import errno
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
from test_neural import AVAILABLE

if AVAILABLE:
    import numpy as np
    from patchwork_automaton.neural import experiment


class StoredPolicy:
    def __init__(self, model):
        self.model = model

    def save(self, path):
        path.write_bytes(b'closed model data')
        path.with_suffix('.manifest.json').write_text('{"schema": 1}\n')


@unittest.skipUnless(AVAILABLE, 'Install neural dependencies with uv')
class CheckpointDurabilityTests(unittest.TestCase):
    def trainer(self):
        optimizer = SimpleNamespace(variables=[np.asarray(7), np.ones(2)], iterations=7)
        return SimpleNamespace(optimizer=optimizer, critic_optimizer=optimizer, rng=np.random.default_rng(19))

    def save(self, directory, round_):
        return experiment.save_checkpoint(directory, SimpleNamespace(), self.trainer(),
            dict(round=round_, config={}), policy_type=StoredPolicy)

    def test_json_flushes_and_syncs_before_replace_then_syncs_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/'report.json'
            pending = target.with_suffix('.json.tmp')
            events = []
            real_sync, real_replace = os.fsync, Path.replace

            def sync(descriptor):
                directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
                events.append('sync-directory' if directory else 'sync-file')
                if not directory:
                    # A small buffered write would still be invisible here
                    # without flush() before fsync().
                    self.assertEqual(json.loads(pending.read_text()), {'round': 4})
                return real_sync(descriptor)

            def replace(source, destination):
                events.append('replace')
                return real_replace(source, destination)

            with patch.object(os, 'fsync', side_effect=sync), patch.object(Path, 'replace', autospec=True, side_effect=replace):
                experiment.atomic_json(target, {'round': 4})
            self.assertEqual(events, ['sync-file', 'replace'] + ([] if os.name == 'nt' else ['sync-directory']))
            self.assertEqual(json.loads(target.read_text()), {'round': 4})

    def test_json_file_sync_failure_leaves_original_pointer_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/'latest.json'
            target.write_text('{"checkpoint": "round-0001"}\n')
            old = target.read_bytes()
            with patch.object(os, 'fsync', side_effect=OSError(errno.EIO, 'storage failure')):
                with self.assertRaises(OSError):
                    experiment.atomic_json(target, {'checkpoint': 'round-0002'})
            self.assertEqual(target.read_bytes(), old)

    def test_all_checkpoint_artifacts_sync_before_rename_and_latest_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'checkpoints'
            events = []
            real_file, real_directory = experiment._fsync_file, experiment._fsync_directory
            real_rename, real_atomic = Path.rename, experiment.atomic_json

            def sync_file(path):
                events.append(('file', Path(path).name))
                return real_file(path)

            def sync_directory(path):
                events.append(('directory', Path(path).name))
                return real_directory(path)

            def rename(source, destination):
                events.append(('rename', Path(source).name))
                return real_rename(source, destination)

            def atomic(path, value):
                if Path(path).name == 'latest.json':
                    events.append(('publish', 'latest.json'))
                return real_atomic(path, value)

            with (patch.object(experiment, '_fsync_file', side_effect=sync_file),
                  patch.object(experiment, '_fsync_directory', side_effect=sync_directory),
                  patch.object(Path, 'rename', autospec=True, side_effect=rename),
                  patch.object(experiment, 'atomic_json', side_effect=atomic)):
                checkpoint = self.save(directory, 2)
            expected = {'policy.keras', 'policy.manifest.json', 'optimizer.npz', 'critic-optimizer.npz', 'state.json'}
            self.assertEqual({name for kind,name in events if kind == 'file'}, expected)
            rename_index = events.index(('rename', 'round-0002.tmp'))
            self.assertLess(max(index for index,event in enumerate(events) if event[0] == 'file'), rename_index)
            self.assertEqual(events[rename_index-1], ('directory', 'round-0002.tmp'))
            self.assertEqual(events[rename_index+1], ('directory', 'checkpoints'))
            self.assertEqual(events[rename_index+2], ('publish', 'latest.json'))
            self.assertEqual(json.loads((directory/'latest.json').read_text()), {'checkpoint': 'round-0002'})
            self.assertEqual({path.name for path in checkpoint.iterdir()}, expected)

    def test_checkpoint_data_sync_failure_does_not_publish_new_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'checkpoints'
            self.save(directory, 1)
            old = (directory/'latest.json').read_bytes()
            with patch.object(experiment, '_fsync_file', side_effect=OSError(errno.EIO, 'storage failure')):
                with self.assertRaises(OSError):
                    self.save(directory, 2)
            self.assertEqual((directory/'latest.json').read_bytes(), old)
            self.assertFalse((directory/'round-0002').exists())
            self.assertTrue((directory/'round-0002.tmp').is_dir())

    def test_checkpoint_directory_sync_failure_blocks_latest_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'checkpoints'
            self.save(directory, 1)
            old = (directory/'latest.json').read_bytes()
            real_directory = experiment._fsync_directory

            def sync(path):
                if Path(path) == directory:
                    raise OSError(errno.EIO, 'directory sync failed')
                return real_directory(path)

            with patch.object(experiment, '_fsync_directory', side_effect=sync):
                with self.assertRaises(OSError):
                    self.save(directory, 2)
            self.assertTrue((directory/'round-0002').is_dir())
            self.assertEqual((directory/'latest.json').read_bytes(), old)

    def test_directory_sync_skips_windows_but_does_not_mask_posix_errors(self):
        with patch.object(os, 'name', 'nt'), patch.object(os, 'open') as open_directory:
            experiment._fsync_directory('unused')
            open_directory.assert_not_called()
        if os.name != 'nt':
            with tempfile.TemporaryDirectory() as temporary:
                real_close = os.close
                with (patch.object(os, 'fsync', side_effect=OSError(errno.EIO, 'directory sync failed')),
                      patch.object(os, 'close', wraps=real_close) as close):
                    with self.assertRaises(OSError):
                        experiment._fsync_directory(temporary)
                    close.assert_called_once()
