"""Filesystem protocol tests: no engine, TensorFlow, training or network."""
import copy
import gzip
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from patchwork_automaton.neural.rollout_exchange import (
    RolloutExchange, ExchangeError, ExchangeLimits, build_chunk, canonical_json,
    file_sha256, json_sha256,
)
import patchwork_automaton.neural.rollout_exchange as exchange_module


class RolloutExchangeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.exchange = RolloutExchange(self.root/'exchange')
        self.manifest = dict(schema_version=1, run_id='pooled-test', round_id=1,
            policy_version=0, weights_sha256='1'*64, source_sha256='2'*64,
            config_sha256='3'*64, identity={'engine':'same', 'encoder':2},
            mode='versus_baseline', model={'memory_size':3}, chunk_size=2,
            games=[dict(index=i, seed=9000+i, seat=i%2, starting_player=f'P{(i//2)%2}')
                   for i in range(8)], laptop_indices=list(range(4)),
            options={'max_own_turns':60})
        self.opened = self.exchange.open_round(self.manifest)
        self.source = self.root/'producer'
        self.source.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def claim(self):
        return self.exchange.claim(1, identity=self.manifest['identity'])

    def sample(self):
        module = types.ModuleType('patchwork_automaton.neural.training')
        Sample = type('Sample', (), {'__module__':module.__name__})
        module.Sample = Sample
        sample = Sample()
        sample.arrays = dict(candidates=np.ones((2, 4), np.float32),
                             candidate_mask=np.ones(2, bool), entities=np.ones((3, 2), np.float32))
        sample.memory = np.zeros(3, np.float32)
        sample.advance, sample.action, sample.value = True, 1, .15
        sample.log_probs = np.log(np.array([.4, .6], np.float32))
        return module, sample

    def rows(self, games, *, terminal=True):
        result = []
        for game in games:
            row = dict(game, mode='versus_baseline', max_own_turns=60, status='win' if terminal else 'cutoff',
                decisions=3, neural_choices=1, trajectories=[], seconds=.1)
            if terminal:
                rewards = [0., 0.]
                rewards[game['seat']], rewards[1-game['seat']] = 1., -1.
                row.update(winner=game['seat'], terminal_rewards=rewards)
                path = self.source/f"game-{game['index']:05d}-p{game['seat']}.pkl.gz"
                module, sample = self.sample()
                with patch.dict(sys.modules, {module.__name__:module}):
                    with gzip.open(path, 'wb') as stream:
                        pickle.dump(sample, stream, protocol=5)
                row['trajectories'] = [dict(path=str(path), samples=1, reward=1., player=game['seat'])]
            result.append(row)
        return result

    def archive(self, lease, *, rows=None, name=None):
        rows = self.rows(lease['games']) if rows is None else rows
        path = self.root/(name or lease['chunk_id']+'.tar')
        build_chunk(self.opened, chunk_id=lease['chunk_id'], claim_token=lease['claim_token'],
                    rows=rows, trajectory_root=self.source, output_path=path)
        return path

    def rewrite(self, archive, mutate):
        with tarfile.open(archive, 'r:') as source:
            members = {member.name:source.extractfile(member).read() for member in source}
        mutate(members)
        output = self.root/(archive.stem+'-modified.tar')
        with tarfile.open(output, 'w', format=tarfile.USTAR_FORMAT) as target:
            for name, payload in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                target.addfile(info, io.BytesIO(payload))
        return output

    def test_import_is_tensorflow_free_and_small_numeric_records_are_accepted(self):
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
        result = subprocess.run([sys.executable, '-c',
            'import sys; import patchwork_automaton.neural.rollout_exchange; '
            'assert "tensorflow" not in sys.modules; assert "keras" not in sys.modules; '
            'assert "patchwork_automaton.neural.training" not in sys.modules'],
            env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        lease = self.claim()
        rows = self.rows(lease['games'])
        rows[0]['pool_inference_delta'] = dict(calls=4, replica_cpu_seconds=.25)
        rows[0]['pool_producer_runtime'] = {'host':'laptop', 'threads':1}
        archive = self.archive(lease, rows=rows)
        self.assertEqual(self.exchange.accept_chunk(archive)['status'], 'accepted')
        accepted = self.exchange.accepted_rows(1)
        self.assertEqual(accepted[0]['pool_inference_delta'], rows[0]['pool_inference_delta'])
        self.assertEqual(accepted[0]['pool_producer_runtime'], rows[0]['pool_producer_runtime'])
        self.assertEqual(accepted[0]['collection_host'], 'laptop')
        self.assertTrue(Path(accepted[0]['trajectories'][0]['path']).is_file())
        self.assertNotEqual(accepted[0]['trajectories'][0]['path'], rows[0]['trajectories'][0]['path'])
        self.assertEqual(file_sha256(accepted[0]['trajectories'][0]['path']),
                         file_sha256(rows[0]['trajectories'][0]['path']))

    def test_distinct_leases_identity_binding_seal_fallback_and_retry(self):
        with self.assertRaisesRegex(ExchangeError, 'identity'):
            self.exchange.claim(1, identity={'encoder':'wrong'})
        first, second = self.claim(), self.claim()
        self.assertEqual([g['index'] for g in first['games']], [0, 1])
        self.assertEqual([g['index'] for g in second['games']], [2, 3])
        self.assertIsNone(self.claim())
        a, b = self.archive(first), self.archive(second)
        self.assertEqual(self.exchange.accept_chunk(a)['status'], 'accepted')
        self.assertEqual(self.exchange.seal(1), [2, 3])
        self.assertEqual(self.exchange.accept_chunk(a)['status'], 'duplicate')
        with patch.object(exchange_module, '_validate_trajectory', side_effect=AssertionError('Must not decode sealed data')):
            self.assertEqual(self.exchange.accept_chunk(b)['status'], 'sealed')
        self.assertEqual(self.exchange.remaining_laptop_indices(1), [2, 3])
        self.assertIsNone(self.exchange.claim(1, identity=self.manifest['identity']))
        rows = self.exchange.accepted_rows(1)+self.rows(self.manifest['games'][2:])
        for row in rows[2:]:
            row.update(collection_host='raytracer', pool_fallback=row['index'] < 4)
        self.assertEqual(self.exchange.complete_round(1, rows)['status'], 'complete')
        self.assertEqual(self.exchange.completed_rows(1), sorted(rows, key=lambda r:r['index']))
        self.assertEqual(self.exchange.complete_round(1, rows)['status'], 'duplicate')
        self.assertTrue(self.exchange.status()['complete'])
        self.assertEqual(RolloutExchange(self.root/'exchange').completed_rows(1), sorted(rows, key=lambda r:r['index']))
        changed = copy.deepcopy(rows)
        changed[0]['seconds'] += 1
        with self.assertRaisesRegex(ExchangeError, 'changed'):
            self.exchange.complete_round(1, changed)

    def test_conflicting_manifest_stale_version_and_wrong_claim_rejected_before_decode(self):
        self.assertEqual(self.exchange.open_round(self.manifest), self.opened)
        changed = copy.deepcopy(self.manifest)
        changed['weights_sha256'] = '4'*64
        with self.assertRaisesRegex(ExchangeError, 'conflicts'):
            self.exchange.open_round(changed)
        archive = self.archive(self.claim())
        for field, value in [('policy_version', 8), ('config_sha256', 'a'*64), ('claim_token', 'b'*32)]:
            def mutate(members, field=field, value=value):
                chunk = json.loads(members['chunk.json'])
                chunk[field] = value
                members['chunk.json'] = canonical_json(chunk)
            altered = self.rewrite(archive, mutate)
            with patch.object(exchange_module, '_validate_trajectory', side_effect=AssertionError('Not trusted yet')):
                with self.assertRaises(ExchangeError):
                    self.exchange.accept_chunk(altered)
        other = copy.deepcopy(self.manifest)
        other['round_id'] = 2
        with self.assertRaisesRegex(ExchangeError, 'unfinished'):
            self.exchange.open_round(other)

    def test_file_checksum_corruption_rejected_before_any_pickle_decode(self):
        archive = self.archive(self.claim())
        def mutate(members):
            name = next(name for name in members if name != 'chunk.json')
            members[name] += b'corruption'
        corrupted = self.rewrite(archive, mutate)
        with patch.object(exchange_module, '_validate_trajectory', side_effect=AssertionError('Not trusted yet')):
            with self.assertRaisesRegex(ExchangeError, 'checksum'):
                self.exchange.accept_chunk(corrupted)
        self.assertEqual(self.exchange.accepted_rows(1), [])
        self.assertEqual(list((self.root/'exchange'/'.incoming').iterdir()), [])

    def test_exact_sample_count_probabilities_and_terminal_seat_returns(self):
        lease = self.claim()
        rows = self.rows(lease['games'])
        rows[0]['trajectories'][0]['samples'] = 2
        rows[0]['neural_choices'] = 2
        archive = self.archive(lease, rows=rows)
        with self.assertRaisesRegex(ExchangeError, 'sample count'):
            self.exchange.accept_chunk(archive)
        rows = self.rows(lease['games'])
        rows[0]['terminal_rewards'] = [-1., 1.]
        with self.assertRaisesRegex(ExchangeError, 'terminal'):
            self.archive(lease, rows=rows, name='reward.tar')
        rows = self.rows(lease['games'])
        rows[0]['trajectories'][0]['player'] = 1
        with self.assertRaisesRegex(ExchangeError, 'opponent'):
            self.archive(lease, rows=rows, name='opponent.tar')
        rows = self.rows(lease['games'])
        module, sample = self.sample()
        sample.log_probs = np.array([0., 0.], np.float32)
        with patch.dict(sys.modules, {module.__name__:module}):
            with gzip.open(rows[0]['trajectories'][0]['path'], 'wb') as stream:
                pickle.dump(sample, stream, protocol=5)
        bad = self.archive(lease, rows=rows, name='probability.tar')
        with self.assertRaisesRegex(ExchangeError, 'probabilities'):
            self.exchange.accept_chunk(bad)

    def test_errors_cutoffs_are_retained_without_training_returns(self):
        lease = self.claim()
        rows = self.rows(lease['games'], terminal=False)
        rows[1].update(status='error', error='native exception', decisions=0, neural_choices=1)
        archive = self.archive(lease, rows=rows)
        self.exchange.accept_chunk(archive)
        accepted = self.exchange.accepted_rows(1)
        self.assertEqual([r['status'] for r in accepted], ['cutoff', 'error'])
        self.assertTrue(all(not r['trajectories'] for r in accepted))
        rows[0]['trajectories'] = [dict(path='fake.pkl.gz', samples=1, reward=0., player=0)]
        with self.assertRaisesRegex(ExchangeError, 'Nonterminal'):
            self.archive(lease, rows=rows, name='fake-return.tar')

    def test_unsafe_paths_links_duplicate_members_and_caps_are_rejected(self):
        for name, kind in [('../escape', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE),
                           ('a\\b', tarfile.REGTYPE), ('trajectory', tarfile.SYMTYPE),
                           ('trajectory', tarfile.LNKTYPE), ('pax', tarfile.XHDTYPE)]:
            path = self.root/'unsafe.tar'
            with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as archive:
                info = tarfile.TarInfo(name)
                info.type = kind
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    info.linkname = '../escape'
                archive.addfile(info, io.BytesIO(b''))
            with self.assertRaises(ExchangeError, msg=(name, kind)):
                self.exchange.accept_chunk(path)
        self.assertFalse((self.root/'escape').exists())
        path = self.root/'duplicate.tar'
        with tarfile.open(path, 'w') as archive:
            for _ in range(2):
                info = tarfile.TarInfo('chunk.json')
                info.size = 2
                archive.addfile(info, io.BytesIO(b'{}'))
        with self.assertRaisesRegex(ExchangeError, 'Duplicate'):
            self.exchange.accept_chunk(path)
        archive = self.archive(self.claim())
        bounded = RolloutExchange(self.root/'exchange', limits=ExchangeLimits(max_archive_bytes=100))
        with self.assertRaisesRegex(ExchangeError, 'oversized'):
            bounded.accept_chunk(archive)
        bounded = RolloutExchange(self.root/'exchange', limits=ExchangeLimits(max_files=1))
        with self.assertRaisesRegex(ExchangeError, 'file count'):
            bounded.accept_chunk(archive)
        bounded = RolloutExchange(self.root/'exchange', limits=ExchangeLimits(max_decompressed_trajectory_bytes=20))
        with self.assertRaisesRegex(ExchangeError, 'Decompressed'):
            bounded.accept_chunk(archive)

    def test_producer_symlink_and_archive_overwrite_protection(self):
        lease = self.claim()
        rows = self.rows(lease['games'])
        original = Path(rows[0]['trajectories'][0]['path'])
        link = self.source/'link.pkl.gz'
        link.symlink_to(original)
        rows[0]['trajectories'][0]['path'] = str(link)
        with self.assertRaisesRegex(ExchangeError, 'root'):
            self.archive(lease, rows=rows)
        archive = self.archive(lease)
        prior = file_sha256(archive)
        with self.assertRaises(FileExistsError):
            self.archive(lease)
        self.assertEqual(file_sha256(archive), prior)

    def test_recover_rename_before_ledger_fsync_and_conflicting_retry(self):
        lease = self.claim()
        archive = self.archive(lease)
        atomic = exchange_module._atomic_json
        def fail_state(path, value):
            if Path(path).name == 'state.json' and value.get('accepted'):
                raise OSError('simulated power loss after accepted rename')
            return atomic(path, value)
        with patch.object(exchange_module, '_atomic_json', side_effect=fail_state):
            with self.assertRaisesRegex(OSError, 'power loss'):
                self.exchange.accept_chunk(archive)
        recovered = RolloutExchange(self.root/'exchange')
        self.assertEqual([r['index'] for r in recovered.accepted_rows(1)], [0, 1])
        self.assertEqual(recovered.accept_chunk(archive)['status'], 'duplicate')
        def mutate(members):
            chunk = json.loads(members['chunk.json'])
            chunk['rows'][0]['seconds'] = 999.
            members['chunk.json'] = canonical_json(chunk)
        with self.assertRaisesRegex(ExchangeError, 'Conflicting retry'):
            recovered.accept_chunk(self.rewrite(archive, mutate))

    def test_restricted_decoder_rejects_pickle_code_execution(self):
        marker = self.root/'SHOULD-NOT-EXIST'
        class Malicious:
            def __reduce__(self):
                return os.system, (f'touch {marker}',)
        lease = self.claim()
        rows = self.rows(lease['games'])
        with gzip.open(rows[0]['trajectories'][0]['path'], 'wb') as stream:
            pickle.dump(Malicious(), stream, protocol=5)
        archive = self.archive(lease, rows=rows)
        with self.assertRaisesRegex(ExchangeError, 'Forbidden'):
            self.exchange.accept_chunk(archive)
        self.assertFalse(marker.exists())

    def test_indexed_numeric_format_uses_actual_records_without_training_import(self):
        from patchwork_automaton.neural.indexed_rollouts import IndexedTrajectoryWriter
        lease = self.claim()
        rows = self.rows(lease['games'])
        for row in rows:
            _, sample = self.sample()
            writer = IndexedTrajectoryWriter(row['trajectories'][0]['path'])
            writer.write(sample)
            writer.close()
            row['trajectories'][0]['path'] = str(writer.path)
        archive = self.archive(lease, rows=rows)
        self.assertEqual(self.exchange.accept_chunk(archive)['status'], 'accepted')
        self.assertTrue(all(r['trajectories'][0]['path'].endswith('.rlt')
                            for r in self.exchange.accepted_rows(1)))

    def test_round_initialization_recovery_and_global_schedule_not_reused(self):
        state = self.root/'exchange'/'round-000001'/'state.json'
        state.unlink()
        self.assertEqual(self.exchange.open_round(self.manifest), self.opened)
        self.exchange.seal(1)
        rows = self.rows(self.manifest['games'], terminal=False)
        self.exchange.complete_round(1, rows)
        second = copy.deepcopy(self.manifest)
        second['round_id'] = 2
        with self.assertRaisesRegex(ExchangeError, 'reuses'):
            self.exchange.open_round(second)
        second['games'] = [dict(g, index=g['index']+8, seed=g['seed']+8)
                           for g in self.manifest['games']]
        second['laptop_indices'] = list(range(8, 12))
        self.exchange.open_round(second)
        self.assertEqual(self.exchange.status()['round_id'], 2)
        with self.assertRaisesRegex(ExchangeError, 'stale'):
            self.exchange.open_round(self.manifest)

    def test_archive_game_ownership_and_corruption_on_completed_restart(self):
        lease = self.claim()
        archive = self.archive(lease)
        def mutate(members):
            chunk = json.loads(members['chunk.json'])
            first, second = chunk['rows']
            first['trajectories'][0]['path'], second['trajectories'][0]['path'] = (
                second['trajectories'][0]['path'], first['trajectories'][0]['path'])
            members['chunk.json'] = canonical_json(chunk)
        with self.assertRaisesRegex(ExchangeError, 'another game'):
            self.exchange.accept_chunk(self.rewrite(archive, mutate))
        self.exchange.accept_chunk(archive)
        self.exchange.seal(1)
        accepted = self.exchange.accepted_rows(1)
        self.exchange.complete_round(1, accepted+self.rows(self.manifest['games'][2:], terminal=False))
        trajectory = Path(accepted[0]['trajectories'][0]['path'])
        trajectory.write_bytes(b'power loss or corruption')
        with self.assertRaisesRegex(ExchangeError, 'file changed'):
            RolloutExchange(self.root/'exchange').completed_rows(1)

    def test_atomic_durable_order_and_no_publish_on_fsync_failure(self):
        path = self.root/'durable.json'
        events = []
        original_sync, original_replace = os.fsync, os.replace
        def sync(descriptor):
            events.append('sync')
            return original_sync(descriptor)
        def replace(source, destination):
            events.append('replace')
            return original_replace(source, destination)
        with patch.object(exchange_module.os, 'fsync', side_effect=sync), \
             patch.object(exchange_module.os, 'replace', side_effect=replace):
            exchange_module._atomic_json(path, {'ok':True})
        self.assertEqual(events, ['sync', 'replace', 'sync'])
        with patch.object(exchange_module.os, 'fsync', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                exchange_module._atomic_json(path, {'ok':False})
        self.assertEqual(json.loads(path.read_text()), {'ok':True})


if __name__ == '__main__':
    unittest.main()
