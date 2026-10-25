"""CPU-only producer lifecycle tests using closed shard fixtures, no games."""
import asyncio
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples import produce_rustmage_rollouts as producer


def fake_build(manifest, *, chunk_id, claim_token, rows, trajectory_root, output_path):
    metadata = dict(round_id=manifest['round_id'], chunk_id=chunk_id,
                    claim_token=claim_token, rows=rows)
    payload = json.dumps(metadata).encode()
    with tarfile.open(output_path, 'w') as archive:
        member = tarfile.TarInfo('chunk.json')
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    return output_path


class FakeCollector:
    policy_digest = 'frozen-actor-and-critic'
    def __init__(self, second_started):
        self.second_started = second_started
        self.calls, self.closed = [], False
    async def start(self, *, version):
        self.version = version
    async def sync_weights(self, model, version):
        self.version = version
    def stats(self):
        return dict(calls=len(self.calls), requests=3*len(self.calls),
                    replica_cpu_seconds=.2*len(self.calls), inference_seconds=.1*len(self.calls))
    async def run_games(self, *, games, seed_base, index_base, output, **options):
        self.calls.append((games, seed_base, index_base, options))
        if len(self.calls) == 2:
            self.second_started.set()
        return [dict(index=index_base+i, seed=seed_base+i, seat=(index_base+i)%2,
            starting_player=f'P{((index_base+i)//2)%2}', mode='versus_baseline',
            status='win', trajectories=[]) for i in range(games)]
    async def close(self):
        self.closed = True


class Transport:
    def __init__(self, manifest, leases, second_started, *, fail_always=False, lost_ack=False):
        self.manifest, self.leases = manifest, list(leases)
        self.second_started = second_started
        self.fail_always, self.lost_ack = fail_always, lost_ack
        self.accepted, self.attempts, self.receipts = set(), [], []
    def request(self, operation, payload):
        if self.leases:
            return dict(status='work', manifest=self.manifest, lease=self.leases.pop(0))
        return dict(status='closed')
    def snapshot(self, round_id, destination):
        Path(destination).mkdir(parents=True)
    def upload(self, round_id, archive):
        with tarfile.open(archive) as source:
            metadata = json.load(source.extractfile('chunk.json'))
            chunk = metadata['chunk_id']
        self.attempts.append(chunk)
        if self.fail_always:
            raise OSError('Connection dropped')
        if chunk == 'laptop-0000' and self.lost_ack:
            if not self.second_started.wait(2):
                raise AssertionError('Producer waited for upload before starting the next chunk')
            self.accepted.add(chunk)
            self.lost_ack = False
            raise OSError('Accepted remotely, ACK lost')
        status = 'duplicate' if chunk in self.accepted else 'accepted'
        self.accepted.add(chunk)
        receipt = dict(status=status, round_id=round_id, chunk_id=chunk,
            manifest_sha256=self.manifest['manifest_sha256'],
            game_indices=[r['index'] for r in metadata['rows']])
        self.receipts.append(receipt)
        return receipt


class ProducerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        (root/'engine').write_bytes(b'pinned runtime')
        self.config = dict(project_root=str(root), spool_root=str(root/'spool'),
            source_files={'engine':hashlib.sha256(b'pinned runtime').hexdigest()},
            source_sha256='1'*64,
            config_sha256='config', identity={'encoder':'encoder'}, upload_attempts=2)
        self.manifest = dict(schema_version=1, round_id=1, policy_version=0,
            manifest_sha256='manifest', weights_sha256='frozen-actor-and-critic',
            mode='versus_baseline', chunk_size=2, identity=self.config['identity'],
            source_sha256=self.config['source_sha256'], config_sha256='config',
            snapshot={name:{} for name in ('policy.keras','policy.manifest.json','weights.npz')},
            options=dict(max_own_turns=60), laptop_indices=list(range(6)),
            games=[dict(index=i,seed=100+i,seat=i%2,starting_player=f'P{(i//2)%2}') for i in range(6)])
        self.leases = [dict(round_id=1,chunk_id=f'laptop-{i:04d}',claim_token=f'token-{i}',
            manifest_sha256='manifest',games=self.manifest['games'][i*2:i*2+2]) for i in range(3)]
        self.second_started = threading.Event()
        self.collector = FakeCollector(self.second_started)
        self.protocol = patch.dict(sys.modules, {'patchwork_automaton.neural.rollout_exchange':
            SimpleNamespace(build_chunk=fake_build)})
        self.protocol.start()

    def tearDown(self):
        self.protocol.stop()
        self.temporary.cleanup()

    def create(self, transport):
        return producer.LaptopRolloutProducer(self.config, transport,
            model_loader=lambda snapshot, manifest, existing: object(),
            collector_factory=lambda model:self.collector)

    def test_upload_overlaps_collection_and_lost_ack_retries_same_closed_artifact(self):
        transport = Transport(self.manifest, self.leases, self.second_started, lost_ack=True)
        worker = self.create(transport)
        with patch.object(producer.time, 'sleep', return_value=None), patch('builtins.print'):
            asyncio.run(worker.run())
        self.assertEqual(transport.accepted, {'laptop-0000','laptop-0001','laptop-0002'})
        self.assertEqual(transport.attempts.count('laptop-0000'), 2)
        self.assertEqual(transport.receipts[0]['status'], 'duplicate')
        self.assertEqual([(c[1],c[2]) for c in self.collector.calls], [(100,0),(102,2),(104,4)])
        self.assertTrue(self.collector.closed)
        self.assertEqual(json.loads(worker.pending_path.read_text()), [])
        self.assertEqual(list((worker.spool/'archives').glob('*')), [])
        events = [json.loads(line) for line in (worker.spool/'producer.jsonl').read_text().splitlines()]
        self.assertEqual(sum(e['stage']=='shard-acknowledged' for e in events), 3)

    def test_failed_upload_survives_restart_without_recollecting_or_rewriting(self):
        failing = Transport(self.manifest, self.leases[:1], self.second_started, fail_always=True)
        worker = self.create(failing)
        with patch.object(producer.time, 'sleep', return_value=None), patch('builtins.print'):
            with self.assertRaises(OSError):
                asyncio.run(worker.run())
        pending = json.loads(worker.pending_path.read_text())
        self.assertEqual(len(pending), 1)
        archive = worker.spool/pending[0]['archive']
        original_hash = producer.sha256(archive)
        succeeding = Transport(self.manifest, [], self.second_started)
        restarted = self.create(succeeding)
        with patch('builtins.print'):
            asyncio.run(restarted.run())
        self.assertEqual(len(self.collector.calls), 1)
        self.assertEqual(pending[0]['sha256'], original_hash)
        self.assertEqual(succeeding.attempts, ['laptop-0000'])
        self.assertEqual(json.loads(worker.pending_path.read_text()), [])

    def test_stale_critic_or_encoder_identity_and_changed_source_fail_before_collection(self):
        bad = copy.deepcopy(self.manifest)
        bad['identity']['encoder'] = 'stale'
        transport = Transport(bad, self.leases[:1], self.second_started)
        worker = self.create(transport)
        with self.assertRaisesRegex(ValueError, 'identity'):
            asyncio.run(worker.run())
        self.assertEqual(self.collector.calls, [])
        (Path(self.config['project_root'])/'engine').write_bytes(b'changed')
        worker = self.create(transport)
        with self.assertRaisesRegex(ValueError, 'runtime source differs'):
            asyncio.run(worker.run())

    def test_snapshot_rejects_traversal_and_nonregular_members(self):
        root = Path(self.temporary.name)
        for malicious in ('../policy.keras', 'policy.keras'):
            archive = root/'bad.tar'
            with tarfile.open(archive, 'w') as output:
                for name in (malicious,'policy.manifest.json','weights.npz'):
                    member = tarfile.TarInfo(name)
                    if malicious == 'policy.keras' and name == malicious:
                        member.type = tarfile.SYMTYPE
                        member.linkname = '/etc/passwd'
                    output.addfile(member, io.BytesIO(b''))
            with self.assertRaisesRegex(ValueError, 'snapshot archive'):
                producer.safe_extract_snapshot(archive, root/'extracted')
        self.assertFalse((root/'extracted').exists())

    def test_cpu_affinity_and_threads_are_bounded_before_runtime_import(self):
        with patch.object(producer.os, 'sched_getaffinity', return_value={7,9,11,13,15}), \
                patch.object(producer.os, 'sched_setaffinity') as affinity, \
                patch.object(producer.os, 'getpriority', return_value=5), \
                patch.dict(producer.os.environ, {}, clear=True):
            runtime = producer.configure_cpu({})
            affinity.assert_called_once_with(0, [7,9,11,13])
            self.assertEqual(runtime['cpu_ids'], [7,9,11,13])
            self.assertEqual(producer.os.environ['CUDA_VISIBLE_DEVICES'], '-1')
            self.assertEqual(producer.os.environ['TF_NUM_INTEROP_THREADS'], '1')
            self.assertEqual(producer.os.environ['OPENBLAS_NUM_THREADS'], '1')
        with self.assertRaises(ValueError):
            producer.configure_cpu({'cpu_limit':5})

    def test_ssh_rpc_pins_runtime_package_and_native_engine_environment(self):
        transport = producer.SSHTransport(dict(ssh_command=['python', '/pinned helper.py'],
            remote_python='/venv/bin/python', remote_project='/frozen runtime', exchange_root='/exchange'))
        argv = transport._argv('upload', round_id=3, chunk_id='laptop-0000')
        self.assertEqual(argv[:2], ['python','/pinned helper.py'])
        import shlex
        tokens = shlex.split(argv[2])
        self.assertEqual(tokens[:4], ['cd','/frozen runtime','&&','exec'])
        self.assertIn('PYTHONPATH=/frozen runtime/src:/frozen runtime', tokens)
        self.assertIn('RUSTMAGE_ROOT=/frozen runtime/engine', tokens)
        self.assertIn('OPENBLAS_NUM_THREADS=1', tokens)
        self.assertIn('CUDA_VISIBLE_DEVICES=-1', tokens)


if __name__ == '__main__':
    unittest.main()
