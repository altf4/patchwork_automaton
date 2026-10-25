"""Scheduling/barrier tests without games, TensorFlow, SSH or optimizers."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from patchwork_automaton.neural.inference_processes import weights_digest
from patchwork_automaton.neural.pooled_collector import PooledGameCollector, export_weights


class Model:
    memory_size, card_count, feature_spec = 3, 2, {'context': 4}
    def __init__(self):
        self.weights = [np.asarray([1., 2.], np.float32), np.asarray([[3.]], np.float64)]
    def get_weights(self):
        return [value.copy() for value in self.weights]


def game(index, seed, host=None):
    result = dict(index=index, seed=seed, seat=index%2, starting_player=f'P{(index//2)%2}',
        mode='versus_baseline', status='win', trajectories=[], worker_cpu_seconds=.2,
        max_own_turns=60, decisions=0, neural_choices=0, winner=index%2,
        terminal_rewards=[1.,-1.] if index%2==0 else [-1.,1.])
    if host:
        result['collection_host'] = host
    return result


class Inner:
    def __init__(self, model):
        self.model, self.version, self.policy_digest = model, 7, weights_digest(model.get_weights())
        self.calls = []
        self.closed = False
    async def run_games(self, *, games, seed_base, index_base=0, progress=None, **options):
        self.calls.append(dict(games=games, seed_base=seed_base, index_base=index_base, **options))
        result = [game(index_base+i, seed_base+i) for i in range(games)]
        for row in result:
            if progress:
                progress(row)
        return result
    async def start(self, *, version):
        self.version = version
    async def sync_weights(self, model, version):
        self.version = version
        self.policy_digest = weights_digest(model.get_weights())
    def stats(self):
        return dict(calls=10, requests=40, replica_cpu_seconds=2., version=self.version,
                    weights_sha256=self.policy_digest)
    async def close(self):
        self.closed = True


class Exchange:
    def __init__(self, remote_count=64):
        self.remote_count = remote_count
        self.manifest = self.completed = None
        self.sealed = False
    def open_round(self, manifest):
        if self.manifest is not None and self.manifest != manifest:
            raise ValueError('Different frozen round')
        self.manifest = copy.deepcopy(manifest)
        return dict(manifest, manifest_sha256='manifest-hash')
    def completed_rows(self, round_id):
        return copy.deepcopy(self.completed)
    def accepted_rows(self, round_id):
        return [game(g['index'], g['seed'], 'laptop') for g in self.manifest['games'][:self.remote_count]]
    def seal(self, round_id):
        self.sealed = True
        return self.manifest['laptop_indices'][self.remote_count:]
    def complete_round(self, round_id, rows):
        if not self.sealed:
            raise AssertionError('Round must seal before update barrier')
        self.completed = copy.deepcopy(rows)


class PooledCollectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        policy = root/'warm'/'policy.keras'
        policy.parent.mkdir()
        policy.write_bytes(b'frozen model fixture')
        policy.with_suffix('.manifest.json').write_text('{}')
        self.config = dict(exchange_root=str(root/'exchange'), run_id='test',
            initial_policy=str(policy), rollout_output=str(root/'run'/'rollouts'),
            source_sha256='1'*64, config_sha256='2'*64,
            identity={'encoder': 'pinned'}, grace_seconds=0.)
        self.model = Model()
        self.inner = Inner(self.model)
        self.exchange = Exchange()
        self.pool = PooledGameCollector(self.inner, self.model, config=self.config, exchange=self.exchange)

    def tearDown(self):
        self.temporary.cleanup()

    def collect(self, **extra):
        options = dict(games=1000, seed_base=5000, index_base=1000,
            mode='versus_baseline', output=Path(self.config['rollout_output']), version=7)
        options.update(extra)
        return asyncio.run(self.pool.run_games(**options))

    def test_partial_remote_seals_then_falls_back_exact_ids_and_recovers_barrier(self):
        progress = []
        rows = self.collect(progress=lambda row: progress.append(row['index']))
        self.assertEqual([r['index'] for r in rows], list(range(1000, 2000)))
        self.assertEqual([r['seed'] for r in rows], list(range(5000, 6000)))
        self.assertEqual(len(set(progress)), 1000)
        self.assertEqual([(c['index_base'], c['games'], c['seed_base']) for c in self.inner.calls],
                         [(1128, 872, 5128), (1064, 64, 5064)])
        self.assertEqual(sum(r['collection_host']=='laptop' for r in rows), 64)
        self.assertEqual(sum(bool(r.get('pool_fallback')) for r in rows), 64)
        self.assertEqual(self.exchange.completed, rows)
        self.assertEqual(self.pool.stats()['calls'], 10)  # Local replica placement stays local.
        self.assertEqual(self.pool.stats()['pool']['remote_games'], 64)
        # A crash after collection but before checkpoint reuses precisely these
        # durable rows, without recollecting or changing their behavior values.
        again = self.collect()
        self.assertEqual(again, rows)
        self.assertEqual(len(self.inner.calls), 2)
        self.assertTrue(self.pool.stats()['pool']['rounds'][0]['recovered_complete_round'])

    def test_evaluation_and_warmup_delegate_without_exchange_or_seed_changes(self):
        for options in (dict(mode='versus_baseline', output=None),
                        dict(mode='selfplay', output=Path(self.config['rollout_output'])),
                        dict(mode='versus_baseline', output=Path(self.config['rollout_output']).parent/'warmup')):
            result = self.collect(games=3, index_base=5, seed_base=99, **options)
            self.assertEqual([r['seed'] for r in result], [99, 100, 101])
        self.assertIsNone(self.exchange.manifest)
        self.assertEqual(self.pool.stats()['pool']['remote_games'], 0)

    def test_stale_weights_or_round_configuration_fail_before_submission(self):
        for changes in (dict(version=6), dict(games=999), dict(index_base=1001)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.collect(**changes)
        self.model.weights[0][0] = 4.
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.collect()
        self.assertEqual(self.inner.calls, [])

    def test_frozen_export_includes_critic_arrays_and_rejects_corrupt_resume(self):
        target = Path(self.temporary.name)/'weights.npz'
        digest = weights_digest(self.model.get_weights())
        export_weights(self.model, target, digest)
        with np.load(target, allow_pickle=False) as source:
            self.assertEqual(source.files, ['weight_00000', 'weight_00001'])
            np.testing.assert_array_equal(source['weight_00001'], self.model.weights[1])
        np.savez(target, weight_00000=self.model.weights[0], weight_00001=np.zeros((1, 1)))
        with self.assertRaisesRegex(ValueError, 'different policy'):
            export_weights(self.model, target, digest)

    def test_real_exchange_producer_and_collector_agree_on_full_round_schema(self):
        from examples.produce_rustmage_rollouts import LaptopRolloutProducer
        from patchwork_automaton.neural.rollout_exchange import RolloutExchange
        exchange = RolloutExchange(self.config['exchange_root'])
        config = dict(self.config, grace_seconds=3.)
        pool = PooledGameCollector(self.inner, self.model, config=config, exchange=exchange)
        source = Path(self.temporary.name)/'runtime.py'
        source.write_bytes(b'pinned source fixture')
        worker_config = dict(project_root=self.temporary.name,
            spool_root=str(Path(self.temporary.name)/'producer'),
            identity=config['identity'], source_sha256=config['source_sha256'],
            config_sha256=config['config_sha256'], source_files={
                'runtime.py':hashlib.sha256(source.read_bytes()).hexdigest()})

        class Transport:
            def request(self, operation, payload):
                current = exchange.status()
                if current['status'] == 'waiting':
                    return dict(status='waiting')
                lease = exchange.claim(current['round_id'], identity=payload['identity'])
                return (dict(status='work', manifest=current['manifest'], lease=lease)
                        if lease else dict(status='waiting'))
            def snapshot(self, round_id, destination):
                Path(destination).mkdir(parents=True)
            def upload(self, round_id, archive):
                return exchange.accept_chunk(archive)

        remote = Inner(Model())
        worker = LaptopRolloutProducer(worker_config, Transport(),
            model_loader=lambda snapshot, manifest, existing: remote.model,
            collector_factory=lambda model:remote)
        async def run():
            rows, _ = await asyncio.gather(pool.run_games(games=1000, seed_base=5000,
                index_base=1000, mode='versus_baseline', output=Path(config['rollout_output']),
                version=7), worker.run(max_chunks=4))
            return rows
        from unittest.mock import patch
        with patch('builtins.print'):
            rows = asyncio.run(run())
        self.assertEqual(len(rows), 1000)
        self.assertEqual(sum(r['collection_host']=='laptop' for r in rows), 128)
        self.assertEqual(len(self.inner.calls), 1)
        self.assertEqual(exchange.completed_rows(2), rows)
        self.assertEqual([c['index_base'] for c in remote.calls], [1000,1032,1064,1096])
        self.assertEqual(pool.stats()['pool']['remote_inference']['replica_cpu_seconds'], 0.)


if __name__ == '__main__':
    unittest.main()
