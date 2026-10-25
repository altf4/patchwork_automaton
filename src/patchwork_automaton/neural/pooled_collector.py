"""Round-barrier CPU rollout pooling; optimizer ownership stays on Raytracer.

Only recorded fixed-opponent training is pooled.  Evaluation and critic warm-up
continue through the original collector, with its original seeds and ordering.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from .inference_processes import weights_digest


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def export_weights(model, path, expected_digest):
    """Persist every actor and critic array, without optimizer/RNG state."""
    import numpy as np
    path = Path(path)
    values = model.get_weights()
    if weights_digest(values) != expected_digest:
        raise ValueError('Learner weights changed during frozen collection')
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            keys = [f'weight_{i:05d}' for i in range(len(values))]
            if sorted(saved.files) != keys or weights_digest([saved[k] for k in keys]) != expected_digest:
                raise ValueError('Existing round snapshot has different policy weights')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.weights-', suffix='.npz', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez(stream, **{f'weight_{i:05d}': value for i, value in enumerate(values)})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def contiguous_ranges(indices):
    """Represent an exact, possibly partially accepted schedule without gaps."""
    values = sorted(indices)
    if len(values) != len(set(values)):
        raise ValueError('Duplicate scheduled game index')
    ranges = []
    for index in values:
        if ranges and index == ranges[-1][0] + ranges[-1][1]:
            start, count = ranges[-1]
            ranges[-1] = start, count + 1
        else:
            ranges.append((index, 1))
    return ranges


class PooledGameCollector:
    """Wrap a persistent ProcessGameCollector with a durable coarse exchange.

    ``config`` pins ``exchange_root``, ``run_id``, ``initial_policy``,
    ``rollout_output``, ``identity``, ``source_sha256`` and ``config_sha256``.
    Prefix game IDs are reserved for the laptop; the remainder is one local
    call.  A sealed round rejects late laptop results before local fallback.
    """
    def __init__(self, collector, model, *, config, exchange=None):
        self.collector, self.model, self.config = collector, model, dict(config)
        for key in ('exchange_root', 'run_id', 'initial_policy', 'rollout_output',
                    'identity', 'source_sha256', 'config_sha256'):
            if key not in self.config:
                raise ValueError(f'Missing pooled collector configuration: {key}')
        if exchange is None:
            from .rollout_exchange import RolloutExchange
            exchange = RolloutExchange(config['exchange_root'])
        self.exchange = exchange
        self.output = Path(config['rollout_output']).resolve()
        self.remote_games = config.get('remote_games', 128)
        self.chunk_games = config.get('chunk_games', 32)
        self.grace_seconds = config.get('grace_seconds', 15.)
        if (type(self.remote_games) is not int or self.remote_games < 1
                or type(self.chunk_games) is not int or self.chunk_games < 1
                or self.remote_games % self.chunk_games
                or not 0 <= self.grace_seconds < float('inf')):
            raise ValueError('Invalid pooled collection quota, chunk or grace')
        self.active = False
        self.round_metrics = {}

    def __getattr__(self, name):
        return getattr(self.collector, name)

    @property
    def version(self):
        return self.collector.version

    async def start(self, *, version=0):
        await self.collector.start(version=version)
        return self

    async def sync_weights(self, model, version):
        if self.active:
            raise RuntimeError('Cannot synchronize an active pooled round')
        await self.collector.sync_weights(model, version)

    def stats(self):
        result = dict(self.collector.stats())
        remote = [r for m in self.round_metrics.values() for r in m['remote_rows']]
        result['pool'] = dict(hosts=['raytracer', 'laptop'],
            cpu_accounting='cross-host process CPU seconds; logical CPU counts are host-specific',
            remote_games=len(remote), remote_worker_cpu_seconds=sum(
                r.get('worker_cpu_seconds', 0.) for r in remote),
            remote_inference={key:sum(r.get('pool_inference_delta', {}).get(key, 0.) for r in remote)
                for key in ('calls','requests','replica_cpu_seconds','inference_seconds')},
            rounds=[{k:v for k,v in m.items() if k != 'remote_rows'}
                    for _,m in sorted(self.round_metrics.items())])
        return result

    def _manifest(self, *, games, seed_base, index_base, options):
        if games != self.config.get('games_per_round', 1000) or self.remote_games >= games:
            raise ValueError('Pooled training requires the configured full round schedule')
        if index_base % games:
            raise ValueError('Pooled training requires a round-aligned global index')
        version = options.get('version', self.version)
        if version != self.version:
            raise ValueError('Stale policy version for pooled collection')
        digest = weights_digest(self.model.get_weights())
        if digest != self.collector.policy_digest:
            raise ValueError('Collector and learner disagree on frozen weights')
        round_id = index_base // games + 1
        root = Path(self.config['exchange_root']).resolve() / 'snapshots' / f'round-{round_id:04d}'
        weights = root / 'weights.npz'
        export_weights(self.model, weights, digest)
        policy = Path(self.config['initial_policy']).resolve()
        files = {'policy.keras': policy, 'policy.manifest.json': policy.with_suffix('.manifest.json'),
                 'weights.npz': weights}
        snapshot = {name: dict(path=str(path), sha256=file_sha256(path), size=path.stat().st_size)
                    for name,path in files.items()}
        game_options = dict(max_decisions=6000, max_own_turns=60, repeat_limit=20,
                            greedy=False, compact_views=True)
        game_options.update({key:options[key] for key in game_options if key in options})
        return dict(schema_version=1, run_id=self.config['run_id'], round_id=round_id,
            policy_version=version, weights_sha256=digest,
            source_sha256=self.config['source_sha256'], config_sha256=self.config['config_sha256'],
            identity=self.config['identity'], mode='versus_baseline', options=game_options,
            model=dict(memory_size=self.model.memory_size, card_count=self.model.card_count,
                       feature_spec=dict(self.model.feature_spec)), snapshot=snapshot,
            games=[dict(index=i, seed=seed_base+i-index_base, seat=i%2,
                        starting_player=f'P{(i//2)%2}') for i in range(index_base,index_base+games)],
            laptop_indices=list(range(index_base,index_base+self.remote_games)),
            chunk_size=self.chunk_games)

    async def run_games(self, *, games, seed_base, index_base=0, progress=None, **options):
        pooled = (options.get('mode') == 'versus_baseline' and options.get('output') is not None
                  and Path(options['output']).resolve() == self.output)
        if not pooled:
            return await self.collector.run_games(games=games, seed_base=seed_base,
                index_base=index_base, progress=progress, **options)
        if self.active:
            raise RuntimeError('Pooled collection is already active')
        self.active = True
        started = time.monotonic()
        try:
            manifest = self._manifest(games=games, seed_base=seed_base, index_base=index_base, options=options)
            opened = self.exchange.open_round(manifest)
            round_id = manifest['round_id']
            completed = self.exchange.completed_rows(round_id)
            if completed is not None:
                for row in completed:
                    if progress:
                        progress(row)
                self.round_metrics[round_id] = dict(round=round_id,
                    manifest_sha256=opened.get('manifest_sha256'), recovered_complete_round=True,
                    laptop_games=sum(r['collection_host']=='laptop' for r in completed),
                    raytracer_games=sum(r['collection_host']=='raytracer' for r in completed),
                    fallback_games=sum(bool(r.get('pool_fallback')) for r in completed),
                    seconds=time.monotonic()-started,
                    remote_rows=[r for r in completed if r['collection_host']=='laptop'])
                return completed
            # The remote prefix makes the primary desktop submission contiguous.
            rows = await self.collector.run_games(games=games-self.remote_games,
                seed_base=seed_base+self.remote_games, index_base=index_base+self.remote_games,
                progress=progress, **options)
            for row in rows:
                row['collection_host'] = 'raytracer'
            deadline = time.monotonic()+self.grace_seconds
            while len(self.exchange.accepted_rows(round_id)) < self.remote_games and time.monotonic() < deadline:
                await asyncio.sleep(min(.25, max(0., deadline-time.monotonic())))
            missing = self.exchange.seal(round_id)
            remote_rows = self.exchange.accepted_rows(round_id)
            for row in remote_rows:
                row['collection_host'] = 'laptop'
                if progress:
                    progress(row)
            for first,count in contiguous_ranges(missing):
                fallback = await self.collector.run_games(games=count,
                    seed_base=seed_base+first-index_base, index_base=first, progress=progress, **options)
                for row in fallback:
                    row['collection_host'] = 'raytracer'
                    row['pool_fallback'] = True
                rows.extend(fallback)
            rows.extend(remote_rows)
            rows.sort(key=lambda row: row['index'])
            expected = [(g['index'],g['seed'],g['seat'],g['starting_player']) for g in manifest['games']]
            actual = [(r['index'],r['seed'],r['seat'],r['starting_player']) for r in rows]
            if actual != expected:
                raise ValueError('Pooled round has missing, duplicate or mismatched scheduled games')
            if weights_digest(self.model.get_weights()) != manifest['weights_sha256']:
                raise ValueError('Learner weights changed during pooled collection')
            self.exchange.complete_round(round_id, rows)
            self.round_metrics[round_id] = dict(round=round_id,
                manifest_sha256=opened.get('manifest_sha256'), laptop_games=len(remote_rows),
                raytracer_games=games-len(remote_rows), fallback_games=len(missing),
                seconds=time.monotonic()-started, remote_rows=remote_rows)
            return rows
        finally:
            self.active = False

    async def close(self):
        if self.active:
            raise RuntimeError('Cannot close an active pooled round')
        await self.collector.close()


RaytracerPooledCollector = PooledGameCollector
