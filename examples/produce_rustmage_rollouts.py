"""Bounded laptop CPU rollout producer, using outbound pinned SSH only.

No optimizer is loaded or run.  Each closed 32-game shard is durably spooled
before upload; one uploader overlaps the following shard, with at most two
unacknowledged artifacts.  Restart resends persisted artifacts idempotently.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.json-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def safe_extract_snapshot(archive, destination, *, max_bytes=64 * 1024 * 1024):
    """Only the three frozen numerical model files may cross this boundary."""
    destination = Path(destination)
    allowed = {'policy.keras', 'policy.manifest.json', 'weights.npz'}
    with tarfile.open(archive, 'r:*') as source:
        members = source.getmembers()
        if (len(members) != len(allowed) or {m.name for m in members} != allowed
                or any(not m.isfile() or m.size < 0 for m in members)
                or sum(m.size for m in members) > max_bytes):
            raise ValueError('Invalid frozen snapshot archive')
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            with source.extractfile(member) as incoming, (destination/member.name).open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)


class SSHTransport:
    """Binary RPC through a configured, pinned SSH helper; never shell=True."""
    def __init__(self, config):
        self.config = config
        prefix = config['ssh_command']
        if not isinstance(prefix, list) or not prefix or not all(isinstance(v, str) and v for v in prefix):
            raise ValueError('ssh_command must be a nonempty argument-vector prefix')

    def _argv(self, action, *, round_id=None, chunk_id=None):
        project = self.config['remote_project']
        environment = dict(PYTHONPATH=f'{project}/src:{project}', RUSTMAGE_ROOT=f'{project}/engine',
            CUDA_VISIBLE_DEVICES='-1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1', TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1')
        environment.update(self.config.get('remote_env', {}))
        if any(not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', key) or not isinstance(value, str)
               for key,value in environment.items()):
            raise ValueError('Invalid remote RPC environment')
        remote = ['env', *[f'{key}={value}' for key,value in environment.items()],
                  self.config['remote_python'], '-m', 'examples.rollout_exchange_rpc',
                  '--root', self.config['exchange_root'], action]
        if round_id is not None:
            remote += ['--round', str(round_id)]
        if chunk_id is not None:
            remote += ['--chunk', str(chunk_id)]
        command = f"cd {shlex.quote(self.config['remote_project'])} && exec {shlex.join(remote)}"
        return [*self.config['ssh_command'], command]

    def request(self, operation, payload):
        if operation != 'status':
            raise ValueError('Unknown producer RPC')
        result = subprocess.run(self._argv(operation), input=json.dumps(payload).encode(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            timeout=self.config.get('rpc_timeout', 180))
        return json.loads(result.stdout)

    def snapshot(self, round_id, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.snapshot-', dir=destination.parent) as temporary:
            temporary = Path(temporary)
            archive = temporary/'snapshot.tar'
            with archive.open('wb') as output:
                subprocess.run(self._argv('snapshot', round_id=round_id),
                    input=json.dumps(dict(round_id=round_id)).encode(), stdout=output,
                    stderr=subprocess.PIPE, check=True, timeout=self.config.get('rpc_timeout', 180))
            extracted = temporary/'files'
            safe_extract_snapshot(archive, extracted)
            if destination.exists():
                raise ValueError('Snapshot destination must be new')
            os.replace(extracted, destination)
        return destination

    def upload(self, round_id, archive_path):
        with tarfile.open(archive_path, 'r:*') as source:
            metadata = json.load(source.extractfile('chunk.json'))
        with Path(archive_path).open('rb') as incoming:
            result = subprocess.run(self._argv('upload', round_id=round_id,
                chunk_id=metadata['chunk_id']), stdin=incoming, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=True, timeout=self.config.get('rpc_timeout', 180))
        return json.loads(result.stdout)


def configure_cpu(config):
    """Run before importing TensorFlow, and let spawned workers inherit it."""
    limit = config.get('cpu_limit', 4)
    if type(limit) is not int or not 1 <= limit <= 4:
        raise ValueError('Laptop producer requires a CPU limit between one and four')
    available = sorted(os.sched_getaffinity(0))
    cpus = config.get('cpu_ids', available[:limit])
    if (not isinstance(cpus, list) or not 1 <= len(cpus) <= limit
            or len(set(cpus)) != len(cpus) or not set(cpus).issubset(available)):
        raise ValueError('Invalid laptop producer CPU affinity')
    os.sched_setaffinity(0, cpus)
    target_nice = config.get('nice', 5)
    if type(target_nice) is not int or not 5 <= target_nice <= 19:
        raise ValueError('Producer priority must be nice 5 or lower')
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if current_nice < target_nice:
        os.nice(target_nice-current_nice)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    for key in ('TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ['PATCHWORK_OBSERVATION_PIPELINE'] = config.get('observation_pipeline', 'rust')
    os.environ['PATCHWORK_ROLLOUT_STORAGE'] = config.get('rollout_storage', 'pickle')
    workers, replicas = config.get('workers', 8), config.get('inference_processes', 2)
    if type(workers) is not int or not 1 <= workers <= 8 or type(replicas) is not int or not 1 <= replicas <= min(workers, 2):
        raise ValueError('Laptop producer permits at most eight workers and two CPU replicas')
    return dict(cpu_ids=cpus, nice=os.getpriority(os.PRIO_PROCESS, 0), workers=workers,
                inference_processes=replicas, cuda_visible_devices='-1')


def verify_sources(config):
    root = Path(config['project_root']).resolve()
    pins = config.get('source_files', config.get('_source_files'))
    if not pins:
        raise ValueError('Producer requires pinned runtime source and engine files')
    for name, expected in pins.items():
        relative = Path(name)
        path = (root/relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f'Producer runtime source differs: {name}')


def verify_manifest(manifest, lease, config):
    for key in ('identity', 'source_sha256', 'config_sha256'):
        if manifest.get(key) != config[key]:
            raise ValueError(f'Stale or incompatible producer {key}')
    if manifest.get('mode') != 'versus_baseline' or manifest.get('schema_version') != 1:
        raise ValueError('Producer supports recorded fixed-opponent rounds only')
    if (lease.get('round_id') != manifest['round_id']
            or lease.get('manifest_sha256') != manifest['manifest_sha256']):
        raise ValueError('Lease belongs to another frozen round')
    chunk_id = lease.get('chunk_id')
    if (not isinstance(chunk_id, str) or not chunk_id or chunk_id in ('.', '..')
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.' for c in chunk_id)):
        raise ValueError('Invalid producer chunk identifier')
    if set(manifest.get('snapshot', {})) != {'policy.keras','policy.manifest.json','weights.npz'}:
        raise ValueError('Frozen snapshot requires exactly the model, manifest and weights')
    by_index = {g['index']:g for g in manifest['games']}
    games = lease['games']
    if not games or len(games) > manifest['chunk_size']:
        raise ValueError('Invalid producer chunk size')
    for offset, game in enumerate(games):
        if (game != by_index.get(game['index']) or game['index'] not in manifest['laptop_indices']
                or game['index'] != games[0]['index']+offset
                or game['seed'] != games[0]['seed']+offset):
            raise ValueError('Lease changed the scheduled game IDs or seeds')


def load_frozen_model(snapshot, manifest, existing=None):
    import numpy as np
    import keras
    from patchwork_automaton.neural import model as registered_models  # noqa: F401
    from patchwork_automaton.neural.devices import configure_runtime, device_scope, assert_model_device
    from patchwork_automaton.neural.inference_processes import weights_digest
    from patchwork_automaton.neural.rustmage_features import RustMageEncoder
    configure_runtime('cpu', hide_gpu=True)
    snapshot = Path(snapshot)
    for name, metadata in manifest['snapshot'].items():
        path = snapshot/name
        if path.stat().st_size != metadata['size'] or sha256(path) != metadata['sha256']:
            raise ValueError('Corrupt or stale frozen model snapshot')
    saved = json.loads((snapshot/'policy.manifest.json').read_text())
    current = RustMageEncoder(allow_concede=False).manifest()
    if (saved.get('backend') != 'rustmage' or saved.get('allow_concede') is not False
            or saved.get('fingerprint') != current['fingerprint']
            or tuple(saved.get('actions', {})) != tuple(current['actions'])):
        raise ValueError('Frozen checkpoint encoder or ordered action vocabulary differs')
    with device_scope('cpu'):
        network = existing if existing is not None else keras.saving.load_model(snapshot/'policy.keras', compile=False)
        if getattr(network, 'optimizer', None) is not None:
            raise ValueError('A rollout producer must never load an optimizer')
        weights = network.get_weights()
        keys = [f'weight_{i:05d}' for i in range(len(weights))]
        with np.load(snapshot/'weights.npz', allow_pickle=False) as source:
            if sorted(source.files) != keys:
                raise ValueError('Frozen snapshot has a different weight schema')
            values = [source[key] for key in keys]
        if (any(a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(b).all()
                for a,b in zip(weights,values))
                or weights_digest(values) != manifest['weights_sha256']):
            raise ValueError('Frozen actor/critic weights are invalid or stale')
        network.set_weights(values)
    if (network.memory_size != manifest['model']['memory_size']
            or network.card_count != manifest['model']['card_count']
            or dict(network.feature_spec) != manifest['model']['feature_spec']):
        raise ValueError('Frozen actor/critic architecture differs')
    assert_model_device(network, 'cpu')
    return network


class LaptopRolloutProducer:
    def __init__(self, config, transport, *, model_loader=load_frozen_model, collector_factory=None):
        self.config, self.transport = config, transport
        self.model_loader, self.collector_factory = model_loader, collector_factory
        self.spool = Path(config['spool_root']).resolve()
        self.spool.mkdir(parents=True, exist_ok=True)
        self.pending_path = self.spool/'pending.json'
        self.pending = json.loads(self.pending_path.read_text()) if self.pending_path.exists() else []
        if len(self.pending) > 2:
            raise ValueError('Producer has more than two unacknowledged shards')
        for entry in self.pending:
            for key in ('archive', 'trajectory_directory'):
                relative = Path(entry[key])
                if relative.is_absolute() or not (self.spool/relative).resolve().is_relative_to(self.spool):
                    raise ValueError('Persisted producer shard escapes its spool')
        self.collector = self.model = self.version = None
        self.uploads = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rollout-upload')
        self.futures = {}
        self.runtime = None

    def _log(self, stage, **fields):
        event = dict(stage=stage, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
        with (self.spool/'producer.jsonl').open('a') as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False)+'\n')
            stream.flush()
        print(json.dumps(event, sort_keys=True), flush=True)

    def _save_pending(self):
        atomic_json(self.pending_path, self.pending)

    def _send(self, entry):
        archive = self.spool/entry['archive']
        if not archive.resolve().is_relative_to(self.spool) or sha256(archive) != entry['sha256']:
            raise ValueError('Persisted producer shard changed before acknowledgment')
        for attempt in range(self.config.get('upload_attempts', 3)):
            try:
                receipt = self.transport.upload(entry['round_id'], archive)
                if receipt.get('status') not in ('accepted', 'duplicate', 'sealed'):
                    raise ValueError(f'Unknown rollout acceptance response: {receipt}')
                if receipt.get('round_id') != entry['round_id'] or receipt.get('chunk_id') != entry['chunk_id']:
                    raise ValueError('Rollout acknowledgment belongs to another shard')
                if receipt['status'] != 'sealed' and (
                        receipt.get('manifest_sha256') != entry['manifest_sha256']
                        or receipt.get('game_indices') != entry['game_indices']):
                    raise ValueError('Rollout acknowledgment changed the frozen manifest or game IDs')
                return receipt
            except (OSError, subprocess.SubprocessError):
                if attempt+1 == self.config.get('upload_attempts', 3):
                    raise
                time.sleep(min(2**attempt, 5))

    def _submit(self, entry):
        self.futures[entry['archive']] = self.uploads.submit(self._send, entry)

    async def _drain_one(self):
        entry = self.pending[0]
        receipt = await asyncio.wrap_future(self.futures[entry['archive']])
        atomic_json(self.spool/'receipts'/f"round-{entry['round_id']:04d}-{entry['chunk_id']}.json", receipt)
        self._log('shard-acknowledged', round_id=entry['round_id'], chunk_id=entry['chunk_id'],
                  receipt=receipt, game_indices=entry['game_indices'], archive_bytes=entry['bytes'])
        self.pending.pop(0)
        self._save_pending()  # Lost ACK replay remains safe until this durable write.
        self.futures.pop(entry['archive'])
        (self.spool/entry['archive']).unlink()
        shutil.rmtree(self.spool/entry['trajectory_directory'])

    async def _prepare(self, manifest):
        if self.version == (manifest['round_id'], manifest['weights_sha256']):
            return
        while self.pending:
            await self._drain_one()
        verify_sources(self.config)
        snapshot = self.spool/'snapshots'/f"round-{manifest['round_id']:04d}-{manifest['weights_sha256']}"
        if not snapshot.exists():
            await asyncio.to_thread(self.transport.snapshot, manifest['round_id'], snapshot)
        self.model = self.model_loader(snapshot, manifest, self.model)
        if self.collector is None:
            if self.collector_factory:
                self.collector = self.collector_factory(self.model)
            else:
                from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector
                self.collector = ProcessGameCollector(self.model, workers=self.config.get('workers', 8),
                    inference_processes=self.config.get('inference_processes', 2), wait_ms=0, inference_device='cpu')
            await self.collector.start(version=manifest['policy_version'])
        else:
            await self.collector.sync_weights(self.model, manifest['policy_version'])
        if self.collector.policy_digest != manifest['weights_sha256']:
            raise ValueError('Laptop inference replicas did not accept the frozen actor/critic weights')
        self.version = manifest['round_id'], manifest['weights_sha256']
        self._log('policy-ready', round_id=manifest['round_id'], version=manifest['policy_version'],
                  weights_sha256=manifest['weights_sha256'], runtime=self.runtime,
                  inference=self.collector.stats())

    async def run(self, *, max_chunks=None):
        from patchwork_automaton.neural.rollout_exchange import build_chunk
        lock = (self.spool/'producer.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = 0
        try:
            verify_sources(self.config)
            for entry in self.pending:
                self._submit(entry)
            last_work = time.monotonic()
            while max_chunks is None or completed < max_chunks:
                while self.pending and (len(self.pending) >= 2 or self.futures[self.pending[0]['archive']].done()):
                    await self._drain_one()
                response = await asyncio.to_thread(self.transport.request, 'status',
                    dict(producer=self.config.get('producer', 'laptop'), identity=self.config['identity']))
                if response.get('status') == 'closed':
                    break
                if response.get('status') == 'waiting':
                    if time.monotonic()-last_work > self.config.get('idle_timeout', 1800):
                        raise TimeoutError('No compatible frozen rollout round became available')
                    await asyncio.sleep(self.config.get('poll_seconds', 1.))
                    continue
                if response.get('status') != 'work':
                    raise ValueError('Unknown rollout coordinator status')
                manifest, lease = response['manifest'], response['lease']
                verify_manifest(manifest, lease, self.config)
                await self._prepare(manifest)
                if len(self.pending) >= 2:
                    raise RuntimeError('Unacknowledged rollout spool limit exceeded')
                directory = Path('shards')/f"round-{manifest['round_id']:04d}"/str(lease['chunk_id'])
                output = self.spool/directory
                # Uncommitted crash leftovers are not accepted data. Never mix
                # them with the new attempt's closed, checksummed shard.
                if output.exists():
                    shutil.rmtree(output)
                output.mkdir(parents=True)
                games = lease['games']
                before = self.collector.stats()
                rows = await self.collector.run_games(games=len(games), seed_base=games[0]['seed'],
                    index_base=games[0]['index'], mode='versus_baseline', output=output,
                    version=manifest['policy_version'], **manifest['options'])
                after = self.collector.stats()
                for row in rows:
                    row['collection_host'] = 'laptop'
                rows[0]['pool_inference_delta'] = {key:after.get(key,0.)-before.get(key,0.)
                    for key in ('calls','requests','replica_cpu_seconds','inference_seconds')}
                rows[0]['pool_producer_runtime'] = self.runtime
                archive = Path('archives')/f"round-{manifest['round_id']:04d}-{lease['chunk_id']}.tar"
                (self.spool/archive).parent.mkdir(parents=True, exist_ok=True)
                build_chunk(manifest, chunk_id=lease['chunk_id'], claim_token=lease['claim_token'],
                    rows=rows, trajectory_root=output, output_path=self.spool/archive)
                entry = dict(round_id=manifest['round_id'], chunk_id=lease['chunk_id'],
                    manifest_sha256=manifest['manifest_sha256'],
                    archive=str(archive), trajectory_directory=str(directory),
                    sha256=sha256(self.spool/archive), bytes=(self.spool/archive).stat().st_size,
                    game_indices=[g['index'] for g in games])
                self.pending.append(entry)
                self._save_pending()
                self._submit(entry)
                completed += 1
                last_work = time.monotonic()
                self._log('shard-closed', **entry)
            while self.pending:
                await self._drain_one()
        finally:
            # Keep pending.json/artifacts on any failed upload for exact replay.
            try:
                if self.collector:
                    await self.collector.close()
            finally:
                try:
                    await asyncio.to_thread(self.uploads.shutdown, wait=True)
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    lock.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', type=Path, required=True)
    return result


async def main(args):
    config = json.loads(args.config.read_text())
    runtime = configure_cpu(config)
    os.chdir(config['project_root'])
    producer = LaptopRolloutProducer(config, SSHTransport(config))
    producer.runtime = runtime
    await producer.run()


if __name__ == '__main__':
    asyncio.run(main(parser().parse_args()))
