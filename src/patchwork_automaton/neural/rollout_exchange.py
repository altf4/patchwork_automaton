"""Durable, TensorFlow-free exchange for closed, versioned rollout chunks.

The SSH transport is deliberately outside this module.  A lease is a fixed part
of a frozen round, never permission to sample a different policy or seed.  Only
closed files authenticated against that lease are decoded; ordinary pickle
imports (and consequently TensorFlow) are never used here.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import shutil
import struct
import tarfile
import tempfile
import uuid

import numpy as np


@dataclass(frozen=True)
class ExchangeLimits:
    max_archive_bytes: int = 512 * 1024**2
    max_unpacked_bytes: int = 1024**3
    max_member_bytes: int = 256 * 1024**2
    max_manifest_bytes: int = 4 * 1024**2
    max_files: int = 128
    max_decompressed_trajectory_bytes: int = 1024**3
    max_samples_per_trajectory: int = 100000


class ExchangeError(ValueError):
    """Untrusted, conflicting, stale, or incomplete exchange data."""


def canonical_json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (ValueError, TypeError) as error:
        raise ExchangeError('Exchange metadata must be finite JSON') from error


def json_sha256(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(path):
    # Windows cannot open directories through os.open; do not suppress genuine
    # Linux fsync errors, including ENOSPC/EIO on the durable commit boundary.
    if os.name == 'nt':
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.'+path.name+'-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path, limit=4 * 1024**2):
    path = Path(path)
    if path.stat().st_size > limit:
        raise ExchangeError('Exchange JSON is too large')
    try:
        value = json.loads(path.read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(
            ExchangeError('Nonfinite exchange JSON')))
        canonical_json(value)
        return value
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ExchangeError('Invalid exchange JSON') from error


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ExchangeError(f'Invalid {label}')
    return value


def _digest(value, label):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ExchangeError(f'Invalid {label} SHA256')
    return value


def _safe_name(name):
    if (not isinstance(name, str) or not name or '\\' in name or '\x00' in name
            or PurePosixPath(name).is_absolute() or any(p in ('', '.', '..') for p in name.split('/'))
            or PurePosixPath(name).as_posix() != name or ':' in name):
        raise ExchangeError('Unsafe archive path')
    return name


def _manifest(value):
    manifest = copy.deepcopy(value)
    supplied = manifest.pop('manifest_sha256', None)
    canonical_json(manifest)
    if manifest.get('schema_version') != 1:
        raise ExchangeError('Unsupported round schema')
    if (not isinstance(manifest.get('run_id'), str) or not manifest['run_id']
            or len(manifest['run_id']) > 200):
        raise ExchangeError('Invalid run identifier')
    _integer(manifest.get('round_id'), 'round ID')
    _integer(manifest.get('policy_version'), 'policy version')
    for field in ('weights_sha256', 'source_sha256', 'config_sha256'):
        _digest(manifest.get(field), field)
    if manifest.get('mode') not in ('versus_baseline', 'selfplay'):
        raise ExchangeError('Unsupported pooled training mode')
    games = manifest.get('games')
    if not isinstance(games, list) or not games or len(games) > 100000:
        raise ExchangeError('Invalid round game schedule')
    indices = []
    seeds = []
    for game in games:
        if not isinstance(game, dict):
            raise ExchangeError('Invalid scheduled game')
        index = _integer(game.get('index'), 'game index')
        seeds.append(_integer(game.get('seed'), 'game seed'))
        if game.get('seat') != index % 2 or type(game.get('seat')) is not int:
            raise ExchangeError('Scheduled seat does not match global index')
        if game.get('starting_player') != f'P{(index//2)%2}':
            raise ExchangeError('Scheduled starter does not match global index')
        indices.append(index)
    if indices != sorted(set(indices)) or len(seeds) != len(set(seeds)):
        raise ExchangeError('Duplicate or unsorted round schedule')
    laptop = manifest.get('laptop_indices')
    chunk = _integer(manifest.get('chunk_size'), 'chunk size', 1)
    if (not isinstance(laptop, list) or not laptop or laptop != indices[:len(laptop)]
            or len(laptop) >= len(games) or len(laptop) % chunk):
        raise ExchangeError('Laptop allocation must be a complete chunked prefix')
    if 'identity' in manifest and not isinstance(manifest['identity'], dict):
        raise ExchangeError('Invalid producer identity')
    digest = json_sha256(manifest)
    if supplied is not None and supplied != digest:
        raise ExchangeError('Round manifest digest changed')
    return dict(manifest, manifest_sha256=digest)


def _chunks(manifest):
    by_index = {g['index']: g for g in manifest['games']}
    indices, size = manifest['laptop_indices'], manifest['chunk_size']
    return {f'laptop-{i//size:04d}': [by_index[n] for n in indices[i:i+size]]
            for i in range(0, len(indices), size)}


def _rows(rows, manifest, expected):
    """Validate schedule, terminal rewards and descriptor ownership, no decoding."""
    canonical_json(rows)
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ExchangeError('Chunk has missing or extra scheduled games')
    expected = {g['index']: g for g in expected}
    seen, paths = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ExchangeError('Invalid game row')
        index = _integer(row.get('index'), 'row index')
        game = expected.get(index)
        if index in seen or game is None:
            raise ExchangeError('Duplicate or unassigned game index')
        seen.add(index)
        if any(type(row.get(k)) is not type(game[k]) or row.get(k) != game[k]
               for k in ('index', 'seed', 'seat', 'starting_player')):
            raise ExchangeError('Game schedule differs from frozen round')
        if row.get('mode') != manifest['mode']:
            raise ExchangeError('Game mode differs from frozen round')
        for key in ('max_own_turns',):
            if key in manifest.get('options', {}) and row.get(key) != manifest['options'][key]:
                raise ExchangeError('Game cutoff differs from frozen round')
        decisions = _integer(row.get('decisions'), 'decision count')
        choices = _integer(row.get('neural_choices'), 'neural choice count')
        # A failed native submission occurs after policy.act has recorded its
        # genuine choice but before the completed-decision counter advances.
        if choices > decisions+(row.get('status') == 'error'):
            raise ExchangeError('Neural choices exceed decisions')
        status, descriptors = row.get('status'), row.get('trajectories')
        if status not in ('win', 'loss', 'draw', 'cutoff', 'error') or not isinstance(descriptors, list):
            raise ExchangeError('Invalid game outcome')
        if status in ('cutoff', 'error'):
            if descriptors:
                raise ExchangeError('Nonterminal game cannot provide training returns')
            continue
        seat = row['seat']
        reward = {'win': 1., 'loss': -1., 'draw': 0.}[status]
        rewards = [0., 0.]
        rewards[seat], rewards[1-seat] = reward, -reward
        winner = None if status == 'draw' else seat if status == 'win' else 1-seat
        if row.get('terminal_rewards') != rewards or row.get('winner') != winner:
            raise ExchangeError('Native terminal outcome/rewards disagree')
        players = set()
        total = 0
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                raise ExchangeError('Invalid trajectory descriptor')
            path = descriptor.get('path')
            if not isinstance(path, str) or path in paths:
                raise ExchangeError('Duplicate or invalid trajectory path')
            paths.add(path)
            samples = _integer(descriptor.get('samples'), 'trajectory samples', 1)
            total += samples
            player = descriptor.get('player')
            if player is None and manifest['mode'] == 'selfplay':
                match = re.fullmatch(r'game-\d+-p([01])\.(pkl\.gz|rlt)', Path(path).name)
                player = int(match.group(1)) if match else None
            if type(player) is not int or player not in (0, 1) or player in players:
                raise ExchangeError('Invalid trajectory seat ownership')
            players.add(player)
            if manifest['mode'] == 'versus_baseline' and player != seat:
                raise ExchangeError('Fixed opponent trajectory must not be recorded')
            if type(descriptor.get('reward')) not in (int, float) or descriptor['reward'] != rewards[player]:
                raise ExchangeError('Trajectory reward differs from terminal outcome')
        if total != choices or (choices and not descriptors):
            raise ExchangeError('Recorded sample count differs from neural choices')
    return sorted(rows, key=lambda row: row['index'])


# Plain data stand-in for the production dataclass: importing training.Sample
# here would import TensorFlow in the RPC/transport service.
class _Sample:
    pass


class _NumericUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('patchwork_automaton.neural.training', 'Sample'):
            return _Sample
        if module == 'numpy' and name in ('dtype', 'ndarray'):
            return getattr(np, name)
        if module in ('numpy.core.numeric', 'numpy._core.numeric') and name == '_frombuffer':
            return np._core.numeric._frombuffer if hasattr(np, '_core') else np.core.numeric._frombuffer
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name in ('_reconstruct', 'scalar'):
            core = np._core if hasattr(np, '_core') else np.core
            return getattr(core.multiarray, name)
        raise ExchangeError(f'Forbidden rollout pickle global: {module}.{name}')


class _BoundedRead:
    def __init__(self, stream, maximum):
        self.stream, self.maximum, self.count = stream, maximum, 0

    def read(self, size=-1):
        if size < 0 or size > self.maximum-self.count:
            size = self.maximum-self.count+1
        result = self.stream.read(size)
        self.count += len(result)
        if self.count > self.maximum:
            raise ExchangeError('Decompressed trajectory exceeds limit')
        return result

    def readline(self, size=-1):
        # Protocol 5 needs read(), but protocol 0 must not bypass the bound.
        result = bytearray()
        while size < 0 or len(result) < size:
            item = self.read(1)
            if not item:
                break
            result.extend(item)
            if item == b'\n':
                break
        return bytes(result)


def _sample_valid(sample, manifest):
    if not isinstance(sample, _Sample) or set(vars(sample)) != {
            'arrays', 'memory', 'advance', 'action', 'value', 'log_probs'}:
        raise ExchangeError('Unexpected trajectory record schema')
    if not isinstance(sample.arrays, dict) or not sample.arrays:
        raise ExchangeError('Missing numeric observation arrays')
    arrays = list(sample.arrays.values())+[sample.memory, sample.log_probs]
    if any(not isinstance(a, np.ndarray) or not 1 <= a.ndim <= 3
           or a.dtype.kind not in 'biuf' or not np.isfinite(a).all() for a in arrays):
        raise ExchangeError('Invalid or nonfinite rollout arrays')
    size = manifest.get('model', {}).get('memory_size')
    if sample.memory.ndim != 1 or (size is not None and sample.memory.shape != (size,)):
        raise ExchangeError('Recurrent memory shape differs from frozen model')
    if type(sample.advance) is not bool or not sample.advance:
        raise ExchangeError('Expected advancing genuine neural choices')
    if type(sample.value) not in (float, int) or not math.isfinite(sample.value):
        raise ExchangeError('Invalid behavior value')
    mask = sample.arrays.get('candidate_mask')
    if (not isinstance(mask, np.ndarray) or mask.ndim != 1 or len(mask) < 2
            or sample.log_probs.shape != mask.shape or not np.all(mask)
            or type(sample.action) is not int or not 0 <= sample.action < len(mask)
            or abs(float(np.exp(sample.log_probs.astype(np.float64)).sum())-1.) > 1e-5):
        raise ExchangeError('Invalid chosen action or full behavior probabilities')
    candidates = sample.arrays.get('candidates')
    if not isinstance(candidates, np.ndarray) or candidates.shape[0] != len(mask):
        raise ExchangeError('Candidate arrays disagree with behavior distribution')


def _validate_trajectory(path, samples, manifest, limits):
    if samples > limits.max_samples_per_trajectory:
        raise ExchangeError('Trajectory sample limit exceeded')
    count = 0
    if str(path).endswith('.rlt'):
        from .indexed_rollouts import IndexedTrajectoryReader, SCALAR_RECORD, HEADER
        # Decode numeric records directly: Reader.get imports training.Sample.
        with IndexedTrajectoryReader(path) as reader:
            if len(reader) != samples:
                raise ExchangeError('Indexed trajectory sample count mismatch')
            for position in range(len(reader)):
                width = len(reader.fields)*4*8
                raw = reader._read(width+(8 if position+1 < len(reader) else 0), reader.index_offset+position*width)
                row = np.frombuffer(raw, dtype='<u8', count=len(reader.fields)*4).reshape(len(reader.fields), 4)
                action, value, advance, has_probs = SCALAR_RECORD.unpack(reader._read(
                    SCALAR_RECORD.size, reader.scalars_offset+position*SCALAR_RECORD.size))
                start = int(row[0, 0])
                end = struct.unpack_from('<Q', raw, width)[0] if position+1 < len(reader) else reader.data_end
                if not HEADER.size <= start <= end <= reader.data_end or end-start > limits.max_member_bytes:
                    raise ExchangeError('Invalid indexed trajectory bounds')
                payload, values = reader._read(end-start, start), []
                for i, (rank, dtype) in enumerate(reader.specs):
                    offset = int(row[i, 0])
                    shape = tuple(map(int, row[i, 1:rank+1]))
                    length = math.prod(shape)*dtype.itemsize
                    if not start <= offset <= offset+length <= end:
                        raise ExchangeError('Invalid indexed trajectory array bounds')
                    values.append(np.ndarray(shape, dtype=dtype, buffer=payload, offset=offset-start))
                record = _Sample()
                record.arrays = dict(zip((f['name'] for f in reader.fields[:-2]), values[:-2]))
                record.memory, record.log_probs = values[-2:]
                record.action, record.value, record.advance = action, value, advance
                if not has_probs:
                    raise ExchangeError('Missing behavior probabilities')
                _sample_valid(record, manifest)
        return
    try:
        with gzip.open(path, 'rb') as compressed:
            bounded = _BoundedRead(compressed, limits.max_decompressed_trajectory_bytes)
            while True:
                # A new Unpickler per record matches pickle.dump per write and
                # prevents memo references from crossing records.
                prefix = bounded.read(1)
                if not prefix:
                    break
                class Prefix:
                    def __init__(self): self.prefix = prefix
                    def read(self, n=-1):
                        if self.prefix:
                            first, self.prefix = self.prefix, b''
                            return first if n == 1 else first+bounded.read(n-1 if n > 0 else -1)
                        return bounded.read(n)
                    def readline(self, n=-1):
                        if self.prefix:
                            first, self.prefix = self.prefix, b''
                            return first+bounded.readline(n-1 if n > 0 else -1)
                        return bounded.readline(n)
                record = _NumericUnpickler(Prefix()).load()
                _sample_valid(record, manifest)
                count += 1
                if count > samples:
                    raise ExchangeError('Trajectory contains extra samples')
    except (EOFError, OSError, pickle.UnpicklingError, AttributeError, TypeError, OverflowError) as error:
        raise ExchangeError('Incomplete or invalid trajectory') from error
    if count != samples:
        raise ExchangeError('Trajectory sample count mismatch')


def _file_inside(root, path):
    root = Path(root).resolve()
    path = Path(path)
    path = path if path.is_absolute() else root/path
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ExchangeError('Trajectory path escapes producer root or is not a regular file')
    # Reject parent symlinks even if they happen to resolve within the root.
    cursor = path.absolute()
    while cursor != root and cursor != cursor.parent:
        if cursor.is_symlink():
            raise ExchangeError('Symlink in producer trajectory path')
        cursor = cursor.parent
    return path.resolve()


def build_chunk(manifest, *, chunk_id, claim_token, rows, trajectory_root, output_path,
                limits=None):
    """Build a closed authenticated USTAR archive, without loading trajectories."""
    limits = limits or ExchangeLimits()
    manifest = _manifest(manifest)
    if chunk_id not in _chunks(manifest):
        raise ExchangeError('Unknown laptop chunk')
    if not isinstance(claim_token, str) or not re.fullmatch('[0-9a-f]{32}', claim_token):
        raise ExchangeError('Invalid claim token')
    rows = _rows(copy.deepcopy(rows), manifest, _chunks(manifest)[chunk_id])
    files, sources = {}, {}
    for row in rows:
        row['collection_host'] = 'laptop'
        for descriptor in row['trajectories']:
            source = _file_inside(trajectory_root, descriptor['path'])
            extension = '.pkl.gz' if str(source).endswith('.pkl.gz') else '.rlt' if source.suffix == '.rlt' else None
            if extension is None:
                raise ExchangeError('Unsupported trajectory format')
            player = descriptor.get('player')
            if player is None:
                match = re.search(r'-p([01])\.', source.name)
                if not match:
                    raise ExchangeError('Cannot determine trajectory player')
                player = int(match.group(1))
            name = f"trajectories/game-{row['index']:05d}-p{player}{extension}"
            if name in files:
                raise ExchangeError('Duplicate canonical trajectory path')
            size = source.stat().st_size
            if not 0 < size <= limits.max_member_bytes:
                raise ExchangeError('Trajectory member size limit exceeded')
            files[name] = dict(size=size, sha256=file_sha256(source), samples=descriptor['samples'])
            sources[name] = source
            descriptor['path'] = name
    body = dict(schema_version=1, producer='laptop', chunk_id=chunk_id, claim_token=claim_token,
                **{k:manifest[k] for k in ('run_id', 'round_id', 'policy_version', 'manifest_sha256',
                                          'weights_sha256', 'source_sha256', 'config_sha256')},
                rows=rows, files=files)
    payload = canonical_json(body)
    if len(payload) > limits.max_manifest_bytes or len(files)+1 > limits.max_files:
        raise ExchangeError('Chunk metadata/file count limit exceeded')
    if sum(f['size'] for f in files.values())+len(payload) > limits.max_unpacked_bytes:
        raise ExchangeError('Chunk unpacked size limit exceeded')
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.'+output.name+'-', dir=output.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            with tarfile.open(fileobj=stream, mode='w', format=tarfile.USTAR_FORMAT) as archive:
                for name in ['chunk.json']+sorted(files):
                    size = len(payload) if name == 'chunk.json' else files[name]['size']
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = size, 0o600, 0
                    if name == 'chunk.json':
                        archive.addfile(info, io.BytesIO(payload))
                    else:
                        with sources[name].open('rb') as source:
                            archive.addfile(info, source)
                        if file_sha256(sources[name]) != files[name]['sha256']:
                            raise ExchangeError('Trajectory changed while building chunk')
            stream.flush()
            os.fsync(stream.fileno())
        if Path(temporary).stat().st_size > limits.max_archive_bytes:
            raise ExchangeError('Archive size limit exceeded')
        # Atomic publication without accidentally overwriting a persisted spool.
        os.link(temporary, output)
        _fsync_dir(output.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return output


def _unpack(archive_path, staging, limits):
    """Parse only bounded regular USTAR members; never follow archive links."""
    path = Path(archive_path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limits.max_archive_bytes:
        raise ExchangeError('Invalid or oversized chunk archive')
    names, total = set(), 0
    with path.open('rb') as stream:
        while True:
            header = stream.read(512)
            if header == b'\0'*512:
                # Valid archives end in two blocks and optional zero padding.
                tail = stream.read()
                if len(tail) < 512 or any(tail):
                    raise ExchangeError('Invalid archive end marker')
                break
            if len(header) != 512:
                raise ExchangeError('Truncated archive header')
            try:
                info = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
            except (tarfile.HeaderError, UnicodeError) as error:
                raise ExchangeError('Invalid archive header') from error
            name = _safe_name(info.name)
            if info.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or info.linkname or info.sparse:
                raise ExchangeError('Only regular archive members are allowed')
            if name in names or len(names) >= limits.max_files:
                raise ExchangeError('Duplicate archive member or file count limit exceeded')
            cap = limits.max_manifest_bytes if name == 'chunk.json' else limits.max_member_bytes
            if not 0 <= info.size <= cap:
                raise ExchangeError('Archive member size limit exceeded')
            total += info.size
            if total > limits.max_unpacked_bytes:
                raise ExchangeError('Archive unpacked size limit exceeded')
            names.add(name)
            destination = Path(staging)/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as target:
                remaining = info.size
                while remaining:
                    block = stream.read(min(1024*1024, remaining))
                    if not block:
                        raise ExchangeError('Truncated archive member')
                    target.write(block)
                    remaining -= len(block)
                target.flush()
                os.fsync(target.fileno())
            padding = stream.read(-info.size % 512)
            if len(padding) != -info.size % 512 or any(padding):
                raise ExchangeError('Invalid archive member padding')
    if 'chunk.json' not in names:
        raise ExchangeError('Missing chunk metadata')
    return names


class RolloutExchange:
    def __init__(self, root, *, limits=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.limits = limits or ExchangeLimits()

    @contextmanager
    def _lock(self):
        with (self.root/'.exchange.lock').open('a+b') as stream:
            if os.name == 'nt':
                import msvcrt
                stream.seek(0)
                stream.write(b'0')
                stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == 'nt':
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _round(self, round_id):
        _integer(round_id, 'round ID')
        return self.root/f'round-{round_id:06d}'

    def _load(self, round_id):
        directory = self._round(round_id)
        manifest = _manifest(_read_json(directory/'manifest.json'))
        state = _read_json(directory/'state.json')
        if state.get('manifest_sha256') != manifest['manifest_sha256']:
            raise ExchangeError('Round state is bound to another manifest')
        self._reconcile(directory, manifest, state)
        return directory, manifest, state

    def _reconcile(self, directory, manifest, state):
        """Recover the rename-before-ledger crash window from committed receipts."""
        accepted = directory/'accepted'
        changed = False
        for child in sorted(accepted.iterdir()) if accepted.exists() else []:
            if child.is_symlink() or not child.is_dir() or child.name not in _chunks(manifest):
                raise ExchangeError('Unexpected committed chunk directory')
            receipt = _read_json(child/'receipt.json')
            if receipt.get('manifest_sha256') != manifest['manifest_sha256'] or receipt.get('chunk_id') != child.name:
                raise ExchangeError('Committed chunk receipt identity differs')
            if state['accepted'].get(child.name) not in (None, receipt):
                raise ExchangeError('Committed chunk conflicts with durable ledger')
            if child.name not in state['accepted']:
                state['accepted'][child.name] = receipt
                changed = True
        if any(not (accepted/name/'receipt.json').is_file() for name in state['accepted']):
            raise ExchangeError('Durable ledger references missing committed chunk')
        if changed:
            _atomic_json(directory/'state.json', state)

    def open_round(self, manifest):
        manifest = _manifest(manifest)
        with self._lock():
            directory = self._round(manifest['round_id'])
            if (directory/'manifest.json').exists():
                # A power loss can leave only the immutable manifest, before
                # the first state commit. No lease can have been issued yet.
                if not (directory/'state.json').exists():
                    old = _manifest(_read_json(directory/'manifest.json'))
                    if old != manifest or (directory/'accepted').exists():
                        raise ExchangeError('Incomplete round initialization conflicts')
                    _atomic_json(directory/'state.json', dict(manifest_sha256=manifest['manifest_sha256'],
                        sealed=False, complete=False, claims={}, accepted={}))
                _, old, state = self._load(manifest['round_id'])
                if old != manifest:
                    raise ExchangeError('Frozen round manifest conflicts with existing round')
            else:
                current = self.root/'current.json'
                if current.exists():
                    _, previous, state = self._load(_read_json(current)['round_id'])
                    if previous['run_id'] != manifest['run_id'] or not state.get('complete'):
                        raise ExchangeError('Cannot replace an unfinished current round')
                    if manifest['round_id'] <= previous['round_id']:
                        raise ExchangeError('Cannot move current round backwards')
                    used_indices, used_seeds = set(), set()
                    for saved in self.root.glob('round-*/manifest.json'):
                        prior = _manifest(_read_json(saved))
                        if prior['run_id'] != manifest['run_id']:
                            raise ExchangeError('Exchange contains a different run')
                        used_indices.update(g['index'] for g in prior['games'])
                        used_seeds.update(g['seed'] for g in prior['games'])
                    if (used_indices.intersection(g['index'] for g in manifest['games'])
                            or used_seeds.intersection(g['seed'] for g in manifest['games'])):
                        raise ExchangeError('Round reuses a global game index or seed')
                directory.mkdir(parents=True, exist_ok=True)
                _atomic_json(directory/'manifest.json', manifest)
                _atomic_json(directory/'state.json', dict(manifest_sha256=manifest['manifest_sha256'],
                    sealed=False, complete=False, claims={}, accepted={}))
            current = self.root/'current.json'
            if current.exists() and _read_json(current)['round_id'] > manifest['round_id']:
                raise ExchangeError('Cannot reactivate a stale round')
            _atomic_json(current, dict(round_id=manifest['round_id'], run_id=manifest['run_id'],
                                       manifest_sha256=manifest['manifest_sha256']))
        return manifest

    def status(self, round_id=None):
        with self._lock():
            if round_id is None:
                if not (self.root/'current.json').exists():
                    return dict(status='waiting', manifest=None)
                round_id = _read_json(self.root/'current.json')['round_id']
            _, manifest, state = self._load(round_id)
            remaining = [g['index'] for name, games in _chunks(manifest).items()
                         if name not in state['accepted'] for g in games]
            return dict(status='closed' if state['sealed'] else 'work', manifest=manifest,
                        round_id=round_id, sealed=state['sealed'], complete=state['complete'],
                        accepted_chunks=sorted(state['accepted']), remaining_laptop_indices=remaining)

    get_current = status

    def claim(self, round_id, producer='laptop', identity=None):
        with self._lock():
            directory, manifest, state = self._load(round_id)
            current = _read_json(self.root/'current.json')
            if current['round_id'] != round_id or state['sealed']:
                return None
            if producer != 'laptop' or identity != manifest.get('identity'):
                raise ExchangeError('Producer identity differs from frozen round')
            for name, games in _chunks(manifest).items():
                if name in state['accepted'] or name in state['claims']:
                    continue
                lease = dict(round_id=round_id, manifest_sha256=manifest['manifest_sha256'],
                             chunk_id=name, claim_token=uuid.uuid4().hex, producer=producer,
                             games=games)
                state['claims'][name] = lease
                _atomic_json(directory/'state.json', state)
                return copy.deepcopy(lease)
            return None

    def accept_chunk(self, archive_path, round_id=None):
        incoming = self.root/'.incoming'
        incoming.mkdir(exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='chunk-', dir=incoming))
        try:
            names = _unpack(archive_path, staging, self.limits)
            chunk = _read_json(staging/'chunk.json', self.limits.max_manifest_bytes)
            payload_digest = json_sha256(chunk)
            actual_round = _integer(chunk.get('round_id'), 'chunk round ID')
            if round_id is not None and round_id != actual_round:
                raise ExchangeError('Uploaded chunk is for another round')
            with self._lock():
                directory, manifest, state = self._load(actual_round)
                for field in ('run_id', 'round_id', 'policy_version', 'manifest_sha256',
                              'weights_sha256', 'source_sha256', 'config_sha256'):
                    if chunk.get(field) != manifest[field]:
                        raise ExchangeError(f'Stale or incompatible chunk {field}')
                name = chunk.get('chunk_id')
                if chunk.get('schema_version') != 1 or chunk.get('producer') != 'laptop' or name not in _chunks(manifest):
                    raise ExchangeError('Invalid chunk identity')
                lease = state['claims'].get(name)
                if lease is None or chunk.get('claim_token') != lease['claim_token']:
                    raise ExchangeError('Chunk does not have an issued claim')
                rows = _rows(chunk.get('rows'), manifest, _chunks(manifest)[name])
                files = chunk.get('files')
                if not isinstance(files, dict) or names != {'chunk.json', *files}:
                    raise ExchangeError('Archive members differ from chunk metadata')
                descriptors = {d['path']:d for r in rows for d in r['trajectories']}
                if set(files) != set(descriptors):
                    raise ExchangeError('Archive files differ from trajectory descriptors')
                for row in rows:
                    for descriptor in row['trajectories']:
                        member = descriptor['path']
                        match = re.fullmatch(r'trajectories/game-(\d+)-p([01])\.(pkl\.gz|rlt)', member)
                        if (not match or int(match.group(1)) != row['index']
                                or (descriptor.get('player') is not None
                                    and int(match.group(2)) != descriptor['player'])):
                            raise ExchangeError('Trajectory member belongs to another game or seat')
                for member, metadata in files.items():
                    _safe_name(member)
                    if not re.fullmatch(r'trajectories/game-\d+-p[01]\.(pkl\.gz|rlt)', member):
                        raise ExchangeError('Unexpected trajectory member path')
                    path = staging/member
                    if (not isinstance(metadata, dict) or metadata.get('size') != path.stat().st_size
                            or metadata.get('sha256') != file_sha256(path)
                            or metadata.get('samples') != descriptors[member]['samples']):
                        raise ExchangeError('Trajectory bytes, checksum or sample declaration differ')
                    _digest(metadata['sha256'], 'trajectory')
                prior = state['accepted'].get(name)
                if prior is not None:
                    if prior['payload_sha256'] != payload_digest:
                        raise ExchangeError('Conflicting retry for an accepted chunk')
                    return dict(status='duplicate', **prior)
                if state['sealed']:
                    return dict(status='sealed', chunk_id=name, round_id=actual_round)
                if _read_json(self.root/'current.json')['round_id'] != actual_round:
                    raise ExchangeError('Chunk is for a stale round')
                # Identity/lease and ALL member byte hashes are checked before
                # any numeric deserialization occurs.
                for member, metadata in files.items():
                    _validate_trajectory(staging/member, metadata['samples'], manifest, self.limits)
                destination = directory/'accepted'/name
                canonical = copy.deepcopy(rows)
                for row in canonical:
                    row['collection_host'] = 'laptop'
                    for descriptor in row['trajectories']:
                        descriptor['path'] = str(destination/descriptor['path'])
                receipt = dict(chunk_id=name, round_id=actual_round,
                    manifest_sha256=manifest['manifest_sha256'], payload_sha256=payload_digest,
                    game_indices=[r['index'] for r in canonical], rows_sha256=json_sha256(canonical))
                _atomic_json(staging/'rows.json', canonical)
                _atomic_json(staging/'receipt.json', receipt)
                for child in sorted(staging.rglob('*'), reverse=True):
                    if child.is_dir():
                        _fsync_dir(child)
                _fsync_dir(staging)
                destination.parent.mkdir(exist_ok=True)
                os.rename(staging, destination)
                _fsync_dir(destination.parent)
                _fsync_dir(directory)
                state['accepted'][name] = receipt
                _atomic_json(directory/'state.json', state)
                return dict(status='accepted', **receipt)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    accept_archive = accept_chunk

    def _accepted_rows(self, directory, state, *, verify_files=False):
        result = []
        for name, receipt in sorted(state['accepted'].items()):
            rows = _read_json(directory/'accepted'/name/'rows.json', self.limits.max_manifest_bytes)
            if json_sha256(rows) != receipt['rows_sha256']:
                raise ExchangeError('Canonical accepted rows changed')
            if verify_files:
                chunk = _read_json(directory/'accepted'/name/'chunk.json', self.limits.max_manifest_bytes)
                if json_sha256(chunk) != receipt['payload_sha256']:
                    raise ExchangeError('Committed chunk metadata changed')
                for member, metadata in chunk['files'].items():
                    path = directory/'accepted'/name/member
                    if (path.is_symlink() or not path.is_file() or path.stat().st_size != metadata['size']
                            or file_sha256(path) != metadata['sha256']):
                        raise ExchangeError('Committed trajectory file changed')
            result.extend(rows)
        return sorted(result, key=lambda row: row['index'])

    def accepted_rows(self, round_id):
        with self._lock():
            directory, _, state = self._load(round_id)
            return self._accepted_rows(directory, state)

    def remaining_laptop_indices(self, round_id):
        return self.status(round_id)['remaining_laptop_indices']

    def seal(self, round_id):
        with self._lock():
            directory, manifest, state = self._load(round_id)
            state['sealed'] = True
            _atomic_json(directory/'state.json', state)
            return [g['index'] for name, games in _chunks(manifest).items()
                    if name not in state['accepted'] for g in games]

    close_round = seal

    def complete_round(self, round_id, rows):
        with self._lock():
            directory, manifest, state = self._load(round_id)
            rows = _rows(copy.deepcopy(rows), manifest, manifest['games'])
            by_index = {r['index']:r for r in rows}
            for row in self._accepted_rows(directory, state, verify_files=True):
                if row != by_index[row['index']]:
                    raise ExchangeError('Accepted laptop row changed during central merge')
            digest = json_sha256(rows)
            if state['complete']:
                if state.get('completed_rows_sha256') != digest:
                    raise ExchangeError('Completed round conflicts with retry')
                return dict(status='duplicate', round_id=round_id, rows_sha256=digest)
            _atomic_json(directory/'completed-rows.json', rows)
            state.update(complete=True, sealed=True, completed_rows_sha256=digest)
            _atomic_json(directory/'state.json', state)
            return dict(status='complete', round_id=round_id, rows_sha256=digest)

    def completed_rows(self, round_id):
        with self._lock():
            directory, _, state = self._load(round_id)
            if not state['complete']:
                return None
            rows = _read_json(directory/'completed-rows.json', 100 * self.limits.max_manifest_bytes)
            if json_sha256(rows) != state.get('completed_rows_sha256'):
                raise ExchangeError('Completed round rows changed')
            self._accepted_rows(directory, state, verify_files=True)
            return rows
