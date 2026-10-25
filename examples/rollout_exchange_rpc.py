"""Narrow SSH stdin/stdout transport for the versioned rollout exchange."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile

from patchwork_automaton.neural.rollout_exchange import RolloutExchange


def read_request():
    raw = sys.stdin.buffer.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError('Oversized RPC request')
    value = json.loads(raw or b'{}')
    if not isinstance(value, dict):
        raise ValueError('RPC request must be an object')
    return value


def unwrap(value):
    if not value:
        return None
    return value.get('manifest', value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('action', choices=('status', 'snapshot', 'upload'))
    parser.add_argument('--round', type=int)
    parser.add_argument('--chunk')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    exchange = RolloutExchange(root)
    if args.action == 'status':
        request = read_request()
        if (root / 'producer-stop.json').exists():
            print(json.dumps(dict(status='closed', reason=json.loads((root/'producer-stop.json').read_text()))))
            return
        current = exchange.get_current()
        manifest = unwrap(current)
        if not manifest:
            print(json.dumps(dict(status='waiting')))
            return
        lease = exchange.claim(manifest['round_id'], producer=request.get('producer', 'laptop'),
                               identity=request.get('identity'))
        print(json.dumps(dict(status='work', manifest=manifest, lease=lease) if lease
                         else dict(status='waiting', round_id=manifest['round_id'])))
    elif args.action == 'snapshot':
        request = read_request()
        round_id = args.round if args.round is not None else request['round_id']
        manifest = unwrap(exchange.status(round_id))
        if not manifest or manifest['round_id'] != round_id:
            raise ValueError('Unknown snapshot version')
        with tarfile.open(fileobj=sys.stdout.buffer, mode='w|', format=tarfile.USTAR_FORMAT) as archive:
            for name, item in sorted(manifest['snapshot'].items()):
                if Path(name).name != name or name in ('', '.', '..'):
                    raise ValueError('Invalid snapshot filename')
                path = Path(item['path'])
                if not path.is_file() or path.is_symlink():
                    raise ValueError('Missing frozen snapshot')
                if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                    raise ValueError('Frozen snapshot changed')
                archive.add(path, arcname=name, recursive=False)
    else:
        incoming = root / 'incoming'
        incoming.mkdir(parents=True, exist_ok=True)
        handle, filename = tempfile.mkstemp(prefix='rpc-', suffix='.tar', dir=incoming)
        try:
            count = 0
            with os.fdopen(handle, 'wb') as stream:
                while block := sys.stdin.buffer.read(1024 * 1024):
                    count += len(block)
                    if count > 256 * 1024 * 1024:
                        raise ValueError('Oversized rollout upload')
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            result = exchange.accept_chunk(filename, round_id=args.round)
            print(json.dumps(result))
        finally:
            Path(filename).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
