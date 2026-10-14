"""Keep an authenticated session alive without choosing any game actions."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import time

from patchwork_automaton import AuthenticationError, MTGOClient, ProtocolError
from patchwork_automaton import __version__


def credentials(path):
    values = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        separators = [(line.index(s), s) for s in ('=', ':') if s in line]
        if not separators:
            raise ValueError('Invalid credential file')
        key, value = line.split(min(separators)[1], 1)
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        values[key.strip().lower()] = value
    return values['username'], values['password']


async def run(args):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    device = None
    if args.run_dir:
        args.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = args.run_dir / 'device-id'
        if path.exists():
            device = path.read_text().strip()
    async with MTGOClient(device_id=device) as client:
        await client.login(*credentials(args.credentials))
        if args.run_dir and not (args.run_dir / 'device-id').exists():
            path = args.run_dir / 'device-id'
            path.write_text(client.device_id)
            path.chmod(0o600)
        # Read-only smoke checks. Do not include account/deck content in logs.
        decks = await client.list_decks()
        matches = await client.current_matches()
        while client.connected and not stop.is_set():
            status = dict(pid=os.getpid(), connected=True, phase='authenticated',
                          library_version=__version__, updated_at=time.time(),
                          decks_received=len(decks.decks), current_matches=len(matches),
                          heartbeats_answered=client.heartbeats_answered)
            if args.run_dir:
                temporary = args.run_dir / 'status.tmp'
                temporary.write_text(json.dumps(status, indent=2))
                temporary.chmod(0o600)
                temporary.replace(args.run_dir / 'status.json')
            print(json.dumps(status), flush=True)
            try:
                await asyncio.wait_for(stop.wait(), 30)
            except TimeoutError:
                pass
        if not client.connected:
            raise ConnectionError('Session disconnected')
    if args.run_dir:
        status.update(connected=False, phase='stopped', updated_at=time.time())
        (args.run_dir / 'status.json').write_text(json.dumps(status, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except (AuthenticationError, ProtocolError, ValueError, KeyError) as error:
        if args.run_dir:
            (args.run_dir / 'status.json').write_text(json.dumps(dict(
                connected=False, phase='failed', error_type=type(error).__name__, updated_at=time.time())))
        print('Authentication or protocol failure; no automatic retry.', flush=True)
        return 2
    except (OSError, TimeoutError) as error:
        if args.run_dir:
            (args.run_dir / 'status.json').write_text(json.dumps(dict(
                connected=False, phase='failed', error_type=type(error).__name__, updated_at=time.time())))
        print('Transport failure.', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
