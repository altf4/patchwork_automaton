"""Four pinned, restartable 10K evaluation panels for the completed A/B agents."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def value_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def verify(config):
    for name, digest in config['preserved_files'].items():
        if sha(name) != digest:
            raise ValueError('Preserved artifact changed: ' + name)


def schedule(panel, config, decks):
    rows = []
    for cell, (learner, opponent) in enumerate(product(decks, repeat=2)):
        repeats = 100 if panel['kind'] == 'neural' else (1108 if cell in (0, 4) else 1112)
        for repeat in range(repeats):
            seat = repeat % 2
            seating = [opponent, opponent]
            seating[seat] = learner
            offset = len(rows)
            row = dict(index=panel['offset'] + offset, seed=config['seeds'][panel['kind']] + offset,
                pairing=sorted((learner, opponent), key=decks.index), decks=seating,
                learner_seat=seat, learner_deck=learner, opponent_deck=opponent,
                starting_player=f'P{(repeat // 2) % 2}', partition='evaluation',
                deck_catalog='hartford' if panel['kind'] == 'neural' else 'standard',
                behavior_version=150, opponent_kind='frozen_neural' if panel['kind'] == 'neural' else 'scripted')
            if panel['kind'] == 'neural':
                row.update(opponent_version=100, opponent_checkpoint_sha256=config['models']['reference']['sha256'])
            rows.append(row)
    random.Random('post-ab:' + panel['kind']).shuffle(rows)
    if len(rows) != 10000 or Counter((r['learner_seat'], r['starting_player']) for r in rows) != {
            (seat, f'P{start}'): 2500 for seat in (0, 1) for start in (0, 1)}:
        raise ValueError('Wrong panel size or seat/start balance')
    return rows


def helpers(config):
    specification = importlib.util.spec_from_file_location('validated_endpoint', config['helper_script'])
    result = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(result)
    return result


def load_handoffs(folder, expected, shared):
    from patchwork_automaton.rl.hartford_evaluation import validate_row
    rows = {}
    for path in sorted((folder / 'handoffs').glob('*.json')):
        envelope = read(path)
        row = envelope['row']
        index = row.get('index')
        if (set(envelope) != {'row', 'sha256'} or value_sha(row) != envelope['sha256']
                or index not in expected or path.name != f'game-{index}.json' or index in rows):
            raise ValueError('Invalid completed-result handoff')
        validate_row(row, expected[index], checkpoint_round=150)
        row = dict(row, actions=None)
        if row.get('adjudicated'):
            row['diagnostic_hashes'] = shared.diagnostic_hashes(folder, row)
        rows[index] = row
    return rows


async def run_panel(config, panel):
    from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS, STANDARD_DECKS
    from patchwork_automaton.neural.devices import configure_runtime, device_scope
    from patchwork_automaton.neural.standard_policy import StandardPolicy
    from patchwork_automaton.neural.inference_processes import weights_digest
    from patchwork_automaton.rl.hartford_evaluation import atomic_json, validate_row, summarize
    from patchwork_automaton.rl.standard_selfplay_processes import StandardSelfplayCollector

    verify(config)
    shared = helpers(config)
    folder = Path(config['output']) / panel['name']
    folder.mkdir(parents=True, exist_ok=True)
    decks = tuple(HARTFORD_DECKS if panel['kind'] == 'neural' else STANDARD_DECKS)
    planned = schedule(panel, config, decks)
    expected = {r['index']: r for r in planned}
    with (folder / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = dict(panel=panel, config_sha256=sha(Path(config['output']) / 'config.json'),
            schedule_sha256=value_sha(planned), models=config['models'], training=False)
        if (folder / 'manifest.json').exists():
            if read(folder / 'manifest.json') != manifest or read(folder / 'schedule.json') != planned:
                raise ValueError('Panel inputs changed on resume')
        else:
            atomic_json(folder / 'manifest.json', manifest)
            atomic_json(folder / 'schedule.json', planned)
        rows = shared.load_results(folder, planned)
        handoffs = load_handoffs(folder, expected, shared)
        for index, row in handoffs.items():
            if index not in rows:
                atomic_json(folder / 'games' / f'game-{index}.json', dict(row=row, sha256=value_sha(row)))
                rows[index] = row
        remaining = [r for r in planned if r['index'] not in rows]
        loop = asyncio.get_running_loop()
        def tick():
            if not loop.is_closed():
                loop.call_later(.002, tick)
        loop.call_later(.002, tick)
        collectors = []
        started = time.monotonic()
        transport_error = None
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')

        def persist(row):
            row = dict(row, actions=None)
            index = row.get('index')
            try:
                if index in rows or index not in expected:
                    raise ValueError('Duplicate or foreign game completion')
                validate_row(row, expected[index], checkpoint_round=150)
                if row.get('adjudicated'):
                    row['diagnostic_hashes'] = shared.diagnostic_hashes(folder, row)
            except BaseException:
                atomic_json(folder / 'failures' / f'game-{index}.json', row)
                raise
            atomic_json(folder / 'games' / f'game-{index}.json', dict(row=row, sha256=value_sha(row)))
            rows[index] = row
            if len(rows) % 100 == 0:
                progress = dict(panel=panel['name'], completed=len(rows), total=10000,
                    elapsed_seconds=time.monotonic() - started, status='running')
                atomic_json(Path(config['output']) / 'progress.json', progress)
                print(json.dumps(progress), flush=True)

        try:
            if remaining:
                configure_runtime('cpu', hide_gpu=True)
                with device_scope('cpu'):
                    focal = StandardPolicy.load(config['models'][panel['model']]['path']).model
                    opponent_key = 'reference' if panel['kind'] == 'neural' else panel['model']
                    opponent = StandardPolicy.load(config['models'][opponent_key]['path']).model
                digests = [weights_digest(m.get_weights()) for m in (focal, opponent)]
                collectors = [StandardSelfplayCollector(focal, opponent, workers=24,
                    inference_processes=1, opponent_inference_processes=1) for _ in range(2)]
                await shared.settle(c.start(version=150, opponent_version=100) for c in collectors)
                if any([c.policy_digest, c.opponent_digest] != digests for c in collectors):
                    raise ValueError('Inference replica weights differ')
                jobs = [dict(index=r['index'], seed=r['seed'], game=r) for r in remaining]
                try:
                    await shared.settle(c.run_schedule(jobs[i::2], progress=persist,
                        game_runner='standard_ppo', version=150, opponent_version=100,
                        opponent_sha256=config['models'][opponent_key]['sha256'], output=None,
                        diagnostics=folder / 'diagnostics', completed_result_directory=folder / 'handoffs',
                        greedy=False, game_timeout=panel['game_timeout'], max_turns=50, max_decisions=6000)
                        for i, c in enumerate(collectors))
                except TimeoutError as error:
                    if str(error) != 'Inference reset timed out':
                        raise
                    transport_error = dict(error=str(error), traceback=traceback.format_exc())
                    atomic_json(folder / 'transport-errors' / (stamp + '.json'), transport_error)
                if [weights_digest(m.get_weights()) for m in (focal, opponent)] != digests:
                    raise ValueError('Evaluation changed model weights')
        finally:
            await shared.settle(c.close() for c in collectors)
        handoffs = load_handoffs(folder, expected, shared)
        recovered = sorted(set(handoffs) - set(rows))
        for index in recovered:
            persist(handoffs[index])
        if set(rows) != set(expected):
            raise ValueError('Incomplete evaluation outcomes')
        if transport_error:
            atomic_json(folder / 'transport-errors' / (stamp + '-resolution.json'),
                dict(resolved=True, recovered_completed_handoffs=recovered, games_replayed=0,
                     all_games_validated=True, collectors_closed=True))
        verify(config)
        ordered = [rows[index] for index in sorted(rows)]
        result = dict(status='complete', completed_at=datetime.now(timezone.utc).isoformat(),
            panel=panel, games=10000, summary=summarize(ordered, decks), rows_sha256=value_sha(ordered),
            recovered_completed_handoffs=recovered, transport_cleanup_error=transport_error,
            training=False, seconds=time.monotonic() - started)
        atomic_json(folder / 'summary.json', result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--panel')
    args = parser.parse_args()
    config = read(args.config)
    if args.panel:
        panel = next(p for p in config['panels'] if p['name'] == args.panel)
        asyncio.run(run_panel(config, panel))
        return
    from patchwork_automaton.rl.hartford_evaluation import atomic_json
    for panel in config['panels']:
        verify(config)
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--config', str(args.config),
            '--panel', panel['name']], check=True)
    verify(config)
    atomic_json(Path(config['output']) / 'progress.json',
        dict(status='complete', completed=40000, total=40000, panels=4))


if __name__ == '__main__':
    main()
