"""Read-only, restartable Standard checkpoint evaluation against fixed scripts.

Run one checkpoint and one opponent style into a new output directory. Reuse
the same seed base for all checkpoints in a comparison, and never use the
held-out tactical script for PPO collection.
"""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('PATCHWORK_OBSERVATION_PIPELINE', 'python')
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from patchwork_automaton.baselines.standard_decks import STANDARD_DECKS


MODES = ('sampled', 'greedy')
OPPONENTS = ('tactical', 'aggro')
TACTICAL_SEED = 902611000000
SCRIPT_SEED = 802610300000  # Historical StandardAggro evaluation seed.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def cells():
    return [(deck, opponent, mode)
            for deck in STANDARD_DECKS for opponent in STANDARD_DECKS for mode in MODES]


def cell_name(deck, opponent, mode):
    return f'{deck}-vs-{opponent}-{mode}'


def scheduled_games(deck, opponent, mode, *, games_per_cell=400, seed_base=TACTICAL_SEED):
    if deck not in STANDARD_DECKS or opponent not in STANDARD_DECKS or mode not in MODES:
        raise ValueError('Unknown Standard evaluation cell')
    if games_per_cell < 4 or games_per_cell % 4:
        raise ValueError('Games per cell must balance both seats and starting players')
    cell_id = cells().index((deck, opponent, mode))
    for i in range(games_per_cell):
        # A cell-local seed deliberately recurs across policies. The index is
        # globally unique across cells for diagnostics and immutable rows.
        index = cell_id * games_per_cell + i
        seat = i % 2
        decks = [opponent, opponent]
        decks[seat] = deck
        game = dict(index=index, seed=seed_base + i,
                    pairing=sorted((deck, opponent), key=list(STANDARD_DECKS).index),
                    decks=decks, learner_seat=seat, learner_deck=deck,
                    opponent_deck=opponent, starting_player=f'P{(i // 2) % 2}',
                    partition='evaluation')
        yield dict(index=index, seed=seed_base + i, game=game)


def expected_game(job, deck, opponent):
    game = job['game']
    if game['learner_deck'] != deck or game['opponent_deck'] != opponent:
        raise ValueError('Scheduled deck identity differs')
    return game


def validate_row(row, job, deck, opponent):
    expected = expected_game(job, deck, opponent)
    if not isinstance(row, dict) or any(row.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Evaluation identity mismatch for {job["index"]}')
    if row.get('trajectories') != []:
        raise ValueError('Evaluation wrote a PPO trajectory')
    if row.get('status') not in ('win', 'loss', 'draw'):
        raise RuntimeError(f'Game {job["index"]} failed: {row.get("error", row.get("status"))}')
    if row['status'] == 'draw':
        if row.get('winner_seat') is not None or row.get('draw') is not True:
            raise ValueError('Inconsistent draw')
        if row.get('learner_won') is not False:
            raise ValueError('Draw marked learner win')
    else:
        if row.get('winner_seat') not in (0, 1) or row.get('draw') is not False:
            raise ValueError('Inconsistent decisive outcome')
        if row.get('learner_won') is not (row['winner_seat'] == expected['learner_seat']):
            raise ValueError('Learner outcome disagrees with winner')
    return row


def summarize(rows):
    result = dict(games=len(rows), wins=0, losses=0, draws=0,
                  on_play=dict(games=0, wins=0, losses=0, draws=0),
                  on_draw=dict(games=0, wins=0, losses=0, draws=0),
                  draw_reasons=dict(Counter(r.get('reason') for r in rows if r['status'] == 'draw')))
    for row in rows:
        side = 'on_play' if row['starting_player'] == f'P{row["learner_seat"]}' else 'on_draw'
        outcome = 'draws' if row['status'] == 'draw' else 'wins' if row['learner_won'] else 'losses'
        result[outcome] += 1
        result[side]['games'] += 1
        result[side][outcome] += 1
    for subset in (result, result['on_play'], result['on_draw']):
        subset['win_rate'] = subset['wins'] / subset['games'] if subset['games'] else None
        subset['score'] = (subset['wins'] + .5 * subset['draws']) / subset['games'] if subset['games'] else None
    return result


def read_rows(path, jobs, deck, opponent):
    allowed = {job['index']: job for job in jobs}
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            index = row.get('index')
            if index not in allowed or index in rows:
                raise ValueError('Unexpected or duplicate durable game result')
            rows[index] = validate_row(row, allowed[index], deck, opponent)
    return rows


def manifest(args, checkpoint):
    from patchwork_automaton.baselines import standard, standard_tactical
    from patchwork_automaton.rl.standard_imitation import playback_mode
    source = Path(standard.__file__)
    tactical_source = Path(standard_tactical.__file__)
    mode = playback_mode()
    direct_hash = None
    if mode == 'direct':
        from patchwork_automaton.neural.direct_standard import library_path
        direct_hash = sha256(library_path())
    return dict(schema=1, checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
                label=args.label, opponent=args.opponent, games_per_cell=args.games_per_cell,
                seed_base=args.seed_base, game_timeout=args.game_timeout, max_turns=args.max_turns,
                workers=args.workers, inference_processes=args.inference_processes,
                playback_pipeline=mode, direct_library_sha256=direct_hash,
                standard_aggro_sha256=sha256(source), tactical_sha256=sha256(tactical_source),
                cells=[cell_name(*cell) for cell in cells()])


async def run(args):
    from patchwork_automaton.baselines.standard_tactical import StandardTactical
    from patchwork_automaton.neural.devices import configure_runtime, device_scope
    from patchwork_automaton.neural.standard_policy import StandardPolicy
    from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector

    checkpoint = args.checkpoint / 'policy.keras' if args.checkpoint.is_dir() else args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = manifest(args, checkpoint)
    output = args.output.resolve()
    manifest_path = output / 'manifest.json'
    if output.exists():
        if not args.resume or not manifest_path.is_file():
            raise FileExistsError('Existing evaluation output requires explicit --resume and a manifest')
        if json.loads(manifest_path.read_text()) != config:
            raise ValueError('Evaluation configuration or source/checkpoint hash changed')
    else:
        output.mkdir(parents=True)
        atomic_json(manifest_path, config)

    print(json.dumps(dict(event='load_checkpoint', checkpoint=args.label)), flush=True)
    configure_runtime('cpu', hide_gpu=True)
    with device_scope('cpu'):
        model = StandardPolicy.load(checkpoint).model
    print(json.dumps(dict(event='checkpoint_loaded', checkpoint=args.label)), flush=True)
    collector = ProcessGameCollector(model, workers=args.workers,
                                     inference_processes=args.inference_processes,
                                     inference_device='cpu')
    factory = StandardTactical if args.opponent == 'tactical' else None
    completed = {}
    try:
        await collector.start(version=0)
        print(json.dumps(dict(event='inference_started', checkpoint=args.label)), flush=True)
        for deck, opponent, mode in cells():
            key = cell_name(deck, opponent, mode)
            directory = output / 'cells' / key
            directory.mkdir(parents=True, exist_ok=True)
            jobs = list(scheduled_games(deck, opponent, mode,
                                        games_per_cell=args.games_per_cell,
                                        seed_base=args.seed_base))
            rows_path = directory / 'games.jsonl'
            rows = read_rows(rows_path, jobs, deck, opponent)
            pending = [job for job in jobs if job['index'] not in rows]
            atomic_json(output / 'progress.json', dict(cell=key, completed=len(rows),
                        total=args.games_per_cell, checkpoint=args.label,
                        timestamp=datetime.now(timezone.utc).isoformat()))
            if pending:
                with rows_path.open('a') as stream:
                    def persist(row):
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                        index = row.get('index')
                        if type(index) is not int or index < jobs[0]['index'] or index > jobs[-1]['index']:
                            raise ValueError('Unexpected evaluation result ID')
                        job = jobs[index - jobs[0]['index']]
                        validate_row(row, job, deck, opponent)
                        if index in rows:
                            raise ValueError('Duplicate evaluation game result')
                        rows[index] = row
                        if len(rows) % 25 == 0:
                            atomic_json(output / 'progress.json', dict(cell=key,
                                        completed=len(rows), total=args.games_per_cell,
                                        checkpoint=args.label,
                                        timestamp=datetime.now(timezone.utc).isoformat()))
                    await collector.run_schedule(pending, progress=persist,
                        game_runner='standard_ppo',
                        opponent_policy_factory=factory, greedy=mode == 'greedy',
                        diagnostics=directory / 'diagnostics',
                        game_timeout=args.game_timeout, max_turns=args.max_turns)
            if len(rows) != len(jobs):
                raise RuntimeError(f'Incomplete cell {key}: {len(rows)} / {len(jobs)}')
            ordered = [rows[job['index']] for job in jobs]
            cell_summary = summarize(ordered)
            completed[key] = cell_summary
            atomic_json(directory / 'summary.json', cell_summary)
            atomic_json(output / 'summary.json', dict(complete=False, cells=completed,
                        checkpoint=args.label, opponent=args.opponent))
            print(json.dumps(dict(event='cell_complete', cell=key,
                                  games=cell_summary['games'],
                                  win_rate=cell_summary['win_rate'])), flush=True)
        if sha256(checkpoint) != config['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed during read-only evaluation')
        if manifest(args, checkpoint) != config:
            raise RuntimeError('Playback configuration or source changed during evaluation')
        aggregate = {mode: summarize([row for deck, opponent, cell_mode in cells()
                     if cell_mode == mode for row in read_rows(
                         output / 'cells' / cell_name(deck, opponent, mode) / 'games.jsonl',
                         list(scheduled_games(deck, opponent, mode,
                             games_per_cell=args.games_per_cell, seed_base=args.seed_base)),
                         deck, opponent).values()]) for mode in MODES}
        atomic_json(output / 'summary.json', dict(complete=True, cells=completed,
                    aggregate=aggregate, checkpoint=args.label, opponent=args.opponent,
                    checkpoint_sha256=config['checkpoint_sha256']))
    finally:
        await collector.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--opponent', choices=OPPONENTS, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--games-per-cell', type=int, default=400)
    p.add_argument('--seed-base', type=int)
    p.add_argument('--workers', type=int, default=48)
    p.add_argument('--inference-processes', type=int, default=4)
    p.add_argument('--game-timeout', type=float, default=60.)
    p.add_argument('--max-turns', type=int, default=50)
    p.add_argument('--resume', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.seed_base is None:
        args.seed_base = TACTICAL_SEED if args.opponent == 'tactical' else SCRIPT_SEED
    if args.games_per_cell < 4 or args.games_per_cell % 4:
        raise ValueError('Games per cell must be a positive multiple of four')
    if args.workers < 1 or not 1 <= args.inference_processes <= args.workers:
        raise ValueError('Invalid worker or inference-process count')
    if args.game_timeout <= 0 or args.max_turns < 1:
        raise ValueError('Positive game limits required')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
