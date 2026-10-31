"""Cost-bounded, read-only arena for the three verified Standard decks.

The default panel is three same-deck matches against StandardTactical, with
greedy neural play and 64 games per cell. Use --plan to inspect the game count
before creating an output directory. Each run evaluates one frozen learner
checkpoint; repeating the schedule for another checkpoint gives paired cells.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('PATCHWORK_OBSERVATION_PIPELINE', 'python')
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from patchwork_automaton.backends.rustmage import OBSERVATION_SCHEMA, RUSTMAGE_REVISION
from patchwork_automaton.baselines.standard_decks import STANDARD_DECKS
from patchwork_automaton.rl.standard_arena_report import summarize_arena


OPPONENTS = ('aggro', 'tactical', 'checkpoint')
PANELS = ('mirrors', 'matrix')
DEFAULT_SEED = 902611000000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    'examples/evaluate_standard_arena.py',
    'src/patchwork_automaton/backends/rustmage.py',
    'src/patchwork_automaton/baselines/standard.py',
    'src/patchwork_automaton/baselines/standard_tactical.py',
    'src/patchwork_automaton/baselines/standard_decks.py',
    'src/patchwork_automaton/neural/model.py',
    'src/patchwork_automaton/neural/batching.py',
    'src/patchwork_automaton/neural/inference_processes.py',
    'src/patchwork_automaton/neural/direct_standard.py',
    'src/patchwork_automaton/neural/standard_features.py',
    'src/patchwork_automaton/neural/standard_inference.py',
    'src/patchwork_automaton/rl/rustmage_processes.py',
    'src/patchwork_automaton/rl/standard_imitation.py',
    'src/patchwork_automaton/rl/standard_direct.py',
    'src/patchwork_automaton/rl/standard_ppo.py',
    'src/patchwork_automaton/rl/standard_selfplay_processes.py',
    'src/patchwork_automaton/rl/standard_arena_report.py',
    'src/patchwork_automaton/data/standard-baselines.json',
)


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


def checkpoint_path(path):
    path = Path(path)
    return path / 'policy.keras' if path.is_dir() else path


def checkpoint_files(path):
    path = checkpoint_path(path)
    sidecar = path.with_suffix('.manifest.json')
    for candidate in (path, sidecar):
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
    return (path.resolve(), sidecar.resolve())


def native_files():
    root = Path(os.environ.get('RUSTMAGE_ROOT', '.local/rustmage')).resolve()
    binding = root / 'python' / 'rustmage.py'
    schema = root / 'data' / 'observation-schema.txt'
    filename = {'linux': 'librustmage.so', 'darwin': 'librustmage.dylib',
                'win32': 'rustmage.dll'}[sys.platform]
    library = Path(os.environ.get('RUSTMAGE_LIBRARY', root / 'target' / 'release' / filename)).resolve()
    files = (binding, schema, library)
    if os.environ.get('PATCHWORK_STANDARD_PLAYBACK', 'direct') == 'direct':
        from patchwork_automaton.neural.direct_standard import library_path
        files += (library_path(),)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return files


def arena_cells(learner_decks, opponent_decks, opponents, panel):
    if panel not in PANELS:
        raise ValueError('Unknown arena panel')
    learner_decks, opponent_decks, opponents = tuple(learner_decks), tuple(opponent_decks), tuple(opponents)
    for name, values, known in (('learner deck', learner_decks, STANDARD_DECKS),
                                ('opponent deck', opponent_decks, STANDARD_DECKS),
                                ('opponent style', opponents, OPPONENTS)):
        if not values or len(values) != len(set(values)) or any(value not in known for value in values):
            raise ValueError(f'Invalid or duplicate {name}')
    cells = tuple((learner, opponent, style)
                  for style in opponents for learner in learner_decks for opponent in opponent_decks
                  if panel == 'matrix' or learner == opponent)
    if not cells:
        raise ValueError('Arena panel has no selected cells')
    return cells


def cell_id(learner_deck, opponent_deck, opponent_style):
    return f'{learner_deck}--{opponent_deck}--{opponent_style}'


def scheduled_games(learner_deck, opponent_deck, opponent_style, *, cell_index,
                    games_per_cell, seed_base, opponent_sha256=None):
    if learner_deck not in STANDARD_DECKS or opponent_deck not in STANDARD_DECKS:
        raise ValueError('Unknown Standard deck')
    if opponent_style not in OPPONENTS:
        raise ValueError('Unknown opponent style')
    if type(cell_index) is not int or cell_index < 0:
        raise ValueError('Nonnegative cell index required')
    if type(games_per_cell) is not int or games_per_cell < 4 or games_per_cell % 4:
        raise ValueError('Games per cell must be a positive multiple of four')
    if type(seed_base) is not int or seed_base < 0 or seed_base + (cell_index + 1) * games_per_cell > 2**64:
        raise ValueError('Arena seeds must fit unsigned 64-bit integers')
    if opponent_style == 'checkpoint':
        if not isinstance(opponent_sha256, str) or len(opponent_sha256) != 64 or any(
                letter not in '0123456789abcdef' for letter in opponent_sha256):
            raise ValueError('Checkpoint opponent requires a SHA256 digest')
    elif opponent_sha256 is not None:
        raise ValueError('Script opponent cannot receive checkpoint identity')
    order = tuple(STANDARD_DECKS)
    pairing = sorted((learner_deck, opponent_deck), key=order.index)
    identifier = cell_id(learner_deck, opponent_deck, opponent_style)
    for offset in range(games_per_cell):
        index = cell_index * games_per_cell + offset
        learner_seat = offset % 2
        decks = [opponent_deck, opponent_deck]
        decks[learner_seat] = learner_deck
        game = dict(index=index, seed=seed_base + index, cell_id=identifier,
                    pairing=pairing, decks=decks, learner_seat=learner_seat,
                    learner_deck=learner_deck, opponent_deck=opponent_deck,
                    opponent_style=opponent_style,
                    opponent_kind='frozen_neural' if opponent_style == 'checkpoint' else 'scripted',
                    starting_player=f'P{(offset // 2) % 2}', partition='evaluation')
        if opponent_style == 'checkpoint':
            game.update(opponent_checkpoint_sha256=opponent_sha256, opponent_version=0)
        yield dict(index=index, seed=seed_base + index, game=game)


def validate_row(row, job):
    expected = job['game']
    if not isinstance(row, dict) or any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Arena game identity mismatch: {job["index"]}')
    if row.get('trajectories') != []:
        raise ValueError('Read-only arena wrote a training trajectory')
    if row.get('behavior_version') != 0:
        raise ValueError('Arena did not use frozen learner version zero')
    status = row.get('status')
    if status not in ('win', 'loss', 'draw', 'cutoff', 'error'):
        raise ValueError('Unknown arena game status')
    if status in ('win', 'loss'):
        winner = 0 if status == 'win' else 1  # Native runner status is P0-relative.
        if row.get('winner_seat') != winner or row.get('draw') is not False:
            raise ValueError('Arena winner disagrees with native result')
        if row.get('learner_won') is not (winner == expected['learner_seat']):
            raise ValueError('Arena learner result disagrees with winner')
    elif status == 'draw':
        if row.get('winner_seat') is not None or row.get('draw') is not True or row.get('learner_won') is not False:
            raise ValueError('Inconsistent arena draw')
    elif row.get('winner_seat') is not None or row.get('learner_won') is True:
        raise ValueError('Failed arena game claims a winner')
    return row


def read_rows(path, jobs):
    expected = {job['index']: job for job in jobs}
    found = {}
    if path.exists():
        offset = 0
        with path.open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):
                    # The writer fsyncs one newline-terminated result at a time.
                    # A crash may leave only the last append torn; replay it.
                    with path.open('r+b') as writable:
                        writable.truncate(offset)
                        writable.flush()
                        os.fsync(writable.fileno())
                    print(json.dumps(dict(event='discarded_incomplete_game_tail',
                                          path=str(path), bytes=len(line))), flush=True)
                    break
                row = json.loads(line)
                index = row.get('index')
                if index not in expected or index in found:
                    raise ValueError('Unexpected or duplicate durable arena game')
                validate_row(row, expected[index])
                found[index] = row
                offset += len(line)
    return found


def prepare_output(output, manifest, *, resume):
    """Never replace an existing run or resume with changed inputs."""
    output = Path(output).resolve()
    manifest_path = output / 'manifest.json'
    if output.exists():
        if not resume or not manifest_path.is_file():
            raise FileExistsError('Existing arena output requires --resume and its manifest')
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Arena configuration, source or checkpoint changed')
    else:
        if resume:
            raise FileNotFoundError('Cannot resume an arena output that does not exist')
        output.mkdir(parents=True)
        atomic_json(manifest_path, manifest)
    return output


def manifest(args, cells, learner_files, opponent_files):
    source_hashes = {relative: sha256(PROJECT_ROOT / relative) for relative in SOURCE_FILES}
    engine_hashes = {str(path): sha256(path) for path in native_files()}
    return dict(schema=1, engine_revision=RUSTMAGE_REVISION,
                observation_schema=OBSERVATION_SCHEMA,
                native_files_sha256=engine_hashes,
                learner_label=args.label,
                learner_checkpoint=str(learner_files[0]),
                learner_files_sha256={str(path): sha256(path) for path in learner_files},
                opponent_label=args.opponent_label if opponent_files else None,
                opponent_checkpoint=str(opponent_files[0]) if opponent_files else None,
                opponent_files_sha256={str(path): sha256(path) for path in opponent_files},
                source_files_sha256=source_hashes,
                panel=args.panel, cells=[cell_id(*cell) for cell in cells],
                learner_decks=list(args.learner_decks), opponent_decks=list(args.opponent_decks),
                opponents=list(args.opponents), mode=args.mode,
                games_per_cell=args.games_per_cell, seed_base=args.seed_base,
                game_timeout=args.game_timeout, max_turns=args.max_turns,
                max_decisions=args.max_decisions, workers=args.workers,
                inference_processes=args.inference_processes,
                opponent_inference_processes=args.opponent_inference_processes)


def _annotate(row, args, style):
    row = dict(row)
    row['learner_label'] = args.label
    row['opponent_label'] = args.opponent_label if style == 'checkpoint' else (
        'StandardTactical' if style == 'tactical' else 'StandardAggro')
    return row


def _progress(output, *, cell, completed, total):
    atomic_json(output / 'progress.json', dict(cell=cell, completed=completed,
                total=total, timestamp=datetime.now(timezone.utc).isoformat()))


async def run(args):
    from patchwork_automaton.baselines.standard_tactical import StandardTactical
    from patchwork_automaton.neural.devices import configure_runtime, device_scope
    from patchwork_automaton.neural.standard_policy import StandardPolicy
    from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector
    from patchwork_automaton.rl.standard_selfplay_processes import StandardSelfplayCollector

    cells = arena_cells(args.learner_decks, args.opponent_decks, args.opponents, args.panel)
    learner_files = checkpoint_files(args.checkpoint)
    opponent_files = checkpoint_files(args.opponent_checkpoint) if 'checkpoint' in args.opponents else ()
    config = manifest(args, cells, learner_files, opponent_files)
    configure_runtime('cpu', hide_gpu=True)
    with device_scope('cpu'):
        learner = StandardPolicy.load(learner_files[0]).model
        opponent = StandardPolicy.load(opponent_files[0]).model if opponent_files else None
    output = prepare_output(args.output, config, resume=args.resume)
    print(json.dumps(dict(event='models_loaded', learner=args.label,
                          opponent=args.opponent_label if opponent is not None else None)), flush=True)

    all_rows = {}
    async def evaluate_style(style, collector):
        for index, (learner_deck, opponent_deck, cell_style) in enumerate(cells):
            if cell_style != style:
                continue
            identifier = cell_id(learner_deck, opponent_deck, style)
            directory = output / 'cells' / identifier
            directory.mkdir(parents=True, exist_ok=True)
            jobs = list(scheduled_games(learner_deck, opponent_deck, style,
                cell_index=index, games_per_cell=args.games_per_cell,
                seed_base=args.seed_base,
                opponent_sha256=config['opponent_files_sha256'].get(str(opponent_files[0]))
                    if style == 'checkpoint' else None))
            expected_jobs = {job['index']: job for job in jobs}
            rows_path = directory / 'games.jsonl'
            rows = read_rows(rows_path, jobs)
            expected_opponent_label = args.opponent_label if style == 'checkpoint' else (
                'StandardTactical' if style == 'tactical' else 'StandardAggro')
            for row in rows.values():
                if row.get('learner_label') != args.label or row.get('opponent_label') != expected_opponent_label:
                    raise ValueError('Durable arena entrant label differs from manifest')
            pending = [job for job in jobs if job['index'] not in rows]
            _progress(output, cell=identifier, completed=len(rows), total=len(jobs))
            if pending:
                with rows_path.open('a') as stream:
                    def persist(raw):
                        row = _annotate(raw, args, style)
                        game_index = row.get('index')
                        if game_index not in expected_jobs or game_index in rows:
                            raise ValueError('Unexpected or duplicate arena game result')
                        validate_row(row, expected_jobs[game_index])
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                        rows[game_index] = row
                        if len(rows) % 16 == 0:
                            _progress(output, cell=identifier, completed=len(rows), total=len(jobs))
                    options = dict(game_runner='standard_ppo',
                                   diagnostics=directory / 'diagnostics',
                                   greedy=args.mode == 'greedy',
                                   game_timeout=args.game_timeout,
                                   max_turns=args.max_turns,
                                   max_decisions=args.max_decisions)
                    if style == 'checkpoint':
                        options.update(opponent_sha256=config['opponent_files_sha256'][str(opponent_files[0])],
                                       opponent_version=0)
                    else:
                        options['opponent_policy_factory'] = StandardTactical if style == 'tactical' else None
                    await collector.run_schedule(pending, progress=persist, **options)
            if len(rows) != len(jobs):
                raise RuntimeError(f'Incomplete arena cell {identifier}: {len(rows)} / {len(jobs)}')
            ordered = [rows[job['index']] for job in jobs]
            all_rows[identifier] = ordered
            cell_report = summarize_arena(ordered)['cells'][0]
            atomic_json(directory / 'summary.json', cell_report)
            _progress(output, cell=identifier, completed=len(rows), total=len(jobs))
            atomic_json(output / 'summary.json', dict(complete=False,
                        **summarize_arena([row for group in all_rows.values() for row in group])))
            print(json.dumps(dict(event='cell_complete', cell=identifier,
                                  attempted=cell_report['attempted'],
                                  win_rate=cell_report['win_rate'])), flush=True)

    script_styles = [style for style in args.opponents if style != 'checkpoint']
    if script_styles:
        collector = ProcessGameCollector(learner, workers=args.workers,
            inference_processes=args.inference_processes, inference_device='cpu')
        try:
            await collector.start(version=0)
            for style in script_styles:
                await evaluate_style(style, collector)
        finally:
            await collector.close()
    if 'checkpoint' in args.opponents:
        collector = StandardSelfplayCollector(learner, opponent, workers=args.workers,
            inference_processes=args.inference_processes,
            opponent_inference_processes=args.opponent_inference_processes)
        try:
            await collector.start(version=0, opponent_version=0)
            await evaluate_style('checkpoint', collector)
        finally:
            await collector.close()

    expected_hashes = {str(path): sha256(path) for path in learner_files}
    if expected_hashes != config['learner_files_sha256']:
        raise RuntimeError('Learner checkpoint changed during arena evaluation')
    expected_opponent_hashes = {str(path): sha256(path) for path in opponent_files}
    if expected_opponent_hashes != config['opponent_files_sha256']:
        raise RuntimeError('Opponent checkpoint changed during arena evaluation')
    if {relative: sha256(PROJECT_ROOT / relative) for relative in SOURCE_FILES} != config['source_files_sha256']:
        raise RuntimeError('Arena source changed during evaluation')
    if {str(path): sha256(path) for path in native_files()} != config['native_files_sha256']:
        raise RuntimeError('Native engine or binding changed during evaluation')
    ordered_rows = [row for cell in cells for row in all_rows[cell_id(*cell)]]
    report = dict(complete=True, clean=not any(row['status'] in ('cutoff', 'error') for row in ordered_rows),
                  **summarize_arena(ordered_rows))
    atomic_json(output / 'summary.json', report)
    return report


async def run_with_timer(args):
    # Keep the event loop responsive to completed executor futures on hosts
    # where TensorFlow work does not wake the selector reliably. A callback
    # also remains active through asyncio.run's executor shutdown.
    loop = asyncio.get_running_loop()
    def tick():
        if not loop.is_closed():
            loop.call_later(.002, tick)
    loop.call_later(.002, tick)
    return await run(args)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='Learner policy.keras or checkpoint directory')
    p.add_argument('--label', required=True, help='Stable learner label recorded in results')
    p.add_argument('--opponent-checkpoint', type=Path,
                   help='Frozen Standard policy.keras or directory when --opponents includes checkpoint')
    p.add_argument('--opponent-label', default='frozen-neural')
    p.add_argument('--opponents', nargs='+', choices=OPPONENTS, default=['tactical'])
    p.add_argument('--learner-decks', nargs='+', choices=tuple(STANDARD_DECKS), default=list(STANDARD_DECKS))
    p.add_argument('--opponent-decks', nargs='+', choices=tuple(STANDARD_DECKS), default=list(STANDARD_DECKS))
    p.add_argument('--panel', choices=PANELS, default='mirrors')
    p.add_argument('--mode', choices=('greedy', 'sampled'), default='greedy')
    p.add_argument('--games-per-cell', type=int, default=64)
    p.add_argument('--seed-base', type=int, default=DEFAULT_SEED)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--inference-processes', type=int, default=1)
    p.add_argument('--opponent-inference-processes', type=int, default=1)
    p.add_argument('--game-timeout', type=float, default=60.)
    p.add_argument('--max-turns', type=int, default=50)
    p.add_argument('--max-decisions', type=int, default=6000)
    p.add_argument('--output', type=Path, help='New output directory; --resume requires an identical manifest')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plan', action='store_true', help='Print cell/game count without loading models or writing files')
    return p


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.games_per_cell < 4 or args.games_per_cell % 4:
        command.error('--games-per-cell must be a positive multiple of four')
    if args.workers < 1 or not 1 <= args.inference_processes <= args.workers or not 1 <= args.opponent_inference_processes <= args.workers:
        command.error('Inference processes must be between one and the game-worker count')
    if args.max_turns < 1 or args.max_decisions < 1 or args.game_timeout <= 0 or not math.isfinite(args.game_timeout):
        command.error('Positive finite game limits required')
    if ('checkpoint' in args.opponents) != (args.opponent_checkpoint is not None):
        command.error('--opponent-checkpoint is required exactly when --opponents includes checkpoint')
    if not args.label.strip() or not args.opponent_label.strip():
        command.error('Entrant labels must be nonempty')
    try:
        cells = arena_cells(args.learner_decks, args.opponent_decks, args.opponents, args.panel)
    except ValueError as error:
        command.error(str(error))
    if args.seed_base < 0 or args.seed_base + len(cells) * args.games_per_cell > 2**64:
        command.error('Arena seeds exceed unsigned 64-bit range')
    if args.plan:
        print(json.dumps(dict(panel=args.panel, mode=args.mode,
                              cells=[cell_id(*cell) for cell in cells],
                              games_per_cell=args.games_per_cell,
                              total_games=len(cells) * args.games_per_cell), indent=2))
        return 0
    if args.output is None:
        command.error('--output is required unless --plan is used')
    os.environ['PATCHWORK_ARENA_POLL_INFERENCE'] = '1'
    report = asyncio.run(run_with_timer(args))
    print(json.dumps(dict(output=str(args.output.resolve()),
                          complete=report['complete'], clean=report['clean'],
                          aggregate=report['aggregate']), sort_keys=True))
    return 0 if report['clean'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
