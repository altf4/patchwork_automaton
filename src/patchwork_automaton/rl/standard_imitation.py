"""Whole Standard games for expert recording and fixed-opponent evaluation.

Collection imports no TensorFlow. Each policy receives only its own private
observation, including during mulligans. Inference can use a shared batcher.
"""
import asyncio
from collections import Counter
from pathlib import Path
import time
import traceback

from ..backends.rustmage import RustMageMatch, domain_action
from ..baselines.standard import StandardAggro
from ..baselines.standard_decks import standard_deck
from ..neural.rustmage_observation import learning_observation


def _play_standard_game_python(game, *, output=None, model=None, batcher=None,
                       learner_seat=None, greedy=False, version=0,
                       opponent_model=None, opponent_batcher=None,
                       opponent_version=None, opponent_sha256=None,
                       opponent_policy_factory=None,
                       max_decisions=6000, max_turns=50, game_timeout=60, progress_path=None,
                       record_ppo=False):
    """Play through turn 50; resource draws are explicitly adjudicated."""
    async def play():
        started = time.monotonic()
        row = dict(game, status='error', decisions=0, trajectories=[])
        writers, policies, counts = {}, {}, Counter()
        last_turn = None
        try:
            opponent_kind = game.get('opponent_kind', 'scripted')
            if opponent_kind not in ('scripted', 'frozen_neural'):
                raise ValueError('Unknown Standard opponent kind')
            if opponent_kind == 'frozen_neural':
                if learner_seat not in (0, 1) or opponent_model is None or opponent_batcher is None:
                    raise ValueError('Frozen neural opponent requires a learner seat and separate inference')
                if game.get('opponent_checkpoint_sha256') != opponent_sha256 or not opponent_sha256:
                    raise ValueError('Frozen neural opponent checkpoint hash mismatch')
                if game.get('opponent_version') != opponent_version or opponent_version is None:
                    raise ValueError('Frozen neural opponent version mismatch')
            make_script = opponent_policy_factory or StandardAggro
            for player, deck_id in enumerate(game['decks']):
                if player == learner_seat:
                    from ..neural.standard_inference import StandardInferencePolicy as StandardPolicy
                    if output is not None and record_ppo:
                        from ..neural.rollout_records import TrajectoryWriter
                        writers[player] = TrajectoryWriter(Path(output) / f"game-{game['index']:06d}-p{player}.pkl.gz")
                    policies[player] = StandardPolicy(model, batcher=batcher,
                        seed=game['seed'] * 2 + player, greedy=greedy, version=version,
                        sink=writers.get(player))
                elif opponent_kind == 'frozen_neural':
                    from ..neural.standard_inference import StandardInferencePolicy as StandardPolicy
                    policies[player] = StandardPolicy(opponent_model, batcher=opponent_batcher,
                        seed=game['seed'] * 2 + player, greedy=greedy,
                        version=opponent_version, sink=None)
                elif output is not None and not record_ppo:
                    from ..neural.demonstrations import DemonstrationWriter, RecordingExpert
                    from ..neural.standard_features import StandardEncoder
                    writers[player] = DemonstrationWriter(Path(output) / f"game-{game['index']:06d}-p{player}.pkl.gz")
                    policies[player] = RecordingExpert(writers[player], make_script(deck_id), StandardEncoder())
                else:
                    policies[player] = make_script(deck_id)
            with RustMageMatch(*(standard_deck(d, catalog=game.get('deck_catalog', 'standard')) for d in game['decks']),
                    seed=game['seed'], starting_player=game['starting_player']) as match:
                while True:
                    views = [learning_observation(match, f'P{p}') for p in range(2)]
                    for player, view in enumerate(views):
                        observe = getattr(policies[player], 'observe', None)
                        if observe:
                            observe(view)
                    row.update(turn=views[0].turn, final_life=[p.life for p in views[0].players])
                    if progress_path is not None and (row['decisions'] % 100 == 0 or row['turn'] != last_turn):
                        path = Path(progress_path)
                        temporary = path.with_suffix('.tmp')
                        temporary.write_text(__import__('json').dumps(row))
                        temporary.replace(path)
                    last_turn = row['turn']
                    if row['turn'] > max_turns:
                        row.update(status='draw', reason='turn_limit', adjudicated=True,
                            turn_limit=max_turns, draw=True, winner_seat=None, winner_deck=None)
                        if learner_seat is not None:
                            row.update(learner_seat=learner_seat, learner_won=False)
                        break
                    if views[0].terminal:
                        outcomes = [v.outcome for v in views]
                        if not views[1].terminal or any(o is None for o in outcomes):
                            raise RuntimeError('Missing terminal seat outcome')
                        winners = [p for p, o in enumerate(outcomes) if o.won]
                        draw = all(o.draw for o in outcomes)
                        if not draw and (len(winners) != 1 or outcomes[1 - winners[0]].won is not False):
                            raise RuntimeError('Inconsistent terminal outcomes')
                        winner = None if draw else winners[0]
                        row.update(status='draw' if draw else 'win' if winner == 0 else 'loss',
                            winner_seat=winner, winner_deck=None if draw else game['decks'][winner], draw=draw)
                        if learner_seat is not None:
                            row.update(learner_seat=learner_seat, learner_won=winner == learner_seat)
                        for player, writer in writers.items():
                            if not writer.count:
                                raise RuntimeError('Empty expert seat trajectory')
                            row['trajectories'].append(dict(path=str(writer.path.resolve()), samples=writer.count,
                                player=player, deck=game['decks'][player]))
                        break
                    if time.monotonic() - started >= game_timeout:
                        row.update(status='draw', reason='wall_clock_timeout', adjudicated=True,
                            timeout_seconds=game_timeout, draw=True, winner_seat=None, winner_deck=None)
                        if learner_seat is not None:
                            row.update(learner_seat=learner_seat, learner_won=False)
                        break
                    if row['decisions'] >= max_decisions:
                        row.update(status='cutoff', reason='decision_limit')
                        break
                    actors = [p for p, v in enumerate(views) if v.decision is not None]
                    if len(actors) != 1:
                        raise RuntimeError('Expected exactly one prompted player')
                    player = actors[0]
                    view = views[player]
                    response = (policies[player].act(view).response
                                if player == learner_seat or opponent_kind == 'frozen_neural'
                                else policies[player].choose(view))
                    option = next(o for o in view.decision.options if o.id == response.option)
                    action = option.constraints['action'] if option.kind == 'finite' else domain_action(option, response.value)
                    counts[f"p{player}:{action['kind']}"] += 1
                    raw = match._raw(view.actor.value)
                    updated = (match.native_game.step(raw, int(option.id.split(':')[1])) if option.kind == 'finite'
                               else match.native_game.act(raw, action))
                    match._after_action(updated, view.actor.value, raw['revision'])
                    row['decisions'] += 1
        except Exception:
            row.update(status='error', error=traceback.format_exc(), trajectories=[])
        finally:
            for player, writer in writers.items():
                writer.close()
                if row.get('adjudicated') and writer.count:
                    row['trajectories'].append(dict(path=str(writer.path.resolve()), samples=writer.count,
                        player=player, deck=game['decks'][player], partial=True))
            row.update(seconds=time.monotonic() - started, actions=dict(counts))
            if record_ppo and row['status'] in ('win', 'loss', 'draw'):
                for trajectory in row['trajectories']:
                    player = trajectory['player']
                    trajectory.update(reward=0. if row['status']=='draw' else
                        1. if player==row['winner_seat'] else -1.,
                        value_sum=writers[player].value_sum,
                        value_square_sum=writers[player].value_square_sum)
        return row
    return asyncio.run(play())


def playback_mode():
    """Use the verified direct engine by default; Python is a reference mode."""
    import os
    mode = os.environ.get('PATCHWORK_STANDARD_PLAYBACK', 'direct')
    if mode not in ('direct', 'python'):
        raise ValueError('PATCHWORK_STANDARD_PLAYBACK must be direct or python')
    return mode


def play_standard_game(game, **options):
    mode = playback_mode()
    if mode == 'python':
        row = _play_standard_game_python(game, **options)
    else:
        from .standard_direct import play
        row = play(game, **options)
    row['playback_pipeline'] = mode
    return row


def adjudicate_timeout(game, *, output=None, diagnostics, seconds=60, learner_seat=None,
                       original_result=None, record_ppo=False):
    """Preserve the seed, diagnostic state and readable expert prefix, never replay."""
    import gzip
    import io
    import json
    import pickle
    import shutil
    import zlib
    from ..neural.demonstrations import Demonstration, DemonstrationWriter
    if record_ppo:
        from ..neural.rollout_records import Sample as Record, TrajectoryWriter as Writer
    else:
        Record, Writer = Demonstration, DemonstrationWriter
    directory = Path(diagnostics) / f"game-{game['index']:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    row = dict(game, status='draw', draw=True, adjudicated=True, reason='wall_clock_timeout',
        timeout_seconds=60, seconds=seconds, winner_seat=None, winner_deck=None, trajectories=[])
    state = directory / 'progress.json'
    if state.exists():
        previous = json.loads(state.read_text())
        for name in ('turn', 'decisions', 'final_life'):
            if name in previous: row[name] = previous[name]
    if original_result is not None:
        (directory / 'original-result.json').write_text(json.dumps(original_result, indent=2)+'\n')
        for name in ('turn', 'decisions', 'final_life'):
            if name in original_result: row[name] = original_result[name]
    if learner_seat is not None:
        row.update(learner_seat=learner_seat, learner_won=False)
    if output is not None:
        for player, deck in enumerate(game['decks']):
            if record_ppo and player != learner_seat: continue
            path = Path(output) / f"game-{game['index']:06d}-p{player}.pkl.gz"
            if not path.exists(): continue
            shutil.copy2(path, directory / path.name)
            # A killed writer can have an unfinished gzip member or last pickle.
            decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
            stream = io.BytesIO(decoder.decompress(path.read_bytes()))
            writer = Writer(path)
            try:
                while True:
                    try: record = pickle.load(stream)
                    except (EOFError, pickle.UnpicklingError): break
                    if not isinstance(record, Record):
                        raise ValueError('Unexpected timeout trajectory record')
                    writer.write(record)
            finally: writer.close()
            if writer.count:
                row['trajectories'].append(dict(path=str(path.resolve()), samples=writer.count,
                    player=player, deck=deck, partial=True))
                if record_ppo:
                    row['trajectories'][-1].update(reward=0.,value_sum=writer.value_sum,
                        value_square_sum=writer.value_square_sum)
    row['diagnostics'] = str(directory.resolve())
    (directory / 'adjudication.json').write_text(json.dumps(row, indent=2)+'\n')
    return row


def _isolated_game(connection, game, options, trace_path, close_connections=()):
    import faulthandler
    import signal
    from .inference_transport import send_message
    # Fork copies descriptors, not ownership. The child must not retain either
    # the reusable transport or the parent ends of its disposable transports.
    for inherited in close_connections:
        inherited.close()
    with open(trace_path, 'w') as trace:
        faulthandler.register(signal.SIGUSR1, file=trace, all_threads=True)
        try:
            row = play_standard_game(game, **options)
            batcher = options.get('batcher')
            if batcher is not None:
                row.update(game_inference_seconds=batcher.wait_seconds,
                           game_inference_requests=batcher.requests)
            opponent_batcher = options.get('opponent_batcher')
            if game.get('opponent_kind') == 'frozen_neural' and opponent_batcher is not None:
                row.update(opponent_game_inference_seconds=opponent_batcher.wait_seconds,
                           opponent_game_inference_requests=opponent_batcher.requests)
            send_message(connection, row)
        finally:
            connection.close()
            for role in ('batcher', 'opponent_batcher'):
                batcher = options.get(role)
                if batcher is not None:
                    batcher.connection.close()


_game_worker_broken = False


def _stop_game_child(process, *, terminate=False):
    """Try every cleanup stage, even when an earlier stage failed."""
    errors = []
    if process.pid is None:
        return errors
    try:
        if terminate and process.is_alive():
            process.terminate()
    except BaseException as error:
        errors.append(error)
    try:
        process.join(1)
    except BaseException as error:
        errors.append(error)
    try:
        if process.is_alive():
            process.kill()
            process.join(1)
        if process.is_alive():
            errors.append(RuntimeError('Native game child did not stop'))
    except BaseException as error:
        errors.append(error)
    return errors


def timed_standard_game(game, *, diagnostics, game_timeout=60, **options):
    """A TF-free worker supervises a disposable native game with a hard deadline.

    Only fork a single-threaded game worker, never the TensorFlow learner.
    Killing the child also interrupts an engine call that cannot return to Python.
    """
    import json
    import multiprocessing as mp
    import os
    import signal
    import sys
    import threading
    from .inference_transport import InferenceCleanupError, InferenceIsolation, recv_message
    global _game_worker_broken
    if 'tensorflow' in sys.modules:
        raise RuntimeError('Native game isolation must run outside TensorFlow')
    if _game_worker_broken:
        raise RuntimeError('Native game worker is unusable after a cleanup failure')
    if threading.active_count() != 1:
        raise RuntimeError('Native game isolation requires a single-threaded worker before fork')
    if game_timeout <= 0: raise ValueError('Positive wall-clock limit required')
    if playback_mode() == 'direct':
        # Load and configure in the single-threaded worker before the game
        # fork; fail before scheduling if the verified bridge is unavailable.
        from ..neural.direct_standard import library
        library()
    directory = Path(diagnostics) / f"game-{game['index']:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    progress = directory / 'progress.json'
    options = dict(options, progress_path=progress, game_timeout=float('inf'))
    context = mp.get_context('fork')
    isolation = InferenceIsolation(context, options)
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_isolated_game,
        args=(child, game, isolation.options, directory / 'python-stack.txt',
              (parent, *isolation.child_close_connections())))
    started = time.monotonic()
    row = error = None
    cleanup_errors = []
    try:
        process.start()
        child.close()
        isolation.start()
        try:
            # Readiness of a frame header does not imply the complete result has
            # arrived. The hard deadline covers the complete result frame.
            row = recv_message(parent, timeout=max(0., game_timeout-(time.monotonic()-started)))
        except TimeoutError:
            try:
                os.kill(process.pid, signal.SIGUSR1)
            except ProcessLookupError:
                pass
            time.sleep(.02)
            cleanup_errors.extend(_stop_game_child(process, terminate=True))
            row = adjudicate_timeout(game, output=options.get('output'), diagnostics=diagnostics,
                seconds=time.monotonic() - started, learner_seat=options.get('learner_seat'),
                record_ppo=options.get('record_ppo',False))
            row['playback_pipeline'] = playback_mode()
            row['timeout_seconds'] = game_timeout
            (directory / 'adjudication.json').write_text(json.dumps(row, indent=2)+'\n')
        except EOFError as cause:
            raise RuntimeError(f"Game child exited without a result: {game}") from cause
        else:
            cleanup_errors.extend(_stop_game_child(process))
            if row.get('adjudicated'):
                row['diagnostics'] = str(directory.resolve())
                (directory / 'adjudication.json').write_text(json.dumps(row, indent=2)+'\n')
    except BaseException as cause:
        error = cause
    finally:
        cleanup_errors.extend(_stop_game_child(process, terminate=True))
        for connection in (parent, child):
            try:
                connection.close()
            except BaseException as cause:
                cleanup_errors.append(cause)
        try:
            isolation.close()
        except BaseException as cause:
            cleanup_errors.append(cause)
        try:
            process.close()
        except BaseException as cause:
            cleanup_errors.append(cause)
    if cleanup_errors:
        _game_worker_broken = True
        isolation.invalidate()
        message = '; '.join(repr(cause) for cause in cleanup_errors)
        if error is None:
            error = InferenceCleanupError(message, completed_row=row)
        else:
            error.add_note('Native game cleanup also failed: ' + message)
    if error is None:
        try:
            if row.get('status') not in ('win', 'loss', 'draw'):
                # A returned failure still needs its original diagnostics.
                (directory / 'original-result.json').write_text(json.dumps(row, indent=2)+'\n')
            elif not row.get('adjudicated'):
                for path in directory.iterdir():
                    path.unlink()
                directory.rmdir()
        except Exception as cause:
            error = cause
    if error is not None:
        if row is not None:
            error.completed_row = row
            try:
                (directory / 'completed-result.json').write_text(json.dumps(row, indent=2)+'\n')
            except Exception as publication_error:
                error.add_note('Completed-result diagnostic publication also failed: '+repr(publication_error))
        raise error
    return row


def play_standard_evaluation(*, seed, index, learner_deck, opponent_deck,
                             model, batcher, **options):
    from ..baselines.standard_decks import STANDARD_DECKS
    order = tuple(STANDARD_DECKS)
    pair = sorted((learner_deck, opponent_deck), key=order.index)
    decks = pair if index % 2 == 0 else pair[::-1]
    seat = decks.index(learner_deck) if learner_deck != opponent_deck else index % 2
    game = dict(index=index, seed=seed, pairing=pair, decks=decks,
                starting_player=f'P{(index // 2) % 2}')
    game_error = None
    try:
        return timed_standard_game(game, model=model, batcher=batcher, learner_seat=seat, **options)
    except BaseException as error:
        game_error = error
        raise
    finally:
        # Disposable proxy traffic has drained; require an exact acknowledgement
        # on the persistent connection before another game may use it.
        if not batcher.broken:
            try:
                batcher.reset(seed)
            except BaseException as reset_error:
                if game_error is None:
                    raise
                game_error.add_note('Inference reset also failed: '+repr(reset_error))
