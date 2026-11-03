"""Standard game playback with Rust-owned state and direct numerical inputs.

Neural seats receive numerical features from their own redacted Rust views.
The opt-in identity sidecar and scripted policies also use that seat's private
Python observation.
"""
from ..neural.compact_dispatch import is_compact,codec,V2

from collections import Counter
import ctypes as C
import json
from pathlib import Path
from types import SimpleNamespace
import time
import traceback
import uuid
import numpy as np

from ..backends.rustmage import _plain, domain_action
from ..baselines.standard import StandardAggro
from ..baselines.standard_decks import standard_deck
from ..core import Handle
from ..neural.direct_standard import Metadata, arrays, check, library, string, validate_model
from ..neural.rustmage_observation import learning_observation
from ..neural.rollout_records import Sample, TrajectoryWriter


def play(game, *, output=None, model=None, batcher=None, learner_seat=None,
         greedy=False, version=0, opponent_model=None, opponent_batcher=None,
         opponent_version=None, opponent_sha256=None, opponent_policy_factory=None,
         max_decisions=6000, max_turns=50, game_timeout=60, progress_path=None,
         record_ppo=False):
    """Play a game with an in-process typed engine and private model inputs."""
    started = time.monotonic()
    row = dict(game, status='error', decisions=0, trajectories=[])
    lib = library()
    handle = None
    writers, scripts, counts = {}, {}, Counter()
    last_turn = None
    try:
        kind = game.get('opponent_kind', 'scripted')
        if kind not in ('scripted', 'frozen_neural'):
            raise ValueError('Unknown Standard opponent kind')
        if learner_seat not in (None, 0, 1):
            raise ValueError('Invalid learner seat')
        if learner_seat is not None:
            validate_model(model)
            if batcher is None: raise ValueError('Neural player requires inference')
        if kind == 'frozen_neural':
            if learner_seat is None or opponent_model is None or opponent_batcher is None:
                raise ValueError('Frozen neural opponent requires a learner seat and separate inference')
            validate_model(opponent_model)
            if not opponent_sha256 or game.get('opponent_checkpoint_sha256') != opponent_sha256:
                raise ValueError('Frozen neural opponent checkpoint hash mismatch')
            if opponent_version is None or game.get('opponent_version') != opponent_version:
                raise ValueError('Frozen neural opponent version mismatch')
        neural = {learner_seat} if learner_seat is not None else set()
        if kind == 'frozen_neural': neural.add(1 - learner_seat)
        if record_ppo and learner_seat is None: raise ValueError('PPO recording requires a learner')
        factory = opponent_policy_factory or StandardAggro
        for player in range(2):
            if player not in neural:
                scripts[player] = factory(game['decks'][player])
            if output is not None and record_ppo and player == learner_seat:
                writers[player] = TrajectoryWriter(Path(output) / f"game-{game['index']:06d}-p{player}.pkl.gz")
            elif output is not None and not record_ppo and player not in neural:
                from ..neural.demonstrations import DemonstrationWriter
                writers[player] = DemonstrationWriter(Path(output) / f"game-{game['index']:06d}-p{player}.pkl.gz")
        decks = [standard_deck(d, catalog=game.get('deck_catalog', 'standard')) for d in game['decks']]
        config = dict(decks=[[c.name for c in d.cards for _ in range(c.quantity)] for d in decks],
                      seed=game['seed'], starting_player=game['starting_player'], deck_rules='Constructed')
        handle = check(lib, lib.ds_new(json.dumps(config, separators=(',', ':')).encode()))
        rng = [np.random.default_rng(game['seed'] * 2 + p) for p in range(2)]
        states, previous = [None, None], [None, None]
        # Materialize only the acting player's already-redacted observation.
        game_handle = Handle('rustmage', 'game', uuid.uuid4().hex)
        session = uuid.uuid4().hex
        while True:
            views, metadata = [], []
            try:
                for player in range(2):
                    view = check(lib, lib.ds_observe(handle, player)); views.append(view)
                    meta = Metadata(); check(lib, lib.ds_metadata(view, C.byref(meta)))
                    side = model if player==learner_seat else opponent_model if player in neural else None
                    if not is_compact(side):
                        check(lib, lib.ds_history_observe(handle, view))
                    metadata.append(meta)
                script_views = {}
                def script_view(player):
                    if player not in script_views:
                        raw = json.loads(string(lib, lib.ds_debug_json(views[player])))
                        match = SimpleNamespace(session=session, handle=game_handle, _raw=lambda _: raw)
                        script_views[player] = learning_observation(match, f'P{player}')
                    return script_views[player]
                for player, policy in scripts.items():
                    observe = getattr(policy, 'observe', None)
                    if observe: observe(script_view(player))
                meta = metadata[0]
                row.update(turn=meta.turn, final_life=[meta.life0, meta.life1])
                if progress_path is not None and (row['decisions'] % 100 == 0 or row['turn'] != last_turn):
                    path = Path(progress_path); tmp = path.with_suffix('.tmp')
                    tmp.write_text(json.dumps(row)); tmp.replace(path)
                last_turn = row['turn']
                if row['turn'] > max_turns:
                    row.update(status='draw', reason='turn_limit', adjudicated=True,
                               turn_limit=max_turns, draw=True, winner_seat=None, winner_deck=None)
                    if learner_seat is not None: row.update(learner_seat=learner_seat, learner_won=False)
                    break
                if meta.outcome != -1:
                    if metadata[1].outcome != meta.outcome: raise RuntimeError('Inconsistent private outcomes')
                    winner = None if meta.outcome == 2 else meta.outcome
                    row.update(status='draw' if winner is None else 'win' if winner == 0 else 'loss',
                               winner_seat=winner, winner_deck=None if winner is None else game['decks'][winner],
                               draw=winner is None)
                    if learner_seat is not None: row.update(learner_seat=learner_seat, learner_won=winner == learner_seat)
                    for player, writer in writers.items():
                        if not writer.count: raise RuntimeError('Empty trajectory')
                        row['trajectories'].append(dict(path=str(writer.path.resolve()), samples=writer.count,
                                                        player=player, deck=game['decks'][player]))
                    break
                if time.monotonic() - started >= game_timeout:
                    row.update(status='draw', reason='wall_clock_timeout', adjudicated=True,
                               timeout_seconds=game_timeout, draw=True, winner_seat=None, winner_deck=None)
                    if learner_seat is not None: row.update(learner_seat=learner_seat, learner_won=False)
                    break
                if row['decisions'] >= max_decisions:
                    row.update(status='cutoff', reason='decision_limit'); break
                actor = meta.actor
                if actor not in (0, 1) or metadata[1].actor != actor:
                    raise RuntimeError('Expected one prompted player')
                view = views[actor]
                if actor in scripts:
                    # Policy compatibility needs a private Python view. It is
                    # never used for neural feature or history preparation.
                    obs = script_view(actor)
                    response = scripts[actor].choose(obs)
                    option = next(o for o in obs.decision.options if o.id == response.option)
                    action = _plain(option.constraints['action'] if option.kind == 'finite'
                                    else domain_action(option, response.value))
                    if actor in writers:
                        size = C.c_size_t(); check(lib, lib.ds_prepare(view, False, C.byref(size)))
                        index = next((i for i in range(size.value)
                                      if json.loads(string(lib, lib.ds_choice_debug_json(view, i))) == action), None)
                        if index is None: raise ValueError('Scripted action is absent from native choices')
                        if size.value > 1:
                            from ..neural.demonstrations import Demonstration
                            data = arrays(lib, lib.ds_encode(view)); data.update(arrays(lib, lib.ds_history_take(handle, view)))
                            if previous[actor] is not None: data['selection'] = previous[actor]
                            writers[actor].write(Demonstration(data, index, action['kind'], obs.decision.kind))
                            previous[actor] = data['candidates'][index].copy()
                    check(lib, lib.ds_apply_script(handle, view, json.dumps(action, separators=(',', ':')).encode()))
                    action_kind = action['kind']
                else:
                    size = C.c_size_t(); check(lib, lib.ds_prepare(view, False, C.byref(size)))
                    if size.value == 1:
                        index = 0
                    else:
                        side_model = model if actor == learner_seat else opponent_model
                        data = arrays(lib, lib.ds_encode(view))
                        compact=is_compact(side_model)
                        if compact:
                            if getattr(side_model,'input_format',None)==V2:
                                from ..neural.compact_identity import IdentityEncoder
                                encoder=IdentityEncoder();obs=script_view(actor)
                                choices=encoder.expand_options(obs.decision)
                                data=encoder.enrich(obs,choices,data)
                            data=codec(side_model).observation(data)
                        else:
                            data.update(arrays(lib, lib.ds_history_take(handle, view)))
                        state = states[actor] if states[actor] is not None else np.zeros(side_model.memory_size, np.float32)
                        if previous[actor] is not None: data['selection'] = previous[actor]
                        service = batcher if actor == learner_seat else opponent_batcher
                        role_version = version if actor == learner_seat else opponent_version
                        result = service.predict(dict(data, memory_in=state, advance_memory=np.ones(1, np.float32)),
                            version=role_version, identity=(game['seed'], actor), unbatched=True)
                        logits = np.asarray(result['logits'])[0].astype(np.float64)
                        if not np.isfinite(logits).all(): raise ValueError('Nonfinite model scores')
                        logp = logits - np.max(logits); logp -= np.log(np.exp(logp).sum())
                        index = int(np.argmax(logits) if greedy else rng[actor].choice(len(logp), p=np.exp(logp)))
                        if actor in writers:
                            value = float(np.asarray(result['value'])[0, 0])
                            if not np.isfinite(value): raise ValueError('Nonfinite behavior value')
                            writers[actor].write(Sample(data, state.copy(), True, index, value, logp.astype(np.float32)))
                        states[actor] = np.asarray(result['memory_out'])[0].copy()
                        previous[actor] = None if compact else data['candidates'][index].copy()
                    action_kind = string(lib, lib.ds_choice_kind(view, index))
                    check(lib, lib.ds_apply_choice(handle, view, index))
                counts[f'p{actor}:{action_kind}'] += 1; row['decisions'] += 1
            finally:
                for view in views: lib.ds_view_free(view)
    except Exception:
        row.update(status='error', error=traceback.format_exc(), trajectories=[])
    finally:
        if handle: lib.ds_free(handle)
        for player, writer in writers.items():
            writer.close()
            if row.get('adjudicated') and writer.count:
                row['trajectories'].append(dict(path=str(writer.path.resolve()), samples=writer.count,
                    player=player, deck=game['decks'][player], partial=True))
        row.update(seconds=time.monotonic() - started, actions=dict(counts))
        if record_ppo and row['status'] in ('win', 'loss', 'draw'):
            for trajectory in row['trajectories']:
                player = trajectory['player']
                trajectory.update(reward=0. if row['status'] == 'draw' else
                    1. if player == row['winner_seat'] else -1., value_sum=writers[player].value_sum,
                    value_square_sum=writers[player].value_square_sum)
    return row
