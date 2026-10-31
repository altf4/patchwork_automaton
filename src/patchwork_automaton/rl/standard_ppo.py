"""TensorFlow-free Standard PPO scheduling, recording, and bounded retention."""
from collections import Counter
from itertools import combinations_with_replacement
import math
from pathlib import Path
import shutil

from ..baselines.standard_decks import STANDARD_DECKS


def _thousand_game_schedule(seed, round_id):
    """Rotate indivisible pairing/role quotas; hold out exactly 200 games."""
    rows = []
    for pair_id, pair in enumerate(combinations_with_replacement(STANDARD_DECKS, 2)):
        # Cumulative quotas keep all six pairings within one game at every
        # round boundary, rather than favoring the same four pairings forever.
        before = ((round_id-1)*1000+pair_id)//6
        after = (round_id*1000+pair_id)//6
        held_before = ((round_id-1)*200+pair_id)//6
        held_after = (round_id*200+pair_id)//6
        roles = (0, 1) if pair[0] != pair[1] else (0,)
        strata = [(role, seat, start) for role in roles for seat in (0, 1) for start in (0, 1)]
        for partition, first, last in (
                ('training', before-held_before, after-held_after),
                ('validation', held_before, held_after)):
            for position in range(first, last):
                role, seat, start = strata[position % len(strata)]
                decks = [pair[1-role], pair[1-role]]
                decks[seat] = pair[role]
                index = (round_id-1)*1000+len(rows)
                rows.append(dict(index=index,seed=seed+index,pairing=list(pair),decks=decks,
                    learner_seat=seat,learner_deck=decks[seat],opponent_deck=decks[1-seat],
                    starting_player=f'P{start}',partition=partition))
    return sorted(rows,key=lambda g:(g['index']%20,g['index']))


def schedule(games=1000, *, seed=402610300000, round_id=1):
    if type(games) is not int or type(round_id) is not int or round_id < 1:
        raise ValueError('Round size and round ID must be positive integers')
    if games == 1000:
        return _thousand_game_schedule(seed, round_id)
    if games < 240 or games % 240 or round_id < 1:
        raise ValueError('Round size must be 1000 or a multiple of 240 for exact per-stratum balance')
    per_pair = games // 6
    rows = []
    for pair in combinations_with_replacement(STANDARD_DECKS, 2):
        roles = (0, 1) if pair[0] != pair[1] else (0,)
        per_stratum = per_pair // (len(roles) * 4)
        for role in roles:
            for seat in (0, 1):
                for start in (0, 1):
                    decks = [pair[1-role], pair[1-role]]
                    decks[seat] = pair[role]
                    for repeat in range(per_stratum):
                        index = (round_id-1)*games + len(rows)
                        rows.append(dict(index=index,seed=seed+index,pairing=list(pair),decks=decks.copy(),
                            learner_seat=seat,learner_deck=decks[seat],opponent_deck=decks[1-seat],
                            starting_player=f'P{start}',partition='validation' if repeat%5==0 else 'training'))
    if len(rows) != games: raise AssertionError('Incomplete Standard schedule')
    # Interleave all strata across both persistent runners, without changing IDs.
    return sorted(rows,key=lambda g:(g['index']%max(1,per_pair//8),g['index']))


def _reset_inference(batcher, seed):
    # A timed-out game may leave an inference response in flight. The bridge
    # acknowledges reset only after draining responses from that game.
    batcher.connection.send(('reset', seed))
    while True:
        if not batcher.connection.poll(120): raise TimeoutError('Inference reset timed out')
        if batcher.connection.recv() == ('reset', seed): break


def play_ppo_game(*, seed, index, game, model, batcher, version,
                  opponent_model=None, opponent_batcher=None,
                  opponent_version=None, opponent_sha256=None, **options):
    from .standard_imitation import timed_standard_game
    if (game['seed'],game['index']) != (seed,index):
        raise ValueError('Scheduled game identity mismatch')
    opponent_kind = game.get('opponent_kind', 'scripted')
    if opponent_kind not in ('scripted', 'frozen_neural'):
        raise ValueError('Unknown Standard opponent kind')
    if opponent_kind == 'frozen_neural':
        if opponent_model is None or opponent_batcher is None:
            raise ValueError('Frozen neural opponent requires separate inference')
        if game.get('opponent_checkpoint_sha256') != opponent_sha256 or not opponent_sha256:
            raise ValueError('Frozen neural opponent checkpoint hash mismatch')
        if game.get('opponent_version') != opponent_version or opponent_version is None:
            raise ValueError('Frozen neural opponent version mismatch')
    game = dict(game, behavior_version=version)
    try:
        return timed_standard_game(game, model=model,batcher=batcher,version=version,
            learner_seat=game['learner_seat'],record_ppo=True,
            opponent_model=opponent_model,opponent_batcher=opponent_batcher,
            opponent_version=opponent_version,opponent_sha256=opponent_sha256,**options)
    finally:
        try:
            _reset_inference(batcher,seed)
        finally:
            if opponent_batcher is not None and opponent_batcher is not batcher:
                _reset_inference(opponent_batcher,seed)


def normalization(games):
    trajectories = [t for g in games for t in g['trajectories']]
    count = sum(t['samples'] for t in trajectories)
    if not count: raise ValueError('No PPO training decisions')
    total = sum(t['reward']*t['samples']-t['value_sum'] for t in trajectories)
    squared = sum(t['reward']**2*t['samples']-2*t['reward']*t['value_sum']+t['value_square_sum'] for t in trajectories)
    mean = total/count; std = math.sqrt(max(0.,squared/count-mean*mean))
    return dict(center=mean if std>1e-8 else 0.,scale=max(std,1.) if std>1e-8 else 1.,
        training_choices=count,source='training_partition_only',advantage_baseline='critic')


def validate_collection(rows, planned, version):
    by_id = {g['index']:g for g in rows}
    if len(by_id)!=len(rows) or set(by_id)!={g['index'] for g in planned}:
        raise ValueError('Missing or duplicate scheduled attempts')
    for g in planned:
        row = by_id[g['index']]
        opponent_kind = g.get('opponent_kind', 'scripted')
        if opponent_kind not in ('scripted', 'frozen_neural'):
            raise ValueError('Unknown scheduled Standard opponent kind')
        if opponent_kind == 'frozen_neural':
            digest = g.get('opponent_checkpoint_sha256')
            if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Frozen neural opponent requires a checkpoint SHA256')
            if type(g.get('opponent_version')) is not int or g['opponent_version'] < 0:
                raise ValueError('Frozen neural opponent requires a version')
        if any(row.get(k)!=v for k,v in g.items()) or row.get('behavior_version')!=version:
            raise ValueError('Game schedule or frozen policy version differs')
        if row['status'] not in ('win','loss','draw'):
            raise RuntimeError(row.get('error',row))
        if len(row['trajectories'])>1: raise ValueError('Only the learner may be recorded')
        for t in row['trajectories']:
            reward = 0. if row['status']=='draw' else 1. if row['winner_seat']==g['learner_seat'] else -1.
            if t['player']!=g['learner_seat'] or t['reward']!=reward or t['deck']!=g['learner_deck']:
                raise ValueError('Wrong learner seat or terminal target')
        if not row['trajectories'] and not row.get('adjudicated'):
            raise ValueError('Natural game omitted its PPO records')


def compact_games(rows):
    return [dict(g,actions=None) for g in sorted(rows,key=lambda g:g['index'])]


def cleanup_rollouts(root, committed):
    """Only committed history older than current and previous can be removed."""
    root = Path(root).resolve()
    deleted = []
    for p in root.glob('round-*'):
        if not p.is_dir() or p.is_symlink(): continue
        suffix = p.name.removeprefix('round-')
        if len(suffix)!=4 or not suffix.isdecimal(): continue
        number = int(suffix)
        if number < committed-1:
            if not (p/'committed.json').is_file(): continue
            if p.resolve().parent != root: raise ValueError('Unsafe rollout cleanup path')
            shutil.rmtree(p); deleted.append(number)
    return deleted
