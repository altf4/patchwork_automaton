"""Predeclared opponent selection for the checkpoint-200 self-play fork."""
from collections import Counter


def snapshot_round_for_round(round_id, *, base_round=200):
    """Use one archived policy per round, adding durable ten-round snapshots."""
    if type(round_id) is not int or not base_round < round_id <= base_round + 50:
        raise ValueError('Self-play round is outside the declared 50-round fork')
    eligible = [base_round - 50, base_round]
    eligible.extend(range(base_round + 10, round_id, 10))
    return eligible[(round_id - base_round) % len(eligible)]


def assign_opponents(planned, *, round_id, mode, checkpoint_round=None,
                     checkpoint_sha256=None):
    """Keep exactly 20% scripted games in each whole-game partition.

    The ordinary scheduler already rotates six pairings and learner roles.
    Selecting every fifth game within each partition also spreads opponent
    types across those strata, with a round-dependent phase for cumulative
    balance. This function never changes a game ID, seed, deck or seat.
    """
    if mode not in ('scripted', 'frozen_pool'):
        raise ValueError('Unknown opponent mix')
    if mode == 'frozen_pool':
        if checkpoint_round != snapshot_round_for_round(round_id):
            raise ValueError('Opponent snapshot differs from predeclared rotation')
        if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64 or any(
                c not in '0123456789abcdef' for c in checkpoint_sha256):
            raise ValueError('Opponent checkpoint SHA256 required')
    elif checkpoint_round is not None or checkpoint_sha256 is not None:
        raise ValueError('Script-only arm cannot receive a neural checkpoint')
    result = [dict(game) for game in planned]
    counts = Counter()
    for partition in ('training', 'validation'):
        positions = sorted((i for i,g in enumerate(result) if g['partition'] == partition),
                           key=lambda i: result[i]['index'])
        for ordinal, position in enumerate(positions):
            game = result[position]
            scripted = mode == 'scripted' or (ordinal + round_id) % 5 == 0
            game['opponent_kind'] = 'scripted' if scripted else 'frozen_neural'
            counts[(partition, game['opponent_kind'])] += 1
            if not scripted:
                game['opponent_version'] = checkpoint_round
                game['opponent_checkpoint_sha256'] = checkpoint_sha256
    if mode == 'frozen_pool' and len(planned) == 1000 and counts != Counter({
            ('training', 'scripted'): 160,
            ('training', 'frozen_neural'): 640,
            ('validation', 'scripted'): 40,
            ('validation', 'frozen_neural'): 160}):
        raise AssertionError('Opponent mix changed declared 80/20 quotas')
    return result
