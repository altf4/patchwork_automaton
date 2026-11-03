"""Deterministic, whole-game schedules for the ten-deck Hartford experiment.

The caller supplies the immutable deck order and distinct training/evaluation
seed bases. Neither scheduler reads a global deck registry or changes a global
random generator. Checkpoint hashes are attached by the experiment driver.
"""

from itertools import product
import random


MAX_GAME_SEED = (2 ** 63 - 2) // 2


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _decks(deck_ids):
    decks = tuple(deck_ids)
    if len(decks) != 10 or any(not isinstance(d, str) or not d for d in decks):
        raise ValueError('Hartford requires exactly ten nonempty deck IDs')
    if len(set(decks)) != len(decks):
        raise ValueError('Hartford deck IDs must be unique')
    return decks


def _cell_size(games_per_cell):
    _integer(games_per_cell, 'Games per cell', minimum=4)
    if games_per_cell % 4:
        raise ValueError('Games per cell must balance all four seat/start combinations')
    return games_per_cell


def _seed_bounds(first, last):
    if last > MAX_GAME_SEED:
        raise ValueError('Game seeds must keep both player seeds below 2**63')
    return first, last


def training_seed_bounds(*, rounds, seed):
    """Inclusive bounds for every scheduled training/holdout game seed."""
    _integer(rounds, 'Rounds', minimum=1)
    _integer(seed, 'Seed')
    return _seed_bounds(seed, seed + 1000 * rounds - 1)


def evaluation_game_count(rounds, games_per_cell=40):
    """Total games through the requested round, including startup windows."""
    _integer(rounds, 'Rounds', minimum=1)
    _cell_size(games_per_cell)
    comparisons = 1 if rounds == 1 else 3 * rounds - 3
    return comparisons * 100 * games_per_cell


def evaluation_seed_bounds(*, rounds, games_per_cell=40, seed):
    """Inclusive seed bounds for all rolling checkpoint evaluation games."""
    _integer(rounds, 'Rounds', minimum=1)
    _integer(seed, 'Seed')
    _cell_size(games_per_cell)
    last = _pair(_pair(rounds, rounds - 1), _pair(99, games_per_cell - 1))
    return _seed_bounds(seed + 1, seed + last)


def previous_checkpoints(checkpoint_round):
    """Return the preceding three available checkpoints, newest first."""
    _integer(checkpoint_round, 'Checkpoint round')
    return tuple(range(checkpoint_round - 1, max(-1, checkpoint_round - 4), -1))


def _row(decks, learner, opponent, *, index, seed, seat, start, partition,
         behavior_version, opponent_version):
    seating = [opponent, opponent]
    seating[seat] = learner
    return dict(index=index, seed=seed, pairing=sorted((learner, opponent), key=decks.index),
                decks=seating, learner_seat=seat, learner_deck=learner,
                opponent_deck=opponent, starting_player=f'P{start}', partition=partition,
                deck_catalog='hartford', behavior_version=behavior_version,
                opponent_kind='frozen_neural', opponent_version=opponent_version)


def _shuffled(rows, *, seed, identity):
    random.Random(f'hartford:{seed}:{identity}').shuffle(rows)
    return rows


def training_schedule(deck_ids, *, round_id, seed):
    """Schedule 1,000 games: eight training and two holdouts in every cell.

    The latest checkpoint supplies 70% of opponents, with the other 30% split
    equally among up to three predecessors. With no predecessor, all opponents
    use checkpoint zero. Quotas hold separately in the 800/200 partitions. Once
    three predecessors exist, every directed matchup has seven latest-policy
    games and one game against each predecessor.

    Training covers each seat/start combination twice per cell. Each holdout
    cell balances learner seat and play/draw; alternating the starting seat
    across rounds covers all four combinations over two rounds.
    """
    decks = _decks(deck_ids)
    _integer(round_id, 'Round ID', minimum=1)
    _integer(seed, 'Seed')
    _seed_bounds(seed + (round_id - 1) * 1000, seed + round_id * 1000 - 1)
    latest = round_id - 1
    older = previous_checkpoints(latest)
    rows = []
    for cell, (learner, opponent) in enumerate(product(decks, repeat=2)):
        historical = {}
        for slot in range(3 if older else 0):
            position = (cell + round_id + 3 * slot) % 10
            # Two predecessors divide the three historical slots 1/2 per cell,
            # reversing this allocation in the next row of cells. The ten-position
            # rotation also gives exact 120/120 and 30/30 partition quotas.
            which = (slot + cell // 10) % len(older) if len(older) == 2 else slot % len(older)
            historical[position] = older[which]
        for position in range(10):
            index = (round_id - 1) * 1000 + cell * 10 + position
            if position < 8:
                seat, start = divmod((position + round_id) % 4, 2)
                partition = 'training'
            else:
                # Decouple this rotation from the historical-slot rotation so
                # opponent age is not correlated with playing first or second.
                seat, start = position - 8, (round_id + cell // 50) % 2
                partition = 'validation'
            rows.append(_row(decks, learner, opponent, index=index, seed=seed + index,
                             seat=seat, start=start, partition=partition,
                             behavior_version=latest,
                             opponent_version=historical.get(position, latest)))
    return _shuffled(rows, seed=seed, identity=f'training:{round_id}')


def _pair(first, second):
    """Cantor pairing gives collision-free IDs without fixed round/game limits."""
    total = first + second
    return total * (total + 1) // 2 + second


def evaluation_schedule(deck_ids, *, checkpoint_round, opponent_round,
                        games_per_cell=40, seed):
    """Schedule all directed matchups against one prior checkpoint.

    Every cell balances focal seat and starting player. IDs and seed offsets
    encode both checkpoints, the cell, and its repetition; they remain unique
    across evaluation points and stable if the number of repetitions changes.
    Evaluation seeds must have a separate base from training seeds.
    """
    decks = _decks(deck_ids)
    _integer(checkpoint_round, 'Checkpoint round', minimum=1)
    _integer(opponent_round, 'Opponent round')
    _cell_size(games_per_cell)
    _integer(seed, 'Seed')
    if opponent_round not in previous_checkpoints(checkpoint_round):
        raise ValueError('Evaluation opponent must be one of the preceding three checkpoints')
    comparison = _pair(checkpoint_round, opponent_round)
    _seed_bounds(seed + _pair(comparison, 0),
                 seed + _pair(comparison, _pair(99, games_per_cell - 1)))
    rows = []
    for cell, (learner, opponent) in enumerate(product(decks, repeat=2)):
        for repeat in range(games_per_cell):
            index = _pair(comparison, _pair(cell, repeat))
            rows.append(_row(decks, learner, opponent, index=index, seed=seed + index,
                             seat=repeat % 2, start=(repeat // 2) % 2,
                             partition='evaluation', behavior_version=checkpoint_round,
                             opponent_version=opponent_round))
    return _shuffled(rows, seed=seed, identity=f'evaluation:{checkpoint_round}:{opponent_round}')
