"""Matched collection schedules for the checkpoint-100 opponent-recipe fork.

Local checkpoint zero is an exact continuation of identity checkpoint 100.
External versions live above 1000, outside the local 0--50 namespace. The
schedule changes only the opponent policy; game identities, decklists, learner
seats, starting players and whole-game partitions remain matched between arms.
"""

from collections import defaultdict, deque
from itertools import product
import random

from .hartford_schedule import _decks, _integer, _row, _seed_bounds, _shuffled


EXTERNAL_BASE = 1000
PARENT_ROUND = 100
ROUNDS = 50
HISTORICAL_CHECKPOINTS = (25, 50, 75)
HISTORICAL_VERSIONS = tuple(EXTERNAL_BASE + n for n in HISTORICAL_CHECKPOINTS)
SUPPORTED_SCRIPT_DECKS = ('mono-green-landfall', 'izzet-prowess', 'bw-lifegain')
RECIPES = ('current', 'diverse')


def _predecessor_version(local_version):
    return local_version if local_version >= 0 else EXTERNAL_BASE + PARENT_ROUND + local_version


def external_versions(recipe):
    """Return every parent checkpoint version needed by one complete arm."""
    if recipe not in RECIPES:
        raise ValueError('Unknown Hartford A/B opponent recipe')
    return ((1097, 1098, 1099) if recipe == 'current' else HISTORICAL_VERSIONS)


def _script_indices(rows, *, seed, round_id):
    """Select four scripted slots per eligible cell with exact role quotas.

This small integral flow preserves 96 training / 24 held-out scripted games,
balanced over all four learner-seat / starting-player combinations. Selecting
only current-policy slots also preserves each historical policy's 80/20 split.
All choices and tie breaks use a private, explicitly seeded RNG.
"""
    latest = round_id - 1
    candidates = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['opponent_deck'] in SUPPORTED_SCRIPT_DECKS and row['opponent_version'] == latest:
            key = (row['learner_deck'], row['opponent_deck'])
            role = (row['partition'], row['learner_seat'], row['starting_player'])
            candidates[key][role].append(row['index'])
    if len(candidates) != 30:
        raise ValueError('Diverse recipe requires all three supported scripted opponent decks')
    rng = random.Random(f'hartford-ab-script:{seed}:{round_id}')
    cells = sorted(candidates)
    rng.shuffle(cells)
    roles = [(partition, seat, f'P{start}') for partition in ('training', 'validation')
             for seat in (0, 1) for start in (0, 1)]
    rng.shuffle(roles)
    source, sink = 0, 1
    cell_nodes = {cell: n + 2 for n, cell in enumerate(cells)}
    role_nodes = {role: n + 2 + len(cells) for n, role in enumerate(roles)}
    graph = [[] for _ in range(2 + len(cells) + len(roles))]

    def edge(first, second, capacity):
        forward = [second, capacity, len(graph[second])]
        reverse = [first, 0, len(graph[first])]
        graph[first].append(forward)
        graph[second].append(reverse)
        return forward

    assignments = []
    for cell in cells:
        edge(source, cell_nodes[cell], 4)
        for role in roles:
            indices = candidates[cell].get(role, [])
            if indices:
                rng.shuffle(indices)
                arc = edge(cell_nodes[cell], role_nodes[role], len(indices))
                assignments.append((arc, indices))
    for role in roles:
        edge(role_nodes[role], sink, 24 if role[0] == 'training' else 6)
    total = 0
    while True:
        previous = [None] * len(graph)
        previous[source] = (source, -1)
        queue = deque([source])
        while queue and previous[sink] is None:
            current = queue.popleft()
            for n, arc in enumerate(graph[current]):
                target, capacity, _ = arc
                if capacity and previous[target] is None:
                    previous[target] = (current, n)
                    queue.append(target)
        if previous[sink] is None:
            break
        amount, node = 120, sink
        while node != source:
            origin, n = previous[node]
            amount = min(amount, graph[origin][n][1])
            node = origin
        node = sink
        while node != source:
            origin, n = previous[node]
            arc = graph[origin][n]
            arc[1] -= amount
            graph[node][arc[2]][1] += amount
            node = origin
        total += amount
    if total != 120:
        raise ValueError('Cannot balance scripted opponents without changing the matched schedule')
    return {index for arc, indices in assignments for index in indices[:len(indices) - arc[1]]}


def training_schedule(deck_ids, *, round_id, seed, recipe):
    """Return 1000 matched games for one of the 50 additional local rounds.

    Current: 700 latest and 100 each against its three immediate predecessors.
    The first three rounds draw missing predecessors from the immutable parent
    history, so a continuation never re-enters the fresh-run startup quotas.

    Diverse: 580 latest, 100 each against parent checkpoints 25/50/75, and 120
    scripted games. Each cell retains eight training and two held-out games.
    Every cell receives each historical policy once; each of the 30 supported
    script-opponent cells receives four scripts and three latest-policy games.
    """
    decks = _decks(deck_ids)
    _integer(round_id, 'Round ID', minimum=1)
    _integer(seed, 'Seed')
    if round_id > ROUNDS:
        raise ValueError('Hartford A/B permits exactly 50 additional rounds')
    if recipe not in RECIPES:
        raise ValueError('Unknown Hartford A/B opponent recipe')
    if recipe == 'diverse' and not set(SUPPORTED_SCRIPT_DECKS).issubset(decks):
        raise ValueError('Diverse recipe requires all three supported scripted opponent decks')
    _seed_bounds(seed + (round_id - 1) * 1000, seed + round_id * 1000 - 1)
    latest = round_id - 1
    older = (tuple(_predecessor_version(latest - age) for age in (1, 2, 3))
             if recipe == 'current' else HISTORICAL_VERSIONS)
    rows = []
    for cell, (learner, opponent) in enumerate(product(decks, repeat=2)):
        historical = {(cell + round_id + 3 * slot) % 10: version
                      for slot, version in enumerate(older)}
        for position in range(10):
            index = (round_id - 1) * 1000 + cell * 10 + position
            if position < 8:
                seat, start = divmod((position + round_id) % 4, 2)
                partition = 'training'
            else:
                seat, start = position - 8, (round_id + cell // 50) % 2
                partition = 'validation'
            rows.append(_row(decks, learner, opponent, index=index, seed=seed + index,
                             seat=seat, start=start, partition=partition,
                             behavior_version=latest,
                             opponent_version=historical.get(position, latest)))
    if recipe == 'diverse':
        scripts = _script_indices(rows, seed=seed, round_id=round_id)
        for row in rows:
            if row['index'] in scripts:
                row.update(opponent_kind='scripted', opponent_version=None)
    return _shuffled(rows, seed=seed, identity=f'training:{round_id}')
