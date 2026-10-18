"""Internal borrowed views of a worker-owned, already-redacted native response.

These records deliberately do not recursively copy/freeze the JSON tree. They
are for the synchronous training collector, whose policies only read them.
The public RustMageGame.snapshot() contract remains deeply immutable. Each
response owns a new tree; advancing a game never mutates an older view.
"""
from typing import NamedTuple

from ..backends.rustmage import PHASES, action_options
from ..core import Handle, DecisionToken, Decision, Outcome


class _Object(NamedTuple):
    id: Handle
    name: str
    zone: str
    controller: Handle
    properties: dict


class _Player(NamedTuple):
    id: Handle
    name: str
    life: int
    hand_count: int
    library_count: int
    native: dict


class _Observation(NamedTuple):
    backend: str
    game: Handle
    actor: Handle
    revision: int
    players: tuple
    objects: dict
    decision: Decision | None
    terminal: bool
    outcome: Outcome | None
    native: dict
    turn: int
    phase: str | None
    active_player: Handle
    priority_player: Handle | None


PLAYERS = {p: Handle('rustmage', 'player', p) for p in ('P0', 'P1')}


def learning_observation(match, viewer):
    raw = match._raw(viewer)
    actor = PLAYERS[viewer]
    players = tuple(_Player(PLAYERS[f'P{i}'], f'P{i}', p['life'], p['hand_count'],
                           p['library_count'], p) for i, p in enumerate(raw['players']))
    objects = {}
    for obj in raw['objects']:
        handle = Handle('rustmage', 'object', str(obj['id']))
        objects[handle] = _Object(handle, obj['name'], obj['zone'].lower(),
                                 PLAYERS[obj['controller']], obj)
    outcome = None
    if raw['outcome'] is not None:
        draw = raw['outcome'] == 'Draw'
        outcome = Outcome(False if draw else raw['outcome']['Winner'] == viewer,
                          draw, {'result': raw['outcome']})
    decision = None
    if outcome is None and raw['actor'] == viewer:
        token = DecisionToken('rustmage', match.session, match.handle, actor, raw['revision'])
        decision = Decision(token, raw['decision'], '', action_options(raw), {'schema': raw['schema']})
    return _Observation('rustmage', match.handle, actor, raw['revision'], players, objects,
        decision, outcome is not None, outcome, raw, raw['turn'],
        PHASES.get(raw['step'], raw['step'].lower()), PLAYERS[raw['active']],
        PLAYERS[raw['actor']] if raw['decision'] == 'priority' and raw['actor'] else None)
