"""In-process RustMage adapter. Rules and hidden information stay in Rust.

Load the upstream ctypes binding from an explicit checkout; no server is used.
The native action list and numeric domains are retained without menu truncation.
Search checkpoints own full hidden state: they belong to the search coordinator,
never a policy observation. Forking does not determinize hidden information.
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

from ..core import (Capabilities, Decision, DecisionToken, Handle, ObjectView,
                    Observation, Option, Outcome, PlayerView, Response)
from ..errors import Disconnected, GameEnded, StaleDecision, UnsupportedInteraction

RUSTMAGE_REVISION = '78d717856af36274018634b54a3fc8cbacf4c829'
OBSERVATION_SCHEMA = 131
CAPABILITIES = Capabilities('rustmage', frozenset((
    'private_views', 'sequential_choices', 'seed', 'concede', 'snapshot', 'restore', 'fork')))
PHASES = {'Pregame': None, 'Main1': 'precombat_main', 'Main2': 'postcombat_main',
          'BeginCombat': 'begin_combat', 'Attackers': 'declare_attackers',
          'Blockers': 'declare_blockers', 'FirstStrikeDamage': 'first_strike_damage',
          'CombatDamage': 'combat_damage', 'EndCombat': 'end_combat'}


def load_binding(checkout=None):
    root = Path(checkout or os.environ.get('RUSTMAGE_ROOT', '.local/rustmage')).resolve()
    path = root / 'python/rustmage.py'
    if not path.is_file():
        raise FileNotFoundError('Build RustMage with scripts/rustmage-local.py or set RUSTMAGE_ROOT: ' + str(path))
    spec = importlib.util.spec_from_file_location('_patchwork_rustmage_binding', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Optional neural dependency: replace only this binding's decoder, never
    # mutate the shared stdlib json module or require neural extras for the API.
    try:
        import orjson
    except ImportError:
        pass
    else:
        module.json = SimpleNamespace(dumps=json.dumps, loads=orjson.loads)
    if module.OBSERVATION_SCHEMA != OBSERVATION_SCHEMA:
        raise UnsupportedInteraction(f'Expected RustMage observation schema {OBSERVATION_SCHEMA}, got {module.OBSERVATION_SCHEMA}')
    return module


def _plain(value):
    if isinstance(value, Mapping): return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_plain(v) for v in value]
    return value


def action_options(raw):
    options = [Option(f'action:{i}', 'finite', a['kind'], {'action': a})
               for i, a in enumerate(raw['actions'])]
    # Representatives in actions are not exhaustive for large numeric domains.
    for field, kind in (('x_choice', 'integer'), ('damage_assignment', 'integer'),
                        ('mana_lot_spend', 'integer'), ('mana_payment', 'mana_payment')):
        if raw.get(field) is not None:
            options.append(Option('domain:' + field, kind, field, raw[field]))
    return tuple(options)


def domain_action(option, value):
    c = option.constraints
    field = option.id.removeprefix('domain:')
    if field == 'mana_payment':
        if (not isinstance(value, (list, tuple)) or len(value) != 6
                or any(type(v) is not int or not lo <= v <= hi
                       for v, lo, hi in zip(value, c['required'], c['available']))
                or sum(v - lo for v, lo in zip(value, c['required'])) != c['generic']):
            raise ValueError('Payment must satisfy the complete six-color domain')
        return {'kind': 'PayCosts', 'payment': list(value)}
    if type(value) is not int:
        raise ValueError('An integer is required')
    if field == 'x_choice':
        if not any(r['minimum'] <= value <= r['maximum'] for r in c['intervals']):
            raise ValueError('X is outside the offered intervals')
        return {'kind': 'ChooseX', 'value': value}
    if not c['minimum'] <= value <= c['maximum']:
        raise ValueError('Value is outside the offered range')
    if field == 'mana_lot_spend':
        if c.get('values') is not None and value not in c['values']:
            raise ValueError('Amount is absent from the exact offered set')
        return {'kind': 'ChooseManaLotSpend', 'amount': value}
    return {'kind': 'AssignDamage', 'target': _plain(c['target']), 'amount': value}


@dataclass(frozen=True)
class Checkpoint:
    session: str
    native_id: int


class RustMageMatch:
    """Owns one native game and two seat-private handles; use close() explicitly.

    Operations on a match run on its owning event-loop thread. Independent matches
    can run on separate threads/processes. Native calls are synchronous and short.
    """
    capabilities = CAPABILITIES

    def __init__(self, first_deck, second_deck=None, *, seed=0, starting_player='P0',
                 checkout=None, library=None, deck_rules='Constructed'):
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError('Seed must be an unsigned 64-bit integer')
        if starting_player not in ('P0', 'P1'):
            raise ValueError('Starting player must be P0 or P1')
        decks = (first_deck, second_deck or first_deck)
        if any(c.sideboard for d in decks for c in d.cards):
            raise UnsupportedInteraction('RustMage match adapter currently accepts mainboards only')
        binding = load_binding(checkout)
        native = binding.Game([[c.name for c in d.cards for _ in range(c.quantity)] for d in decks],
                              seed=seed, starting_player=starting_player,
                              deck_rules=deck_rules, library=library)
        self._initialize(native)
        try: self._raw('P0')  # Check the binary too, not just its Python schema file.
        except BaseException:
            self.close()
            raise

    def _initialize(self, native):
        if os.environ.get('PATCHWORK_OBSERVATION_PIPELINE') == 'rust':
            from ..neural.native_pipeline import enable_native_observations
            enable_native_observations(native)
        self.native_game = native
        self.session = uuid.uuid4().hex
        self.handle = Handle('rustmage', 'game', uuid.uuid4().hex)
        self.closed = False
        self._changed = asyncio.Event()
        self._cache = {}
        self._checkpoints = set()
        self.games = {f'player_{i}': RustMageGame(self, f'P{i}') for i in range(2)}

    def _check(self):
        if self.closed: raise Disconnected('RustMage match is closed')

    def _raw(self, viewer):
        self._check()
        if viewer not in self._cache:
            raw = self.native_game.observe(viewer)
            self._validate_view(raw, viewer)
            self._cache[viewer] = raw
        return self._cache[viewer]

    @staticmethod
    def _validate_view(raw, viewer, *, previous_revision=None):
        if raw['schema'] != OBSERVATION_SCHEMA:
            raise UnsupportedInteraction('RustMage shared library schema does not match the adapter')
        if raw['viewer'] != viewer:
            raise UnsupportedInteraction('RustMage returned a different private viewer')
        if previous_revision is not None and raw['revision'] <= previous_revision:
            raise StaleDecision('RustMage action returned a stale observation')

    def _after_action(self, raw, viewer, previous_revision):
        # The native state has advanced even if response validation fails.
        self._invalidate()
        self._validate_view(raw, viewer, previous_revision=previous_revision)
        self._cache[viewer] = raw

    def _invalidate(self):
        self._cache.clear()
        for game in self.games.values(): game._observation = None
        self._changed.set()

    def checkpoint(self):
        self._check()
        item = Checkpoint(self.session, self.native_game.snapshot())
        self._checkpoints.add(item)
        return item

    def _validate_checkpoint(self, checkpoint):
        self._check()
        if checkpoint not in self._checkpoints:
            raise ValueError('Checkpoint is foreign or has been released')

    def restore(self, checkpoint):
        self._validate_checkpoint(checkpoint)
        self.native_game.restore(checkpoint.native_id)
        self._invalidate()  # Native revision increases; old decisions stay invalid.

    def drop_checkpoint(self, checkpoint):
        self._validate_checkpoint(checkpoint)
        self.native_game.drop_snapshot(checkpoint.native_id)
        self._checkpoints.remove(checkpoint)

    def fork(self):
        self._check()
        result = object.__new__(RustMageMatch)
        result._initialize(self.native_game.clone())
        return result

    def close(self):
        if not self.closed:
            self.native_game.close()
            self.closed = True
            self._invalidate()
            self._checkpoints.clear()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


class RustMageGame:
    capabilities = CAPABILITIES

    def __init__(self, match, player):
        self.match, self.player = match, player
        self.handle = match.handle
        self._observation = None

    async def snapshot(self):
        raw = self.match._raw(self.player)
        if self._observation is not None: return self._observation
        def player(value): return Handle('rustmage', 'player', value)
        actor = player(self.player)
        players = tuple(PlayerView(player(f'P{i}'), f'P{i}', p['life'], p['hand_count'],
                                   p['library_count'], {'poison': p['poison_counters']}, p)
                        for i, p in enumerate(raw['players']))
        objects = {}
        for obj in raw['objects']:
            handle = Handle('rustmage', 'object', str(obj['id']))
            objects[handle] = ObjectView(handle, obj['name'], obj['zone'].lower(),
                                         player(obj['controller']), obj)
        terminal = raw['outcome'] is not None
        outcome = None
        if terminal:
            draw = raw['outcome'] == 'Draw'
            outcome = Outcome(False if draw else raw['outcome']['Winner'] == self.player,
                              draw, {'result': raw['outcome']})
        decision = None
        if not terminal and raw['actor'] == self.player:
            token = DecisionToken('rustmage', self.match.session, self.handle, actor, raw['revision'])
            decision = Decision(token, raw['decision'], '', action_options(raw),
                                {'schema': raw['schema']})
        self._observation = Observation('rustmage', self.handle, actor, raw['revision'], players,
            objects, decision, terminal, outcome, raw, turn=raw['turn'],
            phase=PHASES.get(raw['step'], raw['step'].lower()), active_player=player(raw['active']),
            priority_player=player(raw['actor']) if raw['decision'] == 'priority' and raw['actor'] else None)
        return self._observation

    async def wait_for_decision(self, *, timeout=None):
        async def wait():
            while True:
                obs = await self.snapshot()
                if obs.terminal or obs.decision is not None: return obs
                self.match._changed.clear()
                await self.match._changed.wait()
        return await asyncio.wait_for(wait(), timeout)

    async def submit(self, token, response: Response):
        obs = await self.snapshot()
        if obs.decision is None or obs.decision.token != token:
            raise StaleDecision('RustMage decision is foreign, stale or already submitted')
        option = next((o for o in obs.decision.options if o.id == response.option), None)
        if option is None: raise UnsupportedInteraction('Option is not offered')
        raw = self.match._raw(self.player)
        if option.id.startswith('domain:'):
            action = domain_action(option, response.value)
            updated = self.match.native_game.act(raw, action)
        else:
            if response.value is not None: raise ValueError('Finite actions take no additional value')
            updated = self.match.native_game.step(raw, int(option.id.split(':')[1]))
        self.match._after_action(updated, self.player, raw['revision'])

    async def concede(self):
        raw = self.match._raw(self.player)
        if raw['outcome'] is not None: raise GameEnded(self.handle.value, raw['outcome'])
        updated = self.match.native_game.request('act', actor=self.player, revision=raw['revision'],
                                      action={'kind': 'Concede'}, viewer=self.player)['observation']
        self.match._after_action(updated, self.player, raw['revision'])

    async def wait_for_result(self, *, timeout=None):
        async def wait():
            while True:
                obs = await self.snapshot()
                if obs.terminal: return obs
                self.match._changed.clear()
                await self.match._changed.wait()
        return await asyncio.wait_for(wait(), timeout)
