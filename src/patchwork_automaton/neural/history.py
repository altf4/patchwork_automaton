"""Bounded, actor-local numerical changes between decision snapshots."""
from collections import deque
import threading

import numpy as np

from ..errors import EventOverflow
from ..semantics import lookup_visible
from .features import ZONES, numeric, onehot

EVENTS = ('appear', 'disappear', 'move', 'life', 'hand', 'library', 'turn')
HISTORY_WIDTH = len(EVENTS) + 3 + 2 * len(ZONES) + 6


class History:
    def __init__(self, registry, limit=4096):
        self.registry, self.limit = registry, limit
        # Descriptor IDs are internal lookup keys when a registry supplies
        # transferable traits. They must never become learned card identities.
        self._trait_rows = getattr(registry, 'trait_rows', None)
        self._trait_width = getattr(registry, 'trait_width', 0) if self._trait_rows is not None else 0
        if self._trait_rows is not None:
            self._trait_rows = tuple(tuple(row) for row in self._trait_rows)
            if not self._trait_rows or any(len(row) != self._trait_width for row in self._trait_rows):
                raise ValueError('History trait rows must include a fixed-width unknown row')
        self.width = HISTORY_WIDTH + self._trait_width
        self._lock = threading.Lock()
        self._previous = {}
        self._pending = {}

    def reset(self):
        with self._lock:
            self._previous.clear()
            self._pending.clear()

    @staticmethod
    def key(obs):
        # Handles include backend and game UUID; actor identity separates seats.
        return obs.game, obs.actor

    def _identity(self, obj):
        if self._trait_rows is None:
            card = lookup_visible(obj, self.registry).id
            return card, card
        if any(obj.properties.get(key) for key in ('hideInfo', 'faceDown', 'face_down')):
            return 0, (0, None)
        # RustMage's effective card key also identifies tokens and copied forms.
        # A present but unknown key must not borrow traits from a display name.
        name = obj.properties.get('card')
        if not isinstance(name, str):
            name = obj.name
        descriptor = self.registry.get(name) if name else None
        card = descriptor.id if descriptor is not None else 0
        # Preserve changes between unknown, visible forms without emitting text
        # or conflating them merely because both have descriptor ID zero.
        return card, card if card else (0, name)

    def observe(self, obs):
        if len(obs.players) != 2: return
        key = self.key(obs)
        with self._lock:
            previous = self._previous.get(key)
            if previous is not None and previous['revision'] >= obs.revision: return
            pending = self._pending.setdefault(key, deque())
            objects = {}
            for handle, obj in obs.objects.items():
                card, identity = self._identity(obj)
                side = 0 if obj.controller == obs.actor else 1 if obj.controller else 2
                objects[handle] = (card, obj.zone, side, identity)
            players = {p.id: (p.life, p.hand_count, p.library_count) for p in obs.players}
            def event(kind, *, card=None, side=2, old='unknown', new='unknown', value=None, delta=None):
                if len(pending) >= self.limit:
                    raise EventOverflow('Neural history overflow; consume decisions or increase the explicit limit')
                row = onehot(kind, EVENTS) + onehot(side, range(3))
                row += onehot(old, ZONES) + onehot(new, ZONES)
                row += numeric(value) + numeric(delta) + numeric(obs.turn)
                if self._trait_rows is not None:
                    row += self._trait_rows[card] if card is not None else (0.,) * self._trait_width
                pending.append((0 if self._trait_rows is not None else card or 0, row))
            old_objects = previous['objects'] if previous else {}
            for handle, (card, zone, side, identity) in objects.items():
                old = old_objects.get(handle)
                if old is None or old[3] != identity:
                    event('appear', card=card, side=side, new=zone)
                elif old[1] != zone:
                    event('move', card=card, side=side, old=old[1], new=zone)
            for handle, (card, zone, side, identity) in old_objects.items():
                if handle not in objects: event('disappear', card=card, side=side, old=zone)
            for handle, values in players.items():
                old = previous['players'].get(handle, (None,) * 3) if previous else (None,) * 3
                for kind, value, before in zip(('life', 'hand', 'library'), values, old):
                    if value != before:
                        event(kind, side=int(handle != obs.actor), value=value,
                              delta=value - before if value is not None and before is not None else None)
            if previous is None or obs.turn != previous['turn']: event('turn', value=obs.turn)
            self._previous[key] = dict(revision=obs.revision, objects=objects, players=players, turn=obs.turn)

    def take(self, obs):
        with self._lock:
            pending = list(self._pending.pop(self.key(obs), ()))
        size = max(1, len(pending))
        cards = np.zeros(size, np.int32)
        rows = np.zeros((size, self.width), np.float32)
        mask = np.zeros(size, bool)
        for i, (card, row) in enumerate(pending):
            cards[i], rows[i], mask[i] = card, row, True
        return {'history_cards': cards, 'history': rows, 'history_mask': mask}
