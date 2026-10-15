"""Deterministic, deliberately weak, XMage Modern Burn opponent.

Uses only the actor's observation. Labels and cost symbols are interpreted by
this script, never supplied to a neural model. Tie breaks use visible card name
and advertised option ID. Identical observations produce identical responses.
"""
from collections.abc import Mapping
from importlib.resources import files
import json
import re

from ..core import CardSpec, DeckSpec, Response
from ..errors import UnsupportedInteraction


DECK_METADATA = json.loads(files('patchwork_automaton').joinpath('data/modern-burn-2026-10-04.json').read_text())


def modern_burn_deck():
    """Exact published mainboard/sideboard; no live network lookup needed."""
    if sum(c['quantity'] for c in DECK_METADATA['mainboard']) != 60:
        raise ValueError('Baseline requires exactly 60 mainboard cards')
    return DeckSpec(tuple(CardSpec({'Wear/Tear': 'Wear // Tear'}.get(c['name'], c['name']), c['quantity'], sideboard=side)
        for side, section in ((False, 'mainboard'), (True, 'sideboard'))
        for c in DECK_METADATA[section]), DECK_METADATA['name'])


SPELL_ORDER = ('Goblin Guide', 'Monastery Swiftspear', 'Eidolon of the Great Revel',
    'Boltwave', 'Lava Spike', 'Lightning Bolt', 'Lightning Helix', 'Boros Charm',
    'Skewer the Critics', 'Rift Bolt', 'Skullcrack', 'Searing Blaze')
LAND_ORDER = ('Inspiring Vantage', 'Sacred Foundry', 'Sunbaked Canyon', 'Mountain',
              'Barbarian Ring', 'Elegant Parlor', 'Arid Mesa', 'Bloodstained Mire', 'Wooded Foothills')
FETCHES = frozenset(('Arid Mesa', 'Bloodstained Mire', 'Wooded Foothills'))
PRODUCES = {'Mountain': ('R',), 'Barbarian Ring': ('R',), 'Sacred Foundry': ('R', 'W'),
            'Inspiring Vantage': ('R', 'W'), 'Sunbaked Canyon': ('R', 'W'), 'Elegant Parlor': ('R', 'W')}


def text(value):
    return re.sub(r'<[^>]*>', '', str(value)).lower()


def object_id(option):
    return option.id.removeprefix('object:') if option.kind == 'uuid' else None


def rank(name, names):
    return names.index(name) if name in names else len(names)


class BurnBaseline:
    def __init__(self, *, skip_land_every=4):
        if type(skip_land_every) is not int or skip_land_every < 0:
            raise ValueError('Land-skip interval must be a nonnegative integer')
        self.skip_land_every = skip_land_every
        self.reset()

    def reset(self):
        self._turns = {}
        self._attacked = set()
        self._pending = {}
        self._payment = {}
        self._failed = {}
        self._cached = {}

    @staticmethod
    def _key(obs):
        return obs.game, obs.actor

    def _objects(self, obs):
        # Search/discard selectors may include objects outside the ordinary view.
        objects = {h.value: (o.name, o.properties, o.zone, o.controller) for h, o in obs.objects.items()}
        for field in ('cardsView1', 'cardsView2'):
            cards = obs.decision.native.get(field) or {}
            if isinstance(cards, Mapping):
                for key, card in cards.items():
                    if not isinstance(card, Mapping): continue
                    hidden = card.get('hideInfo') or card.get('faceDown')
                    objects[str(card.get('id') or key)] = (None if hidden else card.get('name'), card, 'selection', None)
        return objects

    @staticmethod
    def _signature(obs):
        return (tuple(sorted((h.value, o.properties.get('tapped')) for h, o in obs.objects.items()
                             if o.zone == 'battlefield' and o.controller == obs.actor)),
                tuple(sorted((p.id.value, tuple(sorted((p.native.get('manaPool') or {}).items()))) for p in obs.players)))

    def choose(self, obs):
        if obs.backend != 'xmage' or obs.decision is None or not obs.decision.options:
            raise UnsupportedInteraction('Burn baseline needs an offered XMage decision')
        key = self._key(obs)
        cached = self._cached.get(key)
        if cached and cached[0] == obs.decision.token: return cached[1]
        turns = self._turns.setdefault(key, [])
        if obs.active_player == obs.actor and obs.turn is not None and obs.turn not in turns:
            turns.append(obs.turn)
        response = self._choose(obs, key, len(turns))
        if not any(o.id == response.option for o in obs.decision.options):
            raise ValueError('Baseline attempted a nonadvertised option')
        self._cached[key] = obs.decision.token, response
        return response

    def _choose(self, obs, key, own_turn):
        decision = obs.decision
        options = tuple(sorted(decision.options, key=lambda o: (o.label, o.id)))
        by_id = {o.id: o for o in options}
        objects = self._objects(obs)
        offered = [o for o in options if o.kind == 'uuid']
        native_options = decision.native.get('options') or {}
        message = text(decision.message)
        kind = decision.kind
        def name(option): return objects.get(object_id(option), (option.label,))[0] or ''
        def response(option, value=None): return Response(option.id, value)
        def no():
            if 'no' not in by_id: raise UnsupportedInteraction('No offered completion/cancel choice')
            return response(by_id['no'])

        if kind == 'GAME_ASK':
            keep = next((o for o in options if text(o.label) == 'keep'), None)
            if keep: return response(keep)  # Keeps even terrible opening hands.
            # Pays shock-land life and accepts optional triggers without analysis.
            return response(by_id.get('yes') or options[0])

        if kind == 'GAME_SELECT':
            if 'possibleAttackers' in native_options:
                turn_key = key, obs.turn
                if 'special' in by_id and turn_key not in self._attacked:
                    self._attacked.add(turn_key)
                    return response(by_id['special'])  # Blind all-out attack.
                return no()
            if 'possibleBlockers' in native_options:
                return no()  # Deliberate weakness: never block.
            # All discretionary spells/abilities are used in our own main phase.
            if obs.active_player != obs.actor or obs.phase not in ('precombat_main', 'postcombat_main'):
                return no()
            if any(o.zone == 'stack' for o in obs.objects.values()): return no()
            self._pending.pop(key, None)
            playable = (obs.native.get('canPlayObjects') or {}).get('objects') or {}
            lands = [o for o in offered if (playable.get(object_id(o)) or {}).get('basicPlayAbilities')]
            if lands and not (self.skip_land_every and own_turn and own_turn % self.skip_land_every == 0):
                return response(min(lands, key=lambda o: (rank(name(o), LAND_ORDER), o.id)))
            # Fetches are activated only on our own main phase, also wasteful.
            fetches = [o for o in offered if name(o) in FETCHES and objects.get(object_id(o), (None, None, None))[2] == 'battlefield']
            if fetches:
                chosen = min(fetches, key=lambda o: (name(o), o.id))
                self._pending[key] = object_id(chosen)
                return response(chosen)
            signature = self._signature(obs)
            enemy_creature = any(o.zone == 'battlefield' and o.controller not in (None, obs.actor)
                                 and 'CREATURE' in (o.properties.get('cardTypes') or ())
                                 for o in obs.objects.values())
            spells = [o for o in offered if name(o) in SPELL_ORDER
                      and objects.get(object_id(o), (None, None, None))[2] == 'hand'
                      and (name(o) != 'Searing Blaze' or enemy_creature)
                      and self._failed.get((key, object_id(o))) != signature]
            if spells:
                chosen = min(spells, key=lambda o: (rank(name(o), SPELL_ORDER), o.id))
                self._pending[key] = object_id(chosen)
                return response(chosen)
            return no()

        if kind == 'GAME_TARGET':
            if 'starting player' in message:
                target = next((o for o in offered if object_id(o) == obs.actor.value), None)
                if target: return response(target)
            # Bottom/discard weakest cards; extra lands first. Opponent targets
            # are otherwise always preferred to creatures, even when removal wins.
            if any(word in message for word in ('discard', 'bottom')):
                if offered:
                    return response(min(offered, key=lambda o: (0 if name(o) in LAND_ORDER else 1,
                                                              -rank(name(o), SPELL_ORDER), name(o), o.id)))
            opponents = {p.id.value for p in obs.players if p.id != obs.actor}
            target = next((o for o in offered if object_id(o) in opponents), None)
            if target: return response(target)
            # Fetching is deterministic: first Foundry, then a Mountain.
            fetch_targets = [o for o in offered if name(o) in ('Sacred Foundry', 'Mountain', 'Elegant Parlor')]
            if fetch_targets:
                return response(min(fetch_targets, key=lambda o: (rank(name(o), ('Sacred Foundry', 'Mountain', 'Elegant Parlor')), o.id)))
            enemy_objects = [o for o in offered if objects.get(object_id(o), (None, None, None, None))[3] not in (None, obs.actor)]
            if enemy_objects: return response(min(enemy_objects, key=lambda o: (name(o), o.id)))
            if offered: return response(min(offered, key=lambda o: (name(o), o.id)))
            return self._cancel(obs, key, by_id)

        if kind in ('GAME_PLAY_MANA', 'GAME_PLAY_XMANA'):
            # Only the unpaid-cost prefix is parsed; HTML card names are ignored.
            cost = decision.message.split('<', 1)[0]
            colors = tuple(c for c in ('W', 'R', 'B', 'U', 'G') if '{' + c + '}' in cost)
            self._payment[key] = colors
            from ..backends.xmage_prompts import prompt_metadata
            debt = prompt_metadata(kind, decision.native).get('mana_cost') or {}
            pool = [o for o in options if o.constraints.get('action_kind') == 'spend_mana'
                    and (debt.get(str(o.constraints.get('mana_color', '')).lower(), 0) > 0
                         or debt.get('generic', 0) > 0)]
            if pool:
                return response(min(pool, key=lambda o: (
                    not debt.get(o.constraints['mana_color'].lower(), 0), o.id)))
            sources = [o for o in offered if name(o) in PRODUCES
                       and not objects.get(object_id(o), (None, {}))[1].get('tapped', True)]
            if sources:
                wanted = colors[0] if colors else 'R'
                source = min(sources, key=lambda o: (wanted not in PRODUCES[name(o)],
                    len(PRODUCES[name(o)]), name(o), o.id))
                return response(source)
            amount = next((o for o in options if o.kind == 'integer'), None)
            if amount: return response(amount, amount.constraints['minimum'])
            return self._cancel(obs, key, by_id)

        if kind == 'GAME_CHOOSE_ABILITY':
            choices = [o for o in options if o.kind == 'uuid']
            wanted = (self._payment.get(key) or ('R',))[0]
            def ability_rank(option):
                label = text(option.label)
                if '4 damage' in label: return 0
                if 'add {' + wanted.lower() + '}' in label: return 0
                if 'spectacle' in label or 'suspend' in label: return 1
                if 'search your library' in label: return 2
                if 'cast ' in label: return 3
                return 4
            if choices: return response(min(choices, key=lambda o: (ability_rank(o), o.label, o.id)))
            return self._cancel(obs, key, by_id)

        if kind == 'GAME_CHOOSE_CHOICE':
            choices = [o for o in options if o.kind == 'string']
            if choices:
                wanted = (self._payment.get(key) or ('R',))[0]
                color_names = {'R': 'red', 'W': 'white', 'B': 'black', 'U': 'blue', 'G': 'green'}
                def choice_rank(option):
                    label = text(option.label)
                    if '4 damage' in label: return 0  # Boros Charm's face mode.
                    if label == color_names[wanted] or 'add {' + wanted.lower() + '}' in label: return 1
                    if 'spectacle' in label or 'suspend' in label: return 2
                    return 3
                return response(min(choices, key=lambda o: (choice_rank(o), o.label, o.id)))
            return no()

        if kind == 'GAME_GET_AMOUNT':
            option = next(o for o in options if o.kind == 'integer')
            return response(option, option.constraints['minimum'])
        if kind == 'GAME_GET_MULTI_AMOUNT':
            option = next(o for o in options if o.kind == 'distribution')
            entries = option.constraints['entries']
            amounts = [e['min'] for e in entries]
            remaining = max(0, option.constraints['minimum'] - sum(amounts))
            for i, entry in enumerate(entries):
                extra = min(remaining, entry['max'] - amounts[i])
                amounts[i] += extra
                remaining -= extra
            if remaining or sum(amounts) > option.constraints['maximum']:
                raise UnsupportedInteraction('Infeasible damage distribution')
            return response(option, tuple(amounts))
        if kind == 'GAME_CHOOSE_PILE': return response(by_id.get('yes') or options[0])
        raise UnsupportedInteraction('Unsupported Burn decision: ' + kind)

    def _cancel(self, obs, key, options):
        pending = self._pending.get(key)
        if pending: self._failed[key, pending] = self._signature(obs)
        if 'no' not in options: raise UnsupportedInteraction('Unable to complete offered selection')
        return Response(options['no'].id)
