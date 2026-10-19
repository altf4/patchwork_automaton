"""One deliberately simple aggro policy, translated into each engine's prompts.

Keep all hands, play a land, cast cheapest/name-first on own main with an empty
stack, never kick/activate abilities, burn the opponent, attack all, never block.
Buff friendly creatures before combat; aim creature removal at enemy creatures.
Only the pinned mono-color Foundations decks are supported by this controller.
It uses visible state only. This is a throughput workload, not a strong player.
"""
import json
from importlib.resources import files

from ..core import CardSpec, DeckSpec, Response
from ..errors import UnsupportedInteraction

DATA = json.loads(files('patchwork_automaton').joinpath('data/foundations-aggro.json').read_text())
FACTS = DATA['cards']
PUMPS = frozenset(('Kindled Fury', 'Sure Strike'))
REMOVAL = frozenset(('Scorching Dragonfire', 'Fiery Annihilation'))


def foundations_deck(color):
    row = DATA['decks'][color]
    return DeckSpec(tuple(CardSpec(**card) for card in row['cards']), row['name'])


def card_order(name):
    return FACTS[name]['mana_value'], name


class FoundationsAggro:
    def __init__(self):
        self._attacked = set()
        self._casting = {}  # XMage can ask for targets before putting a spell on the stack.

    def choose(self, obs):
        if obs.decision is None: raise UnsupportedInteraction('No decision for this seat')
        if obs.backend == 'rustmage': return self._rustmage(obs)
        if obs.backend == 'xmage': return self._xmage(obs)
        raise UnsupportedInteraction('Unsupported Foundations backend: ' + obs.backend)

    @staticmethod
    def _main(obs):
        return (obs.active_player == obs.actor and obs.phase in ('precombat_main', 'postcombat_main')
                and not any(o.zone == 'stack' for o in obs.objects.values()))

    @staticmethod
    def _available(obs):
        me = next(p for p in obs.players if p.id == obs.actor)
        pool = me.native.get('mana', ()) if obs.backend == 'rustmage' else (me.native.get('manaPool') or {}).values()
        return sum(v for v in pool if type(v) is int and v > 0) + sum(
            o.zone == 'battlefield' and o.controller == obs.actor and o.name in ('Mountain', 'Plains')
            and not o.properties.get('tapped', True) for o in obs.objects.values())

    @staticmethod
    def _creatures(obs, *, friendly):
        return [o for o in obs.objects.values() if o.zone == 'battlefield'
                and (o.controller == obs.actor) == friendly
                and 'Creature' in FACTS.get(o.name, {}).get('type_line', '')]

    def _castable(self, obs, name, mana):
        if name not in FACTS or FACTS[name]['mana_value'] > mana: return False
        if name in PUMPS:
            return obs.phase == 'precombat_main' and any(
                not o.properties.get('tapped', True) for o in self._creatures(obs, friendly=True))
        if name in REMOVAL: return bool(self._creatures(obs, friendly=False))
        return True

    def _rustmage(self, obs):
        # A blocked attacker can require an explicit damage assignment. Keep
        # this baseline simple: assign all available damage to the offered
        # target, including when the complete range exceeds the finite menu.
        damage = next((o for o in obs.decision.options if o.id == 'domain:damage_assignment'), None)
        if damage is not None:
            return Response(damage.id, damage.constraints['maximum'])
        candidates = []
        objects = {int(h.value): o for h, o in obs.objects.items()}
        main = self._main(obs)
        mana = self._available(obs)
        pending = obs.native.get('announcing') or obs.native.get('resolving') or {}
        source_name = pending.get('source_card')
        for option in obs.decision.options:
            if option.kind != 'finite': continue
            a = option.constraints['action']; k = a['kind']; rank = None
            obj = objects.get(a.get('card')); name = obj.name if obj else ''
            if k in ('Keep', 'FinishOpeningHand', 'PutTrigger', 'PayCosts'): rank = 0
            elif k == 'ChooseKicker' and not a['pay']: rank = 0
            elif k == 'ChooseOptional' and not a['accept']: rank = 0
            elif k == 'PlayLand' and main: rank = 1
            elif k == 'AnnounceCast' and main and self._castable(obs, name, mana): rank = 2
            elif k == 'TapMana' and obs.decision.kind == 'pay_costs': rank = 3
            elif k == 'ChooseTarget':
                if a['target'].get('Player') not in (None, obs.actor.value): rank = 0
                target = objects.get(a['target'].get('Card'))
                if target and source_name in PUMPS | REMOVAL:
                    if (target.controller == obs.actor) == (source_name in PUMPS): rank = 0
            elif k == 'Attack': rank = 0
            elif k == 'AssignDamage': rank = -a['amount']
            elif k in ('FinishTargets', 'FinishAttackers', 'FinishBlockers'): rank = 5
            elif k in ('Bottom', 'Discard'): rank = 0
            elif k == 'Pass': rank = 10
            if rank is not None:
                order = card_order(name) if name in FACTS else (0, name)
                if k in ('Bottom', 'Discard'): order = (0 if name in ('Mountain', 'Plains') else 1, name)
                candidates.append(((rank, order, a.get('card', 0), option.id), option))
        if not candidates:
            raise UnsupportedInteraction('Uncovered Foundations RustMage decision: ' + obs.decision.kind
                                         + ' ' + repr([o.label for o in obs.decision.options]))
        return Response(min(candidates, key=lambda x: x[0])[1].id)

    def _xmage(self, obs):
        d = obs.decision; by_id = {o.id: o for o in d.options}
        objects = {h.value: o for h, o in obs.objects.items()}
        offered = [o for o in d.options if o.kind == 'uuid']
        def obj(option): return objects.get(option.id.removeprefix('object:'))
        def name(option): return obj(option).name if obj(option) else ''
        def first(choices):
            return Response(min(choices, key=lambda o: (name(o) or o.label, o.id)).id)
        def no():
            if 'no' not in by_id: raise UnsupportedInteraction('Missing completion option: ' + d.kind)
            return Response('no')
        if d.kind == 'GAME_ASK':
            keep = next((o for o in d.options if o.label.lower() == 'keep'), None)
            return Response(keep.id) if keep else no()  # Decline kicker/optional payments.
        if d.kind == 'GAME_SELECT':
            self._casting.pop((obs.game, obs.actor), None)
            native = d.native.get('options') or {}
            if 'possibleAttackers' in native:
                key = obs.game, obs.actor, obs.turn
                if 'special' in by_id and key not in self._attacked:
                    self._attacked.add(key)
                    return Response('special')
                return no()
            if 'possibleBlockers' in native: return no()
            if not self._main(obs): return no()
            lands = [o for o in offered if name(o) in ('Mountain', 'Plains') and obj(o).zone == 'hand']
            if lands: return first(lands)
            mana = self._available(obs)
            spells = [o for o in offered if obj(o) and obj(o).zone == 'hand' and name(o) in FACTS
                      and name(o) not in ('Mountain', 'Plains') and self._castable(obs, name(o), mana)]
            if spells:
                chosen = min(spells, key=lambda o: (card_order(name(o)), o.id))
                self._casting[obs.game, obs.actor] = name(chosen)
                return Response(chosen.id)
            return no()
        if d.kind == 'GAME_TARGET':
            # Starting-player selection is controlled by the benchmark harness.
            from ..backends.xmage_prompts import prompt_metadata, prompt_source
            purpose = prompt_metadata(d.kind, d.native)['purpose']
            if purpose == 'starting_player':
                own = [o for o in offered if o.id == 'object:' + obs.actor.value]
                if own: return first(own)
            enemy = {p.id.value for p in obs.players if p.id != obs.actor}
            targets = [o for o in offered if o.id.removeprefix('object:') in enemy]
            if targets: return first(targets)
            source = objects.get(prompt_source(d.native))
            source_name = source.name if source else self._casting.get((obs.game, obs.actor))
            if source_name in PUMPS | REMOVAL:
                targets = [o for o in offered if obj(o) and obj(o).zone == 'battlefield'
                           and (obj(o).controller == obs.actor) == (source_name in PUMPS)]
                if targets: return first(targets)
            # Mandatory trigger ordering and cleanup discards.
            if offered:
                return Response(min(offered, key=lambda o: (0 if name(o) in ('Mountain', 'Plains') else 1,
                                                            name(o) or o.label, o.id)).id)
            return no()
        if d.kind == 'GAME_PLAY_MANA':
            pool = [o for o in d.options if o.constraints.get('action_kind') == 'spend_mana']
            if pool: return first(pool)
            lands = [o for o in offered if name(o) in ('Mountain', 'Plains') and not obj(o).properties.get('tapped', True)]
            if lands: return first(lands)
            raise UnsupportedInteraction('Affordable Foundations cast cannot be paid')
        if d.kind == 'GAME_CHOOSE_ABILITY':
            # Our creatures' activated abilities are never initiated.
            cast = [o for o in offered if 'cast ' in o.label.lower() and 'kicker' not in o.label.lower()]
            if cast: return first(cast)
            if len(offered) == 1: return first(offered)
        if d.kind == 'GAME_CHOOSE_CHOICE':
            choices = [o for o in d.options if o.kind == 'string']
            plain = [o for o in choices if o.label.lower() in ('no', 'red', 'white') or 'without' in o.label.lower()]
            if plain: return first(plain)
        raise UnsupportedInteraction('Uncovered Foundations XMage decision: ' + d.kind + ' ' + repr(d.native))
