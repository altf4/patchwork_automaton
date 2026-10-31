"""Held-out Standard opponent with conservative combat and interaction choices.

This deterministic policy is reserved for evaluation. It uses the same legal
RustMage options as StandardAggro, but can block, trade, hold removal for a
meaningful threat, and decline attacks into an obvious unfavorable block.
"""

from ..core import Response
from .standard import REMOVAL, StandardAggro, _creature


def _stats(obj):
    if obj is None:
        return (0, 0)
    return (int(obj.properties.get('power') or 0),
            int(obj.properties.get('toughness') or 0))


def _life(obs, actor):
    return next((p.life for p in obs.players if p.id == actor), None)


class StandardTactical(StandardAggro):
    """A legal, seat-private tactical test opponent, never used for PPO data."""

    @staticmethod
    def _cast_allowed(obs, name):
        if not StandardAggro._cast_allowed(obs, name):
            return False
        if name not in REMOVAL:
            return True
        threats = [o for o in obs.objects.values()
                   if o.zone == 'battlefield' and o.controller != obs.actor
                   and _creature(o)]
        own_life = _life(obs, obs.actor)
        return (len(threats) >= 2 or any(_stats(o)[0] >= 3 for o in threats)
                or own_life is not None and own_life <= 10)

    def choose(self, obs):
        options = obs.decision.options if obs.decision is not None else ()
        objects = {int(handle.value): obj for handle, obj in obs.objects.items()}

        if obs.decision is not None and obs.decision.kind == 'blockers':
            candidates = []
            own_life = _life(obs, obs.actor)
            for option in options:
                if option.kind != 'finite':
                    continue
                action = option.constraints['action']
                if action['kind'] != 'Block':
                    continue
                blocker = objects.get(action['blocker'])
                attacker = objects.get(action['attacker'])
                if blocker is None or attacker is None:
                    continue
                our_power, our_toughness = _stats(blocker)
                their_power, their_toughness = _stats(attacker)
                kills = our_power >= their_toughness > 0
                survives = our_toughness > their_power
                urgent = own_life is not None and own_life <= max(7, their_power + 2)
                if kills and survives:
                    class_rank = 0
                elif kills and (urgent or our_power <= their_power):
                    class_rank = 1  # Equal or favorable trade.
                elif urgent and survives:
                    class_rank = 2
                elif urgent:
                    class_rank = 3  # Protect a low life total.
                else:
                    continue
                candidates.append(((class_rank, -their_power, our_power, option.id), option))
            if candidates:
                return Response(min(candidates, key=lambda row: row[0])[1].id)

        if obs.decision is not None and obs.decision.kind == 'attackers':
            enemy = [o for o in obs.objects.values()
                     if o.zone == 'battlefield' and o.controller != obs.actor
                     and _creature(o) and not o.properties.get('tapped')]
            opponent_life = next((p.life for p in obs.players if p.id != obs.actor), None)
            attacks = []
            finish = None
            for option in options:
                if option.kind != 'finite':
                    continue
                action = option.constraints['action']
                if action['kind'] == 'FinishAttackers':
                    finish = option
                if action['kind'] != 'Attack':
                    continue
                creature = objects.get(action['card'])
                power, toughness = _stats(creature)
                poor_block = any(_stats(blocker)[0] >= toughness > 0
                                 and _stats(blocker)[1] > power for blocker in enemy)
                if poor_block and (opponent_life is None or power < opponent_life):
                    continue
                attacks.append(((-power, -toughness, option.id), option))
            if attacks:
                return Response(min(attacks, key=lambda row: row[0])[1].id)
            if finish is not None:
                return Response(finish.id)

        return super().choose(obs)
