"""Inert goldfish opponent: keep, pass, and answer forced choices only."""
from ..core import CardSpec,DeckSpec,Response
from ..errors import UnsupportedInteraction


def passive_deck():
    return DeckSpec((CardSpec('Mountain',60),),'Goldfish: 60 Mountains')


class PassivePolicy:
    def choose(self,obs):
        if obs.decision is None:raise UnsupportedInteraction('Passive policy needs a decision')
        decision=obs.decision;options=sorted(decision.options,key=lambda o:o.id)
        by_id={o.id:o for o in options}
        if decision.kind=='GAME_TARGET':
            # Required cleanup discards still need a card. Starting-player
            # selection is controlled separately by the benchmark harness.
            target=next((o for o in options if o.kind=='uuid'),None)
            if target:return Response(target.id)
        if 'no' in by_id:return Response('no')
        # Never click an offered land/spell during priority, even if the server
        # somehow omits Pass. Fail visibly instead of making the opponent active.
        if decision.kind=='GAME_SELECT':raise UnsupportedInteraction('Passive priority has no Pass')
        if options:
            first=options[0]
            if first.kind=='integer':return Response(first.id,first.constraints['minimum'])
            if first.kind in ('uuid','string','boolean'):return Response(first.id)
        raise UnsupportedInteraction('Unhandled forced passive choice: '+decision.kind)
