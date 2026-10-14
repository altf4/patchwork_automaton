"""Local integration smoke controller; this is not part of the library's policy.

Install [xmage], start scripts/xmage-local.py, then run:
  PYTHONPATH=src python examples/xmage_selfplay.py --episodes 2
Uses a deliberately simple freeform deck to exercise native server choices.
"""
import argparse
import asyncio
import json
from pathlib import Path

from patchwork_automaton.core import CardSpec,DeckSpec,Response
from patchwork_automaton.rl import XMageSelfPlay,EpisodeEnd


def smoke_response(obs):
    decision=obs.decision
    options=decision.options
    if not options:raise RuntimeError('Unhandled callback: '+decision.kind)
    if decision.kind=='GAME_ASK':
        # Smoke policy only: the server's actual buttons distinguish mulligans.
        option=next(o for o in options if o.id==('no' if any(o.label=='Keep' for o in options) else 'yes'))
    elif decision.kind=='GAME_SELECT':
        hand={h.value for h,o in obs.objects.items() if o.zone=='hand'}
        option=next((o for o in options if o.label=='Mountain' and o.id.removeprefix('object:') in hand),None)
        option=option or next((o for o in options if o.id.removeprefix('object:') in hand),None)
        option=option or next(o for o in options if o.id=='no')
    elif decision.kind=='GAME_TARGET':
        players={p.id.value for p in obs.players if p.id.value!=obs.actor.value}
        option=next((o for o in options if o.id.removeprefix('object:') in players),options[0])
    else:option=options[0]
    return Response(option.id,option.constraints.get('minimum',0) if option.kind=='integer' else None)


async def run(episodes=2,limit=100):
    deck=DeckSpec((CardSpec('Mountain',24),CardSpec('Lightning Bolt',36)),'Transport test burn')
    factory=XMageSelfPlay(deck,deck_type='Constructed - Freeform Unlimited',max_decisions=limit+10)
    summaries=[]
    try:
        for episode in range(episodes):
            runner=await factory.reset()
            kinds=set();land_plays=0;conceded=False
            while True:
                turn=await runner.next()
                if isinstance(turn,EpisodeEnd):
                    summary={'episode':episode,'decisions':runner.steps,'callbacks':sorted(kinds),'land_selections':land_plays,
                        'truncated':turn.truncated,'outcomes':{a:o.outcome.won if o.outcome else None for a,o in turn.observations.items()}}
                    print(json.dumps(summary),flush=True);summaries.append(summary);break
                kinds.add(turn.observation.decision.kind)
                if runner.steps>=limit and not conceded:
                    await runner.games[turn.actor].concede()
                    runner.current=None
                    await asyncio.gather(*(g.wait_for_result(timeout=30) for g in runner.games.values()))
                    conceded=True
                    continue
                response=smoke_response(turn.observation)
                selected=next(o for o in turn.observation.decision.options if o.id==response.option)
                if selected.label=='Mountain' and turn.observation.decision.kind=='GAME_SELECT':land_plays+=1
                await runner.submit(response)
    finally:await factory.close()
    return summaries


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=int,default=2)
    parser.add_argument('--limit',type=int,default=100,help='Concede after this many decisions if the game has not ended')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=asyncio.run(run(args.episodes,args.limit))
    if args.output:args.output.write_text(json.dumps(result,indent=2))
