"""Captured XMage feature layout for synthetic states; no strategy labels."""
from dataclasses import replace
from importlib.resources import files
import json
from ..burn_semantics import CARD_REGISTRY
from ..semantics import AbilityKind,CostKind,Color
from ..backends.xmage_choices import object_views

LAYOUT=json.loads(files('patchwork_automaton').joinpath('data/burn-opening-layout.json').read_text())


def card_layout(card):
    result=dict(card)
    for key in ('canAttack','canBlock','faceDown','isAbility','isToken'):result.setdefault(key,False)
    if 'color' not in result and result.get('name') in CARD_REGISTRY:
        spells=[a for a in CARD_REGISTRY[result['name']].abilities if a.kind==AbilityKind.SPELL]
        colors={c.color for c in spells[0].costs if c.kind==CostKind.MANA} if spells else set()
        result['color']={color.name.lower():color in colors for color in (Color.WHITE,Color.BLUE,Color.BLACK,Color.RED,Color.GREEN)}
        if 'manaValue' not in result:
            result['manaValue']=int(sum(c.quantity.value for c in spells[0].costs if c.kind==CostKind.MANA)) if spells else 0
    if not ('CREATURE' in result.get('cardTypes',())):
        result.setdefault('power','0');result.setdefault('toughness','0')
    return result


def with_native_layout(obs,uid):
    view=dict(obs.native)
    view['myHand']={key:card_layout(card) for key,card in view.get('myHand',{}).items()}
    view['stack']={key:card_layout(card) for key,card in view.get('stack',{}).items()}
    players=[]
    for raw in view['players']:
        player=dict(raw)
        for zone in ('battlefield','graveyard','exile'):
            player[zone]={key:card_layout(card) for key,card in player.get(zone,{}).items()}
        if player['playerId']==obs.actor.value:
            board={}
            for template in LAYOUT['sideboard']:
                key=uid();board[key]=dict(template,id=key,controllerId=obs.actor.value)
            # Real maps do not promise a presentation order. Random UUIDs give
            # a varying order without feeding their values into the model.
            player['sideboard']=dict(sorted(board.items()))
        players.append(player)
    view['players']=players
    helpers={}
    if obs.phase is not None:
        for template in LAYOUT['helpers']:
            key=uid();helpers[key]=dict(template,id=key,controllerId=obs.actor.value)
    view['myHelperEmblems']=helpers
    by_id={p['playerId']:p for p in players}
    return replace(obs,native=view,objects=object_views(view),
        players=tuple(replace(p,native=by_id[p.id.value]) for p in obs.players),
        turn=LAYOUT['turn'] if obs.phase is None else obs.turn)
