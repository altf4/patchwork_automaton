"""Pure XMage callback -> typed primitive choices. No prompt-text policy."""
from collections.abc import Mapping
import uuid

from ..core import Handle, ObjectView, Option
from ..properties import freeze
from ..errors import UnsupportedInteraction


def object_views(view):
    objects={}
    def add(cards,zone,controller=None):
        if isinstance(cards,Mapping):
            iterable=cards.items()
        elif isinstance(cards,(list,tuple)):
            iterable=((c.get('id'),c) for c in cards if isinstance(c,Mapping))
        else:
            return
        for key,card in iterable:
            if not isinstance(card,Mapping):
                continue
            object_id=card.get('id') or key
            if not object_id:
                continue
            handle=Handle('xmage','object',str(object_id))
            owner=card.get('controllerId') or controller
            objects[handle]=ObjectView(handle,None if card.get('hideInfo') or card.get('faceDown') else card.get('name'),zone,
                Handle('xmage','player',str(owner)) if owner else None,freeze(card))
    me=view.get('myPlayerId')
    add(view.get('myHand',{}),'hand',me)
    add(view.get('stack',{}),'stack')
    add(view.get('myHelperEmblems',{}),'command',me)
    for player in view.get('players',()):
        pid=player.get('playerId')
        for key,zone in (('battlefield','battlefield'),('graveyard','graveyard'),('exile','exile'),('commandList','command')):
            add(player.get(key,{}),zone,pid)
        if pid==me:
            add(player.get('sideboard',{}),'sideboard',pid)
    for key,zone in (('exiles','exile'),('revealed','revealed'),('lookedAt','looked_at'),('companion','sideboard')):
        for group in view.get(key,()):
            if isinstance(group,Mapping):
                add(group.get('cards',{}),zone)
    return objects


def choices(method,data,view,*,source=None):
    options=[];bindings={};offered=set()
    def add(key,kind,label,command,args=None,**constraints):
        if key in bindings:
            return
        options.append(Option(key,kind,str(label),freeze(constraints)))
        bindings[key]=(command,dict(args or {}),dict(constraints))
    def boolean(value,label):
        add('yes' if value else 'no','boolean',label,'sendPlayerBoolean',{'value':value})
    def select(value,label=None,legality='advertised',**constraints):
        value=str(uuid.UUID(str(value)))
        if method=='GAME_PLAY_MANA' and value in by_id:
            from .burn_actions import mana_click_available
            if not mana_click_available(by_id[value]): return
        add('object:'+value,'uuid',label or value,'sendPlayerUUID',{'value':value},legality=legality,**constraints)
        offered.add(value)
    native=data.get('options') or {}
    objects=object_views(view)
    by_id={key.value:value for key,value in objects.items()}
    def label(value):
        obj=by_id.get(str(value))
        if obj:return obj.name or str(value)
        return next((p.get('name',str(value)) for p in view.get('players',()) if p.get('playerId')==str(value)),str(value))
    if method=='GAME_ASK':
        boolean(True,native.get('UI.left.btn.text') or 'Yes')
        boolean(False,native.get('UI.right.btn.text') or 'No')
    elif method=='GAME_CHOOSE_PILE':
        boolean(True,'Pile 1');boolean(False,'Pile 2')
    elif method=='GAME_GET_AMOUNT':
        add('amount','integer',data.get('message','Amount'),'sendPlayerInteger',minimum=data['min'],maximum=data['max'])
    elif method=='GAME_GET_MULTI_AMOUNT':
        add('amounts','distribution',data.get('message') or 'Choose amounts','sendPlayerString',
            entries=data.get('messages') or [],minimum=data['min'],maximum=data['max'])
        if native.get('canCancel'):
            boolean(False,'Cancel')
    elif method=='GAME_CHOOSE_ABILITY':
        from .burn_actions import bind_ability
        for key,text in (data.get('choices') or {}).items():
            select(key,text,**bind_ability(key,text,view,objects,source))
        boolean(False,'Cancel')
    elif method=='GAME_CHOOSE_CHOICE':
        choice=data.get('choice') or {}
        values=choice.get('keyChoices') or {str(v):str(v) for v in choice.get('choices',())}
        for i,(value,text) in enumerate(values.items()):
            add(f'choice:{i}','string',text,'sendPlayerString',{'value':value})
        if not choice.get('required',True):
            boolean(False,'Cancel')
    elif method in ('GAME_SELECT','GAME_TARGET','GAME_PLAY_MANA','GAME_PLAY_XMANA'):
        if method=='GAME_PLAY_MANA':
            # Spending already-produced mana is a distinct server response,
            # not another permanent click and not a boolean confirmation.
            me=view.get('myPlayerId')
            player=next((p for p in view.get('players',()) if p.get('playerId')==me),{})
            pool=player.get('manaPool') or {}
            if not isinstance(pool,Mapping):pool={}
            try:uuid.UUID(str(me))
            except (ValueError,TypeError):pool={}
            for color in ('WHITE','BLUE','BLACK','RED','GREEN','COLORLESS'):
                amount=pool.get(color.lower())
                if type(amount) is int and amount>0:
                    add('mana:'+color,'string','Spend '+color.lower()+' mana from pool','sendPlayerManaType',
                        {'playerId':me,'manaType':color},action_kind='spend_mana',source=me,mana_color=color,
                        legality='pool available; server validates restrictions')
        if method=='GAME_TARGET' and native.get('queryType')=='PICK_ABILITY':
            # HumanPlayer.chooseTriggeredAbility expects the trigger instance ID,
            # not its source card ID. These choices live in cardsView1 rather
            # than data.targets, which is null for simultaneous trigger order.
            for value,ability in (data.get('cardsView1') or {}).items():
                if isinstance(ability,Mapping):
                    source=(ability.get('sourceCard') or {}).get('id')
                    select(value,ability.get('sourceName') or 'Triggered ability',
                           selection_kind='trigger_order',source=source)
        for value in data.get('targets') or ():
            select(value,label(value))
        for key in ('possibleTargets','possibleAttackers','possibleBlockers'):
            for value in native.get(key) or ():
                if isinstance(value,str):select(value,label(value))
        if method!='GAME_TARGET':
            from .burn_actions import mana_click_available
            playable=(view.get('canPlayObjects') or {}).get('objects') or {}
            for value,stats in playable.items():
                if method=='GAME_PLAY_MANA' and not stats.get('basicManaAbilities'):
                    continue
                select(value,label(value))
            if method=='GAME_PLAY_MANA' and not offered:
                # Native UI accepts object clicks then asks for an ability. Do
                # not label these inferred clicks as a fully legal action mask.
                for handle,obj in objects.items():
                    if (obj.zone=='battlefield' and obj.controller and obj.controller.value==view.get('myPlayerId')
                            and mana_click_available(obj)):
                        select(handle.value,obj.name,legality='clickable; server validates mana ability')
        if method in ('GAME_SELECT','GAME_PLAY_MANA','GAME_PLAY_XMANA') or not data.get('flag',True) or native.get('UI.right.btn.text'):
            boolean(False,native.get('UI.right.btn.text') or ('Pass' if method=='GAME_SELECT' else 'Done / Cancel'))
        if native.get('UI.left.btn.text'):
            boolean(True,native['UI.left.btn.text'])
        if native.get('specialButton'):
            add('special','string',native['specialButton'],'sendPlayerString',{'value':'special'})
        # X mana is collected by the server with an integer response.
        if method=='GAME_PLAY_XMANA':
            add('amount','integer','X mana','sendPlayerInteger',minimum=0,maximum=2147483647)
    # Unknown callback data remain attached to Decision.native. No guessed send.
    return options,bindings


def encode_response(binding,value):
    command,args,constraints=binding
    args=dict(args)
    if command=='sendPlayerInteger' and 'value' not in args:
        if type(value) is not int or not constraints['minimum']<=value<=constraints['maximum']:
            raise ValueError('Number is outside the offered integer bounds')
        args['value']=value
    elif command=='sendPlayerString' and 'entries' in constraints:
        entries=constraints['entries']
        if not isinstance(value,(list,tuple)) or len(value)!=len(entries):
            raise ValueError('Supply one amount per offered recipient')
        if any(type(n) is not int or not e['min']<=n<=e['max'] for n,e in zip(value,entries)):
            raise ValueError('Amount is outside recipient bounds')
        if not constraints['minimum']<=sum(value)<=constraints['maximum']:
            raise ValueError('Amounts violate the offered total bounds')
        args['value']=' '.join(str(n) for n in value)
    elif value is not None:
        raise ValueError('This finite option takes no additional value')
    return command,args
