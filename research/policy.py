"""Small solitaire policy: lands, untargeted spells, payment, priority."""
from patchwork_automaton.state import PROPERTY_NAMES
import struct

REVERSE = {name:int(key) for key,name in PROPERTY_NAMES.items()}


def property_id(name):
    return REVERSE.get(name,int(name.removeprefix('attribute_')) if name.startswith('attribute_') else -1)


def actions(card):
    props=card['properties']
    for key,value in props.items():
        code=property_id(key)
        if 202329600<=code<202329855 and isinstance(value,dict):
            if value.get('ACTION_IS_NON_GAME_ACTION'):
                continue
            yield code,value,value.get('ACTION_NAME_OVERRIDE_STRING') or value.get('ACTION_NAME_STRING') or props.get('ACTION_NAME_STRING','')


def card_payload(card_id,action_id,timestamp,targets=()):
    return struct.pack('<iiiIBBBB',card_id,action_id,0,timestamp,len(targets),0,1,0)+struct.pack('<i',len(targets))+b''.join(struct.pack('<I',x) for x in targets)


class SoloPolicy:
    def __init__(self,info):
        self.info=info

    def card_info(self,card):
        return self.info.get(str(card['properties'].get('CARDTEXTURE_NUMBER',0)//2),{})

    def choose(self,state):
        prompt=state.prompt
        if prompt is None or prompt['player'] not in (0,255):
            return None
        buttons=[b for b in prompt['buttons'] if b['enabled']]
        for button in buttons:
            if button['label'].lower()=='keep':
                return ('prompt',button['code'],0,'Keep')
        text=prompt['text'].lower()
        available=[]
        for card in state.cards.values():
            if card['properties'].get('CONTROLLER')!=0:
                continue
            for code,action,label in actions(card):
                available.append((card,code,action,label))
        if 'pay' in text or 'mana' in text:
            for card,code,action,label in available:
                info=self.card_info(card)
                if info.get('name')=='Mountain' and card['properties'].get('ZONE')==1000961 and not card['properties'].get('TAPPED') and ('add' in label.lower() or action.get('ACTION_FLAGS',0)&0x20):
                    return ('card',card['id'],code,info.get('name','Mana source'))
            for mana in prompt['mana']:
                if mana['player']==0 and mana['flags']&128:
                    return ('prompt',128,0,'Pay red mana')
            if prompt.get('flags',0)&128:
                return ('prompt',128,0,'Pay red mana')
            for button in buttons:
                if button['label'].lower() in ('cancel','cancel casting'):
                    return ('prompt',button['code'],0,'Cancel unaffordable spell')
        if prompt['phase'] in (4,10):
            for card,code,action,label in available:
                if label.lower()=='play' and card['properties'].get('ZONE')==1000448:
                    return ('card',card['id'],code,'Play '+self.card_info(card).get('name','land'))
            lands=sum(1 for c in state.cards.values() if c['properties'].get('ZONE')==1000961 and self.card_info(c).get('name')=='Mountain' and not c['properties'].get('TAPPED'))
            spells=[]
            for card,code,action,label in available:
                if label.lower()!='cast' or card['properties'].get('ZONE')!=1000448:
                    continue
                info=self.card_info(card)
                if not info or info['cmc']>lands:
                    continue
                # Direct burn in SOLO only has our player/own cards as targets.
                if any(property_id(k) in range(202482432,202482687) for k in action) or action.get('MIN_TARGETS',0)>0:
                    continue
                if info.get('name') in ('Fireblast','Skewer the Critics'):
                    continue
                spells.append((0 if 'CREATURE' in info.get('type_line','') else 1,info['cmc'],card,code,info['name']))
            if spells:
                _,_,card,code,name=min(spells,key=lambda x:(x[0],x[1],x[2]['id']))
                return ('card',card['id'],code,'Cast '+name)
        for button in buttons:
            if button['label'].lower() in ('ok','combat','end turn','done','yes','next') or button['code']==prompt.get('default'):
                return ('prompt',button['code'],0,button['label'])
        return None
