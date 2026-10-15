"""Small scripted Burn continuations, not a general Magic rules engine.

Every step uses the actual XMage decoder and observation encoder. Only the
terminal synthetic task outcome supplies a reward. Full-game training continues
to use verified server win/loss/draw outcomes instead.
"""
from uuid import UUID
from ..core import Handle,DecisionToken,Decision,Observation,PlayerView,Outcome
from ..backends.xmage_choices import choices,object_views
from ..backends.xmage_prompts import MANA_COLORS
from .scenarios import sample,advance_bottom


class ScenarioEpisode:
    def __init__(self,task,rng,*,heldout=False):
        self.scenario=sample(task,rng,heldout=heldout,distribution='diverse')
        self.success=True;self.terminal=False;self.reward=None;self.steps=0

    def observation(self): return self.scenario.observation

    def step(self,response):
        if self.terminal:raise ValueError('Episode is terminal')
        self.success &= self.scenario.reward(response.option)>0;self.steps+=1
        following=(advance_bottom(self.scenario,response.option)
                   if self.scenario.variant.startswith('bottom_') else None)
        if following is None:
            self.terminal=True;self.reward=1. if self.success else -1.
        else:self.scenario=following
        return self.reward


class CastBoltEpisode:
    """Play a land, cast Bolt, target, pay, resolve; or pass with no payment.

    Opponent responses are scripted passes. The goal is to resolve Bolt at the
    opponent this turn, not to maximize arbitrary life changes in real games.
    """
    def __init__(self,rng,*,heldout=False,unpayable=False,payment_only=False,mana_color='red',charm=False,land_pool=None):
        if mana_color not in ('red','white') or mana_color=='white' and not payment_only:
            raise ValueError('White payment is a scripted Boros Charm continuation')
        self.rng=rng;self.heldout=heldout;self.unpayable=unpayable
        self.wanted=mana_color;charm=charm or mana_color=='white'
        self.spell_name='Boros Charm' if charm else 'Lightning Bolt'
        self.damage=4 if charm else 3
        self.me,self.other,self.game=[self.uid() for _ in range(3)]
        self.actor=Handle('xmage','player',self.me);self.game_handle=Handle('xmage','game',self.game)
        self.hand={};self.board={};self.graveyard={};self.stack={};self.steps=0
        self.pool=dict.fromkeys(MANA_COLORS[:-1],0);self.life=20;self.enemy_life=9 if heldout else 12
        self.terminal=False;self.reward=None;self.stage='priority';self.target=None
        pool=list(land_pool) if land_pool is not None else ['Sacred Foundry','Sunbaked Canyon'] if heldout else ['Mountain','Inspiring Vantage','Sacred Foundry']
        if not pool or any(n not in ('Mountain','Inspiring Vantage','Sacred Foundry','Sunbaked Canyon') for n in pool):
            raise ValueError('Unsupported scripted mana source')
        if charm:pool=[n for n in pool if n!='Mountain']
        if not pool:raise ValueError('Charm requires a red/white source')
        self.land_name=str(rng.choice(pool))
        self.land=self.uid();self.bolt=self.uid()
        self.land_ability,self.cast_ability=self.uid(),self.uid()
        self.hand[self.bolt]=self.card(self.spell_name,self.bolt)
        self.hand[self.land]=self.card(self.land_name,self.land)
        self.library=58
        if unpayable or payment_only:
            self.board[self.land]=dict(self.hand.pop(self.land),tapped=bool(unpayable))
            if self.land_name=='Sacred Foundry':self.life-=2
        if payment_only:
            self.stack[self.bolt]=self.hand.pop(self.bolt);self.stage='payment';self.target=self.other
            if charm:
                # One pip has already been paid from another dual. The board
                # alone cannot reveal which color remains; the engine debt can.
                paid=self.uid();self.board[paid]=dict(self.card('Inspiring Vantage',paid),tapped=True)
                self.library-=1
        # Realistic irrelevant battlefield creatures also create extra Bolt
        # targets. No generated creature has a granted mana ability.
        self.nuisance={}
        self.enemy_graveyard={}
        for _ in range(int(rng.integers(0,4))):
            key=self.uid();self.nuisance[key]=dict(self.card('Goblin Guide',key),controllerId=self.other,
                tapped=bool(rng.integers(2)),power='2',toughness='2')
        self.enemy_library=55-len(self.nuisance)
        self._offer()

    def uid(self):return str(UUID(bytes=self.rng.bytes(16)))

    def card(self,name,key):
        from ..baselines.modern_burn import LAND_ORDER
        from ..burn_semantics import CARD_REGISTRY
        from ..semantics import AbilityKind,CostKind
        land=name in LAND_ORDER
        creature=name in ('Goblin Guide','Monastery Swiftspear','Eidolon of the Great Revel')
        spells=[a for a in CARD_REGISTRY[name].abilities if a.kind==AbilityKind.SPELL]
        mana_value=int(sum(c.quantity.value for c in spells[0].costs if c.kind==CostKind.MANA)) if spells else 0
        return dict(id=key,name=name,controllerId=self.me,cardTypes=['ENCHANTMENT','CREATURE'] if name=='Eidolon of the Great Revel' else ['LAND' if land else 'CREATURE' if creature else
            'INSTANT' if name in ('Lightning Bolt','Boros Charm','Searing Blaze','Skullcrack') else 'SORCERY'],
            manaValue=mana_value,power='1' if name=='Monastery Swiftspear' else '2' if creature else '0',
            toughness='2' if creature else '0',faceDown=False,isAbility=False,isToken=False)

    def _offer(self):
        self.data={};self.source=None
        if self.stage=='priority':
            self.method='GAME_SELECT';self.data=dict(message='Play spells and abilities',options={'queryType':'SELECT'})
        elif self.stage=='shock':
            self.method='GAME_ASK';self.data=dict(message='pay 2 life? (otherwise Sacred Foundry becomes tapped)',
                options={'UI.left.btn.text':'Yes','UI.right.btn.text':'No',
                         'secondMessage':f"<font object_id='{self.land}'>Sacred Foundry</font>"})
        elif self.stage=='target':
            self.method='GAME_TARGET';self.data=dict(message='Select any target',targets=[self.me,self.other,*self.nuisance],flag=False,
                options={'queryType':'PICK_TARGET','targetZone':'ALL',
                         'secondMessage':f"<font object_id='{self.bolt}'>Lightning Bolt</font>"})
        elif self.stage=='payment':
            symbol='R' if self.wanted=='red' else 'W'
            self.method='GAME_PLAY_MANA';self.data=dict(message=f"Pay {{{symbol}}}<div><font object_id='{self.bolt}'>{self.spell_name}</font></div>",
                options={'queryType':'PLAY_MANA'})
        elif self.stage=='ability':
            self.method='GAME_CHOOSE_ABILITY';self.source=self.land
            self.red,self.white=self.uid(),self.uid()
            prefix='{T}, Pay 1 life:' if self.land_name=='Sunbaked Canyon' else '{T}:'
            self.data=dict(message='Choose ability',choices={self.red:prefix+' Add {R}.',self.white:prefix+' Add {W}.'},
                           _unpaid_mana=dict.fromkeys(MANA_COLORS,0)|{self.wanted:1})

    def view(self):
        players=[dict(playerId=self.me,name='Player',life=self.life,handCount=len(self.hand),libraryCount=self.library,
                      manaPool=dict(self.pool),battlefield=self.board,graveyard=self.graveyard),
                 dict(playerId=self.other,name='Opponent',life=self.enemy_life,handCount=5,libraryCount=self.enemy_library,
                      manaPool=dict.fromkeys(MANA_COLORS[:-1],0),battlefield=self.nuisance,graveyard=self.enemy_graveyard)]
        if self.heldout:players.reverse()
        view=dict(myPlayerId=self.me,myHand=self.hand,stack=self.stack,players=players)
        if self.stage=='priority':
            playable={}
            if self.land in self.hand:playable[self.land]={'basicPlayAbilities':[{'id':self.land_ability,'value':'Play '+self.land_name}]}
            if self.bolt in self.hand:playable[self.bolt]={'basicCastAbilities':[{'id':self.cast_ability,'value':'Cast Lightning Bolt'}]}
            view['canPlayObjects']={'objects':playable}
        return view

    def observation(self):
        view=self.view();options,_=choices(self.method,self.data,view,source=self.source)
        players=tuple(PlayerView(Handle('xmage','player',p['playerId']),p['name'],p['life'],p['handCount'],p['libraryCount'],native=p)
                      for p in view['players'])
        token=DecisionToken('xmage','synthetic-episode',self.game_handle,self.actor,self.steps+1)
        decision=None if self.terminal else Decision(token,self.method,self.data.get('message',''),tuple(options),self.data)
        return Observation('xmage',self.game_handle,self.actor,self.steps+1,players,object_views(view),decision,
            terminal=self.terminal,outcome=Outcome(won=self.reward>0,draw=False,native={'synthetic':True}) if self.terminal else None,
            native=view,turn=5 if self.heldout else 3,phase='precombat_main',active_player=self.actor,priority_player=self.actor)

    def finish(self,success):
        self.terminal=True;self.reward=1. if success else -1.
        return self.reward

    def resolve(self):
        self.graveyard.update(self.stack);self.stack={}
        if self.target==self.other:self.enemy_life-=self.damage
        elif self.target==self.me:self.life-=self.damage
        elif self.target in self.nuisance:self.enemy_graveyard[self.target]=self.nuisance.pop(self.target)
        return self.finish(self.target==self.other and self.life>0)

    def step(self,response):
        if self.terminal:raise ValueError('Episode is terminal')
        obs=self.observation()
        if response.option not in {o.id for o in obs.decision.options}:raise ValueError('Action not offered')
        self.steps+=1;selected=response.option.removeprefix('object:')
        if self.stage=='priority':
            if response.option=='no':return self.finish(self.unpayable)
            if self.unpayable:return self.finish(False)
            if selected==self.land:
                self.board[self.land]=dict(self.hand.pop(self.land),tapped=False)
                self.stage='shock' if self.land_name=='Sacred Foundry' else 'priority'
            else:
                if self.land not in self.board or self.board[self.land].get('tapped'):return self.finish(False)
                self.stack[self.bolt]=self.hand.pop(self.bolt);self.stage='target'
        elif self.stage=='shock':
            if response.option!='yes':
                self.board[self.land]['tapped']=True
                return self.finish(False)
            self.life-=2;self.stage='priority'
        elif self.stage=='target':
            if response.option=='no':return self.finish(False)
            self.target=selected;self.stage='payment'
        elif self.stage=='payment':
            if response.option=='no':return self.finish(False)
            if response.option.startswith('mana:'):
                color=response.option.split(':',1)[1].lower()
                if color==self.wanted:
                    self.pool[color]-=1
                    return self.resolve()
                self._offer();return None
            if selected!=self.land:raise ValueError('Unknown mana source')
            if self.land_name=='Mountain':
                self.board[self.land]['tapped']=True
                return self.resolve()
            self.stage='ability'
        elif self.stage=='ability':
            if selected not in (self.red,self.white):return self.finish(False)
            self.board[self.land]['tapped']=True
            if self.land_name=='Sunbaked Canyon':self.life-=1
            color='red' if selected==self.red else 'white'
            if color==self.wanted:return self.resolve()
            self.pool[color]+=1;self.stage='payment'
        self._offer()
        return None
