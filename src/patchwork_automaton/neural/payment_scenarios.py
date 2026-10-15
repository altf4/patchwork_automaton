"""Controlled manual-payment and cancel/retry continuations.

The server can require a separate pool click after tapping a source. These
puzzles deliberately preserve that step. They are a small scripted subset of
Magic, with external demonstration labels and terminal task outcomes only.
"""
from .episodes import CastBoltEpisode
from ..backends.xmage_prompts import MANA_COLORS

PAYMENT_TASKS = ('spend_pool', 'tap_then_spend', 'cancel_retry', 'blaze_targets')
SYMBOLS = dict(white='W', blue='U', black='B', red='R', green='G', colorless='C')


class PaymentEpisode(CastBoltEpisode):
    def __init__(self, task, rng, *, heldout=False):
        super().__init__(rng, heldout=heldout, payment_only=True)
        self.task=task;self.hand={};self.board={};self.stack={};self.graveyard={}
        self.pool=dict.fromkeys(MANA_COLORS[:-1],0)
        self.debt=dict.fromkeys(MANA_COLORS,0);self.debt['red']=1
        self.spell_name='Lightning Bolt';self.damage=3;self.enemy_life=3
        self.stage='payment';self.target=self.other;self.creature_target=None
        self.recover=False;self.payable=True;self.tap_only=False;self.completion=False
        self.land_name=str(rng.choice(['Sacred Foundry','Sunbaked Canyon'] if heldout else
            ['Mountain','Inspiring Vantage','Sacred Foundry','Sunbaked Canyon']))
        self.board[self.land]=dict(self.card(self.land_name,self.land),tapped=True)
        self.stack[self.bolt]=self.card(self.spell_name,self.bolt)
        if task=='spend_pool':
            variant=int(rng.integers(4))
            if variant in (1,2):
                self.spell_name='Boros Charm';self.damage=4;self.enemy_life=4
                self.wanted='white' if variant==2 else 'red'
                self.debt['red']=0;self.debt[self.wanted]=1
                self.pool.update(red=1,white=1)
                self.land_name='Inspiring Vantage'
                self.board[self.land]=dict(self.card(self.land_name,self.land),tapped=True)
                self.add_source('Inspiring Vantage')
                # One pip was already paid; two additional units remain floating.
                self.add_source('Inspiring Vantage')
            elif variant==3:
                self.spell_name='Skullcrack';self.debt['generic']=1
                self.pool['red']=int(rng.choice([2,3]))
                for _ in range(self.pool['red']-1):self.add_source('Mountain')
            else:self.pool['red']=1
            self.branch=self.spell_name+':'+self.wanted
        elif task=='tap_then_spend':
            self.board[self.land]['tapped']=False;self.tap_only=True
            if self.land_name!='Mountain' and rng.integers(2):
                self.spell_name='Boros Charm';self.damage=4;self.enemy_life=4
                self.wanted=str(rng.choice(['red','white']))
                self.debt['red']=0;self.debt[self.wanted]=1
                self.add_source('Inspiring Vantage')
            self.branch=self.spell_name+':'+self.wanted+':'+self.land_name
        elif task=='cancel_retry':
            self.recover=True;variant=int(rng.integers(3))
            self.pool.update(red=int(variant==0),white=int(variant==2))
            self.payable=variant==0
            self.land_name='Sacred Foundry'
            self.board[self.land]=dict(self.card(self.land_name,self.land),tapped=True)
            self.stage='payment' if rng.integers(2) else 'priority'
            if self.stage=='priority':self.hand[self.bolt]=self.stack.pop(self.bolt)
            self.branch=('payable' if self.payable else 'unpayable')+':'+self.stage
        elif task=='blaze_targets':
            self.spell_name='Searing Blaze';self.pool['red']=2;self.debt['red']=2
            self.land_name='Mountain'
            self.board[self.land]=dict(self.card('Mountain',self.land),tapped=True)
            self.add_source('Mountain');self.nuisance={};self.enemy_graveyard={}
            self.swift=self.uid()
            self.board[self.swift]=dict(self.card('Monastery Swiftspear',self.swift),
                tapped=False,summoningSickness=False,canAttack=True,damage=0)
            self.blocker=None
            if rng.integers(2):
                self.blocker=self.uid()
                self.nuisance[self.blocker]=dict(self.card('Goblin Guide',self.blocker),
                    controllerId=self.other,tapped=False,damage=1)
            self.stage='priority';self.hand[self.bolt]=self.stack.pop(self.bolt)
            self.branch='kill_blocker' if self.blocker else 'own_creature_prowess'
            if rng.integers(2):
                # Ordinary, often nonlethal targeting prompts must also finish.
                # Start during targeting so unrelated cards stay contextual,
                # without inventing a priority legality mask for their casts.
                self.completion=True;self.blocker=None;self.nuisance={}
                self.enemy_life=int(rng.choice([3,7,12,20]));self.life=int(rng.choice([3,8,13,20]))
                for _ in range(int(rng.integers(1,4))):
                    key=self.uid();name=str(rng.choice(['Goblin Guide','Monastery Swiftspear']))
                    self.nuisance[key]=dict(self.card(name,key),controllerId=self.other,tapped=False,damage=0)
                for _ in range(int(rng.integers(2,7))):
                    key=self.uid();name=str(rng.choice(['Lava Spike','Lightning Bolt','Boros Charm','Searing Blaze','Mountain']))
                    self.hand[key]=self.card(name,key)
                self.stack[self.bolt]=self.hand.pop(self.bolt)
                self.stage=str(rng.choice(['target','creature_target']))
                self.branch='complete_cast:'+self.stage
        else:raise ValueError('Unknown payment task')
        card=self.card(self.spell_name,self.bolt)
        (self.hand if self.stage=='priority' else self.stack)[self.bolt]=card
        self.library=60-len(self.board)-len(self.hand)-len(self.stack)
        self.enemy_library=55-len(self.nuisance)
        self._offer()

    def add_source(self,name):
        key=self.uid();self.board[key]=dict(self.card(name,key),tapped=True)

    def _offer(self):
        if not hasattr(self,'task'):return super()._offer()
        self.source=None
        if self.stage=='priority':
            self.method='GAME_SELECT';self.data=dict(message='Play spells and abilities',options={'queryType':'SELECT'})
        elif self.stage in ('target','creature_target'):
            self.method='GAME_TARGET'
            targets=([self.me,self.other] if self.task=='blaze_targets' else [self.me,self.other,*self.nuisance])
            if self.stage=='creature_target':targets=[self.swift,*self.nuisance]
            self.data=dict(message='Select target creature' if self.stage=='creature_target' else
                'Select target player' if self.task=='blaze_targets' else 'Select any target',
                targets=targets,flag=False,options={'queryType':'PICK_TARGET','targetZone':'ALL',
                    'secondMessage':f"<font object_id='{self.bolt}'>{self.spell_name}</font>"})
        elif self.stage=='payment':
            cost=('{' + str(self.debt['generic']) + '}') if self.debt['generic'] else ''
            cost+=''.join(('{'+symbol+'}')*self.debt[color] for color,symbol in SYMBOLS.items())
            self.method='GAME_PLAY_MANA';self.data=dict(message=f"Pay {cost}<div><font object_id='{self.bolt}'>{self.spell_name}</font></div>",
                options={'queryType':'PLAY_MANA'})
        elif self.stage=='ability':
            super()._offer()
            self.data['_unpaid_mana']=dict(self.debt)

    def view(self):
        view=super().view()
        if hasattr(self,'task') and self.stage=='priority':
            view['canPlayObjects']={'objects':{self.bolt:{'basicCastAbilities':[
                {'id':self.cast_ability,'value':'Cast '+self.spell_name}]}}}
        return view

    def oracle_options(self):
        if self.stage=='priority':return {'no'} if self.recover and not self.payable else {'object:'+self.bolt}
        if self.stage=='target':return {'object:'+self.other}
        if self.stage=='creature_target':
            if self.completion:return {'object:'+k for k in self.nuisance}
            return {'object:'+(self.blocker or self.swift)}
        if self.stage=='ability':return {'object:'+(self.red if self.debt['red'] else self.white)}
        pool={'mana:'+c.upper() for c in SYMBOLS if self.pool[c]>0 and
              (self.debt[c]>0 or self.debt['generic']>0)}
        if pool:return pool
        if self.tap_only and not self.board[self.land]['tapped']:return {'object:'+self.land}
        return {'no'}

    def cancel(self):
        if self.completion:return self.finish(False)
        if self.recover or self.task=='blaze_targets':
            self.hand.update(self.stack);self.stack={};self.stage='priority';self.target=None
            self.creature_target=None;self._offer();return None
        return self.finish(False)

    def resolve(self):
        if self.task!='blaze_targets':return super().resolve()
        # No land entered this turn: Blaze deals one to each target. Successful
        # casting triggers prowess; then both players pass to scripted combat.
        swift=self.board[self.swift];swift.update(power='2',toughness='3')
        if self.target==self.other:self.enemy_life-=1
        elif self.target==self.me:self.life-=1
        if self.creature_target==self.swift:swift['damage']=1
        elif self.creature_target in self.nuisance:
            creature=self.nuisance[self.creature_target];creature['damage']=creature.get('damage',0)+1
            if creature['damage']>=int(creature['toughness']):
                self.enemy_graveyard[self.creature_target]=self.nuisance.pop(self.creature_target)
        self.graveyard.update(self.stack);self.stack={}
        if self.completion:
            return self.finish(self.target==self.other and self.creature_target is not None and self.life>0)
        if not self.nuisance:self.enemy_life-=2
        return self.finish(self.enemy_life<=0 and self.life>0)

    def step(self,response):
        if self.terminal:raise ValueError('Episode is terminal')
        if response.option not in {o.id for o in self.observation().decision.options}:raise ValueError('Action not offered')
        self.steps+=1;key=response.option.removeprefix('object:')
        if response.option=='no':
            if self.stage=='priority':return self.finish(self.recover and not self.payable)
            return self.cancel()
        if self.stage=='priority':
            self.stack[self.bolt]=self.hand.pop(self.bolt);self.stage='target'
        elif self.stage=='target':
            self.target=key;self.stage='creature_target' if self.task=='blaze_targets' else 'payment'
        elif self.stage=='creature_target':self.creature_target=key;self.stage='payment'
        elif self.stage=='ability':
            self.board[self.land]['tapped']=True
            if self.land_name=='Sunbaked Canyon':self.life-=1
            self.pool['red' if key==self.red else 'white']+=1;self.stage='payment'
        elif response.option.startswith('mana:'):
            color=response.option.split(':',1)[1].lower()
            paid=min(self.debt[color],self.pool[color])
            self.debt[color]-=paid;self.pool[color]-=paid
            generic=min(self.debt['generic'],self.pool[color])
            self.debt['generic']-=generic;self.pool[color]-=generic
            if not any(self.debt.values()):return self.resolve()
        elif key==self.land and not self.board[self.land]['tapped']:
            if self.land_name=='Mountain':
                self.board[self.land]['tapped']=True;self.pool['red']+=1
            else:self.stage='ability'
        else:raise ValueError('Unknown payment action')
        self._offer();return None
