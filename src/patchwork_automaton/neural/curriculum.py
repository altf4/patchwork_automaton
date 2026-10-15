"""Five short Burn curricula, with terminal outcomes and external teacher labels.

These are controlled puzzles, not a replacement Magic engine. All offered
actions pass through the production XMage decoder. Teacher choices are never
attached to an observation or given to the network.
"""
from dataclasses import replace
import numpy as np

from ..core import Decision
from ..backends.xmage_choices import choices, object_views
from ..backends.xmage_prompts import prompt_metadata
from ..baselines.burn_mulligan import keep_hand
from ..baselines.modern_burn import DECK_METADATA,LAND_ORDER
from .episodes import CastBoltEpisode
from .native_layout import with_native_layout
from .scenarios import sample, Scenario

TASKS = ('openings', 'lands_and_casting', 'mana_colors', 'lethal', 'creatures_and_attacking')
from .payment_scenarios import PAYMENT_TASKS,PaymentEpisode
ALL_TASKS = TASKS + PAYMENT_TASKS


class OpeningEpisode:
    """Keep/redraw, bottom the London cards, then answer the next keep prompt."""
    def __init__(self, rng, *, heldout=False):
        self.rng=rng
        self.scenario=sample('mulligan',rng,heldout=heldout,distribution='native')
        self.branch=self.scenario.variant
        self.steps=0;self.terminal=False;self.reward=None

    def observation(self): return self.scenario.observation

    def oracle_options(self): return self.scenario.winning_options

    def _replace(self, hand, method, data, winners, variant):
        obs=self.observation()
        players=[dict(p,handCount=len(hand),libraryCount=60-len(hand))
                 if p['playerId']==obs.actor.value else dict(p) for p in obs.native['players']]
        view=dict(obs.native,myHand=hand,players=players)
        options,_=choices(method,data,view)
        self.rng.shuffle(options)
        decision=Decision(replace(obs.decision.token,generation=self.steps+1),method,data['message'],tuple(options),data)
        observation=replace(obs,revision=self.steps+1,native=view,objects=object_views(view),decision=decision,
            players=tuple(replace(p,hand_count=len(hand),library_count=60-len(hand),native=next(
                raw for raw in players if raw['playerId']==p.id.value)) if p.id==obs.actor else p for p in obs.players))
        self.scenario=Scenario(observation,frozenset(winners),variant)

    @staticmethod
    def bottom_options(hand):
        lands=[k for k,c in hand.items() if 'LAND' in c['cardTypes']]
        # This is the explicit, limited bottoming teacher for these puzzles.
        # Keep two lands where possible, then bottom the highest printed MV.
        if len(lands)>2: return {'object:'+k for k in lands}
        spells={k:c for k,c in hand.items() if k not in lands}
        if not spells: return {'object:'+k for k in hand}
        maximum=max(c['manaValue'] for c in spells.values())
        return {'object:'+k for k,c in spells.items() if c['manaValue']==maximum}

    def step(self,response):
        if self.terminal: raise ValueError('Episode is terminal')
        obs=self.observation()
        if response.option not in {o.id for o in obs.decision.options}: raise ValueError('Action not offered')
        correct=response.option in self.oracle_options();self.steps+=1
        if not correct:
            self.terminal=True;self.reward=-1.;return self.reward
        meta=prompt_metadata(obs.decision.kind,obs.decision.native)
        if meta['purpose']=='mulligan':
            if response.option=='no':
                self.terminal=True;self.reward=1.;return self.reward
            next_size=len(obs.native['myHand'])-1
            # Real London redraws are seven cards; bottoming happens before
            # XMage asks whether to keep the resulting smaller hand.
            fresh=sample('mulligan',self.rng,distribution='diverse').observation
            templates={c['name']:dict(c) for c in fresh.native['myHand'].values()}
            from ..burn_semantics import CARD_REGISTRY
            from ..semantics import AbilityKind,CostKind
            deck=[c['name'] for c in DECK_METADATA['mainboard'] for _ in range(c['quantity'])]
            from uuid import UUID
            hand={}
            lands=set(LAND_ORDER)
            for i in self.rng.choice(len(deck),size=7,replace=False):
                name=deck[i];key=str(UUID(bytes=self.rng.bytes(16)))
                spells=[a for a in CARD_REGISTRY[name].abilities if a.kind==AbilityKind.SPELL]
                mv=int(sum(c.quantity.value for c in spells[0].costs if c.kind==CostKind.MANA)) if spells else 0
                creature=name in ('Goblin Guide','Monastery Swiftspear','Eidolon of the Great Revel')
                hand[key]=dict(templates.get(name,{}),id=key,name=name,controllerId=obs.actor.value,
                    cardTypes=['ENCHANTMENT','CREATURE'] if name=='Eidolon of the Great Revel' else ['LAND' if name in lands else 'CREATURE' if creature else
                               'INSTANT' if name in ('Lightning Bolt','Boros Charm','Skullcrack','Searing Blaze') else 'SORCERY'],
                    manaValue=mv,power='1' if name=='Monastery Swiftspear' else '2' if creature else '0',
                    toughness='2' if creature else '0',faceDown=False,isAbility=False,isToken=False)
            from .native_layout import card_layout
            hand={k:card_layout(c) for k,c in hand.items()}
            remaining=7-next_size
            data=dict(message=f'Select a card ({remaining} more) to put on the bottom of your library',
                targets=list(hand),flag=True,options={'targetZone':'HAND','queryType':'PICK_TARGET'})
            self._replace(hand,'GAME_TARGET',data,self.bottom_options(hand),'opening_bottom')
        else:
            hand={k:c for k,c in obs.native['myHand'].items() if 'object:'+k!=response.option}
            remaining=meta['bottom_remaining']-1
            if remaining:
                data=dict(obs.decision.native,targets=list(hand),
                    message=f'Select a card ({remaining} more) to put on the bottom of your library')
                self._replace(hand,'GAME_TARGET',data,self.bottom_options(hand),'opening_bottom')
            else:
                data=dict(message=f'Mulligan <font color=#ffff00>down to {len(hand)-1} cards</font>?',
                    flag=False,options={'UI.left.btn.text':'Mulligan','UI.right.btn.text':'Keep','queryType':'ASK'})
                self._replace(hand,'GAME_ASK',data,{'no' if keep_hand(hand.values()) else 'yes'},'opening_keep')
        return None


class CastingEpisode(CastBoltEpisode):
    def __init__(self,rng,*,heldout=False,colors=False):
        payment_only=colors or bool(rng.integers(4)==0)
        unpayable=not payment_only and bool(rng.integers(5)==0)
        super().__init__(rng,heldout=heldout,unpayable=unpayable,payment_only=payment_only,
            mana_color=str(rng.choice(['red','white'])) if colors else 'red',charm=colors,
            land_pool=None if heldout else ['Mountain','Inspiring Vantage','Sacred Foundry','Sunbaked Canyon'])
        # A second land in hand must not be advertised after using the turn's
        # land play. Production legality remains the server's responsibility.
        if not payment_only:
            extra=self.uid();self.hand[extra]=self.card('Mountain',extra);self.library-=1
        self.branch=('need_'+self.wanted if colors else 'unpayable' if unpayable else
                     'payment' if payment_only else 'land_then_cast')+':'+self.land_name

    def oracle_options(self):
        if self.stage=='priority': return {'no' if self.unpayable else 'object:'+(self.land if self.land in self.hand else self.bolt)}
        if self.stage=='shock': return {'yes'}
        if self.stage=='target': return {'object:'+self.other}
        if self.stage=='payment': return {'object:'+self.land}
        return {'object:'+(self.red if self.wanted=='red' else self.white)}


class LethalEpisode(CastBoltEpisode):
    """One or two one-mana burn spells must actually resolve for a win."""
    def __init__(self,rng,*,heldout=False):
        super().__init__(rng,heldout=heldout)
        self.hand={};self.board={};self.stack={};self.graveyard={}
        self.enemy_life=int(rng.choice([2,5] if heldout else [1,3,4,6]))
        self.life=int(rng.choice([1,2,3,8,17,20]))
        for _ in range(2):
            k=self.uid();self.board[k]=dict(self.card('Mountain',k),tapped=False)
        for _ in range(int(rng.integers(2,4))):
            k=self.uid();name=str(rng.choice(['Lightning Bolt','Lava Spike','Boltwave']))
            self.hand[k]=dict(self.card(name,k),cardTypes=['INSTANT' if name=='Lightning Bolt' else 'SORCERY'])
        self.library=60-len(self.board)-len(self.hand)
        self.branch='two_spells' if self.enemy_life>3 else 'one_spell'
        self.stage='priority';self._offer()

    def _offer(self):
        # Parent construction calls this before the subclass is initialized.
        if not hasattr(self,'branch'): return super()._offer()
        if self.stage=='priority':
            self.method='GAME_SELECT';self.source=None
            self.data=dict(message='Play spells and abilities',options={'queryType':'SELECT'})
        elif self.stage=='target':
            self.method='GAME_TARGET';self.source=None
            targets=[self.me,self.other,*self.nuisance] if self.spell_name=='Lightning Bolt' else [self.me,self.other]
            self.data=dict(message='Select any target' if self.spell_name=='Lightning Bolt' else 'Select target player',
                targets=targets,flag=True,options={'queryType':'PICK_TARGET','targetZone':'ALL',
                    'secondMessage':f"<font object_id='{self.bolt}'>{self.spell_name}</font>"})
        else:
            self.method='GAME_PLAY_MANA';self.source=None
            self.data=dict(message=f"Pay {{R}}<div><font object_id='{self.bolt}'>{self.spell_name}</font></div>",
                options={'queryType':'PLAY_MANA'})

    def view(self):
        view=super().view()
        if hasattr(self,'branch') and self.stage=='priority':
            view['canPlayObjects']={'objects':{k:{'basicCastAbilities':[{'id':self.cast_ability,'value':'Cast '+c['name']}]}
                for k,c in self.hand.items()} if any(not c['tapped'] for c in self.board.values()) else {}}
        return view

    def oracle_options(self):
        if self.stage=='priority': return {'object:'+k for k in self.hand}
        if self.stage=='target': return {'object:'+self.other}
        return {'object:'+k for k,c in self.board.items() if not c['tapped']}

    def step(self,response):
        if self.terminal: raise ValueError('Episode is terminal')
        if response.option not in {o.id for o in self.observation().decision.options}: raise ValueError('Action not offered')
        self.steps+=1;k=response.option.removeprefix('object:')
        if response.option=='no': return self.finish(False)
        if self.stage=='priority':
            self.bolt=k;self.spell_name=self.hand[k]['name'];self.stack[k]=self.hand.pop(k);self.stage='target'
            if self.spell_name=='Boltwave': self.target=self.other;self.stage='payment'
        elif self.stage=='target': self.target=k;self.stage='payment'
        else:
            self.board[k]['tapped']=True;self.graveyard.update(self.stack);self.stack={}
            if self.target==self.other: self.enemy_life-=3
            elif self.target==self.me: self.life-=3
            elif self.target in self.nuisance: self.enemy_graveyard[self.target]=self.nuisance.pop(self.target)
            if self.life<=0: return self.finish(False)
            if self.enemy_life<=0: return self.finish(True)
            if not self.hand or not any(not c['tapped'] for c in self.board.values()): return self.finish(False)
            self.stage='priority'
        self._offer();return None


class CreatureEpisode(CastingEpisode):
    """Deploy a haste creature, pass into combat and declare all safe attackers.

    There are no blockers in this introductory puzzle. It does not teach a
    general rule to attack into blockers. Summoning-sick creatures can also be
    present; only server-style eligible attackers are offered.
    """
    def __init__(self,rng,*,heldout=False):
        super().__init__(rng,heldout=heldout)
        # Always start before the land drop, with a playable haste creature.
        self.hand={self.land:self.card(self.land_name,self.land)};self.board={};self.stack={}
        self.spell_name=str(rng.choice(['Goblin Guide','Monastery Swiftspear']))
        self.hand[self.bolt]=self.card(self.spell_name,self.bolt)
        self.hand[self.bolt].update(power='2' if self.spell_name=='Goblin Guide' else '1',toughness='2')
        self.unpayable=False;self.stage='priority';self.nuisance={};self.enemy_graveyard={}
        self.library=58;self.enemy_library=55;self.enemy_life=20
        self.deployed=False;self.branch=self.spell_name+':'+self.land_name
        if rng.integers(2):
            k=self.uid();self.board[k]=dict(self.card('Eidolon of the Great Revel',k),power='2',toughness='2',
                tapped=False,summoningSickness=True,canAttack=False,attacking=False);self.library-=1
        self._offer()

    def _offer(self):
        if getattr(self,'stage',None)=='attack':
            self.method='GAME_SELECT';self.source=None
            eligible=[k for k,c in self.board.items() if c.get('canAttack') or c.get('attacking')]
            self.data=dict(message='Select attackers',flag=False,options={'possibleAttackers':eligible,'queryType':'SELECT',
                'specialButton':'All attack'})
        else: super()._offer()

    def view(self):
        view=super().view()
        if self.stage=='attack': view.pop('canPlayObjects',None)
        return view

    def observation(self):
        obs=super().observation()
        return replace(obs,phase='declare_attackers') if self.stage=='attack' else obs

    def oracle_options(self):
        if self.stage=='priority' and self.deployed: return {'no'}
        if self.stage=='attack':
            remaining={'object:'+k for k,c in self.board.items() if c.get('canAttack') and not c.get('attacking')}
            return remaining|{'special'} if remaining else {'no'}
        return super().oracle_options()

    def resolve(self):
        card=self.stack.pop(self.bolt)
        self.board[self.bolt]=dict(card,tapped=False,summoningSickness=False,canAttack=True,attacking=False)
        self.deployed=True;self.stage='priority';self._offer();return None

    def step(self,response):
        if self.terminal: raise ValueError('Episode is terminal')
        if response.option not in {o.id for o in self.observation().decision.options}: raise ValueError('Action not offered')
        if self.stage=='priority' and self.deployed:
            self.steps+=1;self.stage='attack';self._offer();return None
        if self.stage=='attack':
            self.steps+=1
            if response.option=='no':
                eligible=[c for c in self.board.values() if c.get('canAttack') or c.get('attacking')]
                success=bool(eligible) and all(c.get('attacking') for c in eligible)
                if success:self.enemy_life-=sum(int(c['power']) for c in eligible)
                return self.finish(success)
            keys=[k for k,c in self.board.items() if c.get('canAttack')] if response.option=='special' else [response.option.removeprefix('object:')]
            for k in keys:
                c=self.board[k];selected=not c.get('attacking',False)
                c.update(attacking=selected,tapped=selected,canAttack=not selected)
            self._offer();return None
        reward=super().step(response)
        if not self.terminal and self.stage=='target':
            # Creature spells do not choose a damage target.
            self.stage='payment';self._offer()
        return reward


class NativeEpisode:
    """Stable auxiliary IDs, native layout and shuffled menus across a sequence."""
    def __init__(self,episode,rng):
        self.episode=episode;self.keys=[episode.uid() for _ in range(18)]
        self.seed=int(rng.integers(2**31));self.cached=None

    def __getattr__(self,key): return getattr(self.episode,key)

    def observation(self):
        if self.cached is None or self.cached.revision!=self.episode.steps+1:
            obs=with_native_layout(self.episode.observation(),iter(self.keys).__next__)
            options=list(obs.decision.options);np.random.default_rng(self.seed+self.episode.steps).shuffle(options)
            self.cached=replace(obs,decision=replace(obs.decision,options=tuple(options)))
        return self.cached

    def step(self,response):
        self.cached=None
        return self.episode.step(response)


def make_episode(task,rng,*,heldout=False):
    if task in PAYMENT_TASKS:return NativeEpisode(PaymentEpisode(task,rng,heldout=heldout),rng)
    if task not in TASKS: raise ValueError('Unknown curriculum task')
    if task=='openings': return OpeningEpisode(rng,heldout=heldout)
    cls=LethalEpisode if task=='lethal' else CreatureEpisode if task=='creatures_and_attacking' else CastingEpisode
    kwargs={'colors':True} if task=='mana_colors' else {}
    return NativeEpisode(cls(rng,heldout=heldout,**kwargs),rng)
