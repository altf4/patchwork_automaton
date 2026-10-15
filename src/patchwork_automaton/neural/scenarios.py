"""Synthetic one-decision Burn environments, using the real adapter and encoder.

These test local tactical objectives, not complete XMage games. Rewards are
terminal success (+1) or failure (-1); the oracle never enters model inputs.
"""
from dataclasses import dataclass, replace
from uuid import UUID

from ..core import Handle, DecisionToken, Decision, Observation, PlayerView
from ..backends.xmage_choices import choices, object_views

TACTICAL_TASKS = ('mana', 'bolt_target', 'charm_mode')
TASKS = (*TACTICAL_TASKS, 'mulligan', 'mulligan_bottom')

@dataclass
class Scenario:
    observation: Observation
    winning_options: frozenset
    variant: str

    def reward(self, option):
        if option not in {o.id for o in self.observation.decision.options}:
            raise ValueError('Action not offered')
        return 1. if option in self.winning_options else -1.


def sample(task, rng, *, heldout=False, distribution='legacy'):
    if distribution not in ('legacy','diverse','native'):
        raise ValueError('Unknown scenario distribution')
    if task in ('mulligan', 'mulligan_bottom'):
        return opening_sample(task, rng, heldout=heldout, distribution=distribution)
    def uid(): return str(UUID(bytes=rng.bytes(16)))
    me, opponent, game, source, spell, creature = [uid() for _ in range(6)]
    actor = Handle('xmage','player',me)
    game_handle = Handle('xmage','game',game)
    life = int(rng.integers(17,21) if heldout else rng.choice([*range(1,17),*range(21,41)])
               if distribution!='legacy' else rng.integers(6,17))
    pool = {c:0 for c in ('red','white','blue','black','green','colorless')}
    board = {}; stack = {}; data = {}; phase = 'precombat_main'
    other_life = 20
    if task == 'mana':
        # Boros Charm is being paid for, with exactly one of R/W already floating.
        need = 'white' if rng.integers(2) else 'red'
        pool['red' if need == 'white' else 'white'] = 1
        land = 'Elegant Parlor' if heldout else str(rng.choice(['Sacred Foundry','Inspiring Vantage']))
        board[source] = dict(id=source,name=land,cardTypes=['LAND'],tapped=False)
        stack[spell] = dict(id=spell,name='Boros Charm',cardTypes=['INSTANT'],controllerId=me,manaValue=2)
        red, white = uid(),uid()
        data={'choices':{red:'1. {T}: Add {R}.',white:'2. {T}: Add {W}.'}}
        method='GAME_CHOOSE_ABILITY'; winner='object:'+(white if need=='white' else red)
        variant='need_'+need
    elif task == 'bolt_target':
        other_life = 2 if heldout else int(rng.choice([1,3]))
        stack[spell] = dict(id=spell,name='Lightning Bolt',cardTypes=['INSTANT'],controllerId=me,manaValue=1)
        data={'targets':[me,opponent],'flag':False};method='GAME_TARGET'
        winner='object:'+opponent;variant='opponent_lethal'
    elif task == 'charm_mode':
        # A scripted continuation pays for Charm, resolves its prowess trigger,
        # and advances through combat with no blockers or further responses.
        # In the damage branch Swiftspear is NOT attacking; double strike cannot
        # win. In the other branch, even four damage plus normal combat is short.
        double = bool(rng.integers(2))
        power = (5 if heldout else 4) if double else (2 if heldout else 1)
        other_life = (12 if heldout else 10) if double else (3 if heldout else 4)
        if distribution!='legacy' and not heldout:
            # Overlapping powers prevent a fixed power threshold from standing
            # in for attacking/combat state. Held-out powers 2/5 and life 3/12
            # remain absent from the respective training branches.
            power=int(rng.choice([4,6,7,8,9,10] if double else [1,3,4,6,7,8,9,10]))
            other_life=int(rng.choice([n for n in range(power+6,2*(power+1)+1) if n!=12])) if double else int(rng.choice([1,2,4]))
        board[creature]=dict(id=creature,name='Monastery Swiftspear',cardTypes=['CREATURE'],
            power=str(power),toughness=str(power+1),attacking=double,tapped=double,controllerId=me)
        stack[spell]=dict(id=spell,name='Boros Charm',cardTypes=['INSTANT'],controllerId=me,manaValue=2)
        damage,indestructible,strike = uid(),uid(),uid()
        data={'choices':{damage:'Boros Charm deals 4 damage to target player or planeswalker.',
            indestructible:'Permanents you control gain indestructible until end of turn.',
            strike:'Target creature gains double strike until end of turn.'}}
        method='GAME_CHOOSE_ABILITY';phase='declare_blockers'
        winner='object:'+(strike if double else damage)
        variant='double_strike' if double else 'direct_damage'
    else: raise ValueError(task)
    players=[dict(playerId=me,name='Player',life=life,handCount=0,libraryCount=int(rng.integers(30,50)),
                  manaPool=pool,battlefield=board),
             dict(playerId=opponent,name='Opponent',life=other_life,handCount=int(rng.integers(0,5)),
                  libraryCount=int(rng.integers(30,50)),manaPool={c:0 for c in pool},battlefield={})]
    rng.shuffle(players)
    view=dict(myPlayerId=me,players=players,stack=stack,myHand={})
    options,_=choices(method,data,view,source=source if task=='mana' else spell if task=='charm_mode' else None)
    rng.shuffle(options)
    token=DecisionToken('xmage','scenario',game_handle,actor,1)
    obs=Observation('xmage',game_handle,actor,1,
        tuple(PlayerView(Handle('xmage','player',p['playerId']),p['name'],p['life'],p['handCount'],p['libraryCount'],native=p) for p in players),
        object_views(view),Decision(token,method,'',tuple(options),data),native=view,
        turn=int(rng.integers(4,15)),phase=phase,active_player=actor,priority_player=actor)
    if distribution=='native':
        from .native_layout import with_native_layout
        obs=with_native_layout(obs,uid)
    return Scenario(obs,frozenset([winner]),variant)


def opening_sample(task, rng, *, heldout=False, distribution='legacy'):
    """User-specified Burn opening objectives, not an optimal Magic oracle.

    Live XMage bottoms cards after each redraw, before the next keep prompt.
    Diverse cases use the full pinned main deck without exceeding copy counts;
    held-out cases reserve hand sizes and compositions, not entire deck cards.
    Legacy cases retain the original identity-transfer diagnostic.
    """
    def uid(): return str(UUID(bytes=rng.bytes(16)))
    me, opponent, game = [uid() for _ in range(3)]
    lands = ('Sacred Foundry', 'Sunbaked Canyon') if heldout else ('Mountain', 'Inspiring Vantage')
    cheap = ('Boltwave', 'Monastery Swiftspear') if heldout else ('Lightning Bolt', 'Lava Spike', 'Goblin Guide')
    expensive = ('Boros Charm',) if heldout else ('Skullcrack',)
    if task == 'mulligan':
        cheap = (*cheap, 'Rift Bolt' if heldout else 'Skewer the Critics')
    if distribution!='legacy':
        from ..baselines.modern_burn import DECK_METADATA,LAND_ORDER
        from ..burn_semantics import CARD_REGISTRY
        from ..semantics import AbilityKind,CostKind
        deck=tuple(c['name'] for c in DECK_METADATA['mainboard'] for _ in range(c['quantity']))
        def mana_value(name):
            spells=[a for a in CARD_REGISTRY[name].abilities if a.kind==AbilityKind.SPELL]
            return int(sum(c.quantity.value for c in spells[0].costs if c.kind==CostKind.MANA)) if spells else 0
        lands=tuple(n for n in deck if n in LAND_ORDER)
        expensive=tuple(n for n in deck if n not in LAND_ORDER and mana_value(n)==2)
        cheap=tuple(n for n in deck if n not in LAND_ORDER and
                    (mana_value(n)!=2 if task=='mulligan' else mana_value(n)==1))
    def draw(pool,count):
        return [str(pool[i]) for i in rng.choice(len(pool),size=count,replace=distribution=='legacy')]
    size = 6 if heldout else int(rng.choice([5, 7]))
    if task == 'mulligan':
        variant = str(rng.choice(['keep_two_lands', 'keep_three_lands', 'keep_one_land_low_curve',
                                 'mulligan_one_land_high_curve', 'mulligan_no_lands', 'mulligan_flooded', 'keep_small_hand']))
        if variant == 'keep_small_hand':
            size = 3 if heldout else 4
            land_count = int(rng.integers(0,size+1))
        elif variant == 'mulligan_no_lands': land_count = 0
        elif variant == 'mulligan_flooded': land_count = int(rng.integers(4, size+1))
        elif variant == 'keep_two_lands': land_count = 2
        elif variant == 'keep_three_lands': land_count = 3
        else: land_count = 1
        names = draw(lands,land_count)+draw(cheap,size-land_count)
        if variant == 'mulligan_one_land_high_curve':
            two_count = int(rng.integers(3, size))
        elif variant == 'keep_one_land_low_curve':
            two_count = int(rng.integers(0, 3))
        else:
            two_count = int(rng.integers(0, size-land_count+1))
        for i,name in enumerate(draw(expensive,two_count)): names[land_count+i] = name
        method = 'GAME_ASK'
        data = dict(message=f'Mulligan <font color=#ffff00>down to {size-1} cards</font>?',
                    min=0, max=0, flag=False, options={'UI.left.btn.text':'Mulligan',
                    'UI.right.btn.text':'Keep', 'queryType':'ASK'})
    else:
        size = 5 if heldout else int(rng.choice([6, 7]))
        variant = str(rng.choice(['bottom_excess_land', 'bottom_expensive_spell']))
        land_count = size - 2 if variant == 'bottom_excess_land' else 2
        names = draw(lands,land_count)+draw(cheap,size-land_count)
        if variant == 'bottom_expensive_spell': names[-1] = str(rng.choice(expensive))
        remaining = int(rng.integers(1, land_count - 1)) if variant == 'bottom_excess_land' else 1
        method = 'GAME_TARGET'
        data = dict(message=f'Select a card ({remaining} more) to put on the bottom of your library',
                    min=0, max=0, flag=True, options={'chosenTargets':[], 'targetZone':'HAND', 'queryType':'PICK_TARGET'})
    rng.shuffle(names)
    hand = {}
    for name in names:
        key = uid()
        creature = name in ('Goblin Guide', 'Monastery Swiftspear','Eidolon of the Great Revel')
        hand[key] = dict(id=key, name=name, cardTypes=['ENCHANTMENT','CREATURE'] if name=='Eidolon of the Great Revel' else ['LAND' if name in lands else 'CREATURE' if creature else
            'INSTANT' if name in ('Lightning Bolt', 'Skullcrack', 'Boros Charm','Searing Blaze') else 'SORCERY'],
            manaValue=0 if name in lands else 2 if name in expensive else 3 if name in ('Rift Bolt','Skewer the Critics') else 1,
            power='2' if name in ('Goblin Guide','Eidolon of the Great Revel') else '1' if creature else '0',
            toughness='2' if creature else '0', faceDown=False, isAbility=False, isToken=False)
    if task == 'mulligan_bottom': data['targets'] = list(hand)
    pool = {c:0 for c in ('red','white','blue','black','green','colorless')}
    players = [dict(playerId=me,name='Player',life=20,handCount=size,libraryCount=60-size,manaPool=pool,battlefield={}),
               dict(playerId=opponent,name='Opponent',life=20,handCount=7,libraryCount=53,manaPool=pool,battlefield={})]
    rng.shuffle(players)
    view = dict(myPlayerId=me,players=players,myHand=hand,stack={})
    options,_ = choices(method,data,view)
    rng.shuffle(options)
    actor = Handle('xmage','player',me)
    game_handle = Handle('xmage','game',game)
    obs = Observation('xmage',game_handle,actor,1,
        tuple(PlayerView(Handle('xmage','player',p['playerId']),p['name'],p['life'],p['handCount'],p['libraryCount'],native=p) for p in players),
        object_views(view),Decision(DecisionToken('xmage','scenario',game_handle,actor,1),method,data['message'],tuple(options),data),
        native=view,turn=0,phase=None)
    if task == 'mulligan':
        from ..baselines.burn_mulligan import keep_hand
        winners = {'no' if keep_hand(hand.values()) else 'yes'}
    else:
        winners = {'object:'+key for key,card in hand.items() if
                   (card['name'] in lands if variant=='bottom_excess_land' else card['name'] in expensive)}
    if distribution=='native':
        from .native_layout import with_native_layout
        obs=with_native_layout(obs,uid)
    return Scenario(obs,frozenset(winners),variant)


def advance_bottom(scenario, option):
    """Advance a synthetic bottom sequence; callers score only its final outcome.

    Returns None after the last card. The caller retains whether every choice
    met the synthetic objective; this never supplies intermediate game rewards.
    """
    from ..backends.xmage_prompts import prompt_metadata
    obs = scenario.observation
    metadata = prompt_metadata(obs.decision.kind, obs.decision.native)
    if metadata['purpose'] != 'bottom_hand': raise ValueError('Expected a bottom-card scenario')
    if option not in {o.id for o in obs.decision.options}: raise ValueError('Action not offered')
    if metadata['bottom_remaining'] == 1: return None
    hand = {key:card for key,card in obs.native['myHand'].items() if 'object:'+key != option}
    remaining = metadata['bottom_remaining'] - 1
    data = dict(obs.decision.native, targets=list(hand),
                message=f'Select a card ({remaining} more) to put on the bottom of your library')
    players = [dict(p,handCount=len(hand),libraryCount=p['libraryCount']+1) if p['playerId']==obs.actor.value
               else dict(p) for p in obs.native['players']]
    view = dict(obs.native,myHand=hand,players=players)
    options,_ = choices('GAME_TARGET',data,view)
    decision = replace(obs.decision,token=replace(obs.decision.token,generation=obs.decision.token.generation+1),
                       message=data['message'],native=data,options=tuple(options))
    observation = replace(obs,revision=obs.revision+1,native=view,decision=decision,objects=object_views(view),
        players=tuple(replace(p,hand_count=len(hand),library_count=p.library_count+1) if p.id==obs.actor else p for p in obs.players))
    winners = {'object:'+key for key,card in hand.items() if
               ('LAND' in card['cardTypes'] if scenario.variant=='bottom_excess_land' else card['manaValue']==2)}
    return replace(scenario,observation=observation,winning_options=frozenset(winners))


def sample_rewards(scenarios, actions, rng, outcome_noise=0.):
    """Terminal outcomes with optional independent luck; no shaping signals.

    With noise q, the correct choice wins with probability 1-q and an incorrect
    choice wins with probability q. The optimal action is unchanged for q<.5.
    """
    import numpy as np
    if not 0 <= outcome_noise < .5:
        raise ValueError('Outcome noise must be in [0, .5)')
    if len(scenarios)!=len(actions):
        raise ValueError('Actions must match scenarios')
    rewards=np.asarray([s.reward(s.observation.decision.options[a].id) for s,a in zip(scenarios,actions)],np.float32)
    if outcome_noise:
        rewards=np.where(rng.random(len(rewards))<outcome_noise,-rewards,rewards)
    return rewards
