"""Factor land identity and nuisance life apart in a failed mana transfer test."""
from examples.train_scenarios import evaluate
import argparse
from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from patchwork_automaton.neural.policy import NeuralPolicy
from patchwork_automaton.neural.features import ABILITY_ID_SLICE,ABILITY_IDS
from patchwork_automaton.neural.scenarios import sample
from patchwork_automaton.neural.batching import example_batch,input_signature
from patchwork_automaton.backends.xmage_choices import choices,object_views
import tensorflow as tf


def mutable(value):
    if isinstance(value,Mapping):return {k:mutable(v) for k,v in value.items()}
    if isinstance(value,tuple):return [mutable(v) for v in value]
    return value


def counterfactual(scenario,*,life=None,land=None):
    obs=scenario.observation;view=mutable(obs.native)
    player=next(p for p in view['players'] if p['playerId']==obs.actor.value)
    source=next(iter(player['battlefield']))
    if life is not None:player['life']=life
    if land is not None:player['battlefield'][source]['name']=land
    options,_=choices(obs.decision.kind,obs.decision.native,view,source=source)
    by_id={o.id:o for o in options}
    # Rebind semantics through the real adapter and retain option order/UUIDs.
    decision=replace(obs.decision,options=tuple(by_id[o.id] for o in obs.decision.options))
    players=tuple(replace(p,life=player['life'],native=player) if p.id==obs.actor else p for p in obs.players)
    return replace(scenario,observation=replace(obs,players=players,objects=object_views(view),decision=decision,native=view))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--identity-fallback',action='store_true')
    p.add_argument('--cases',type=int,default=500);p.add_argument('--seed',type=int,default=123456)
    args=p.parse_args();policy=NeuralPolicy.load(args.checkpoint);model=policy.model
    forward=tf.function(model.call,input_signature=[input_signature(example_batch(model.memory_size))],autograph=False)
    training=json.loads((args.checkpoint.parent/'report.json').read_text());config=training['config']
    outcome_noise=config.get('outcome_noise',0.)
    if args.identity_fallback:
        seed=training['runs'][0]['seed'];calibration_rng=np.random.default_rng(50000+seed)
        tasks=config.get('tasks',['mana','bolt_target','charm_mode'])
        cards=np.zeros(model.card_count,bool);abilities=np.zeros(len(ABILITY_IDS),bool);cards[0]=True;abilities[0]=True
        for i in range(512):
            s=sample(tasks[i%len(tasks)],calibration_rng,distribution=config.get('distribution','legacy'))
            arrays=policy.encoder.encode(s.observation).arrays
            cards[arrays['card_ids']]=True
            abilities|=np.any(arrays['candidates'][:,ABILITY_ID_SLICE],axis=0)
        compiled=forward
        def forward(inputs):
            changed=dict(inputs,card_ids=inputs['card_ids'].copy(),candidates=inputs['candidates'].copy())
            changed['card_ids'][~cards[changed['card_ids']]]=0
            ids=changed['candidates'][...,ABILITY_ID_SLICE]
            removed=np.sum(ids[...,~abilities],axis=-1)
            ids*=abilities;ids[...,0]+=removed
            return compiled(changed)
    rng=np.random.default_rng(args.seed);states=[sample('mana',rng,heldout=True) for _ in range(args.cases)]
    result=dict(checkpoint=str(args.checkpoint),identity_fallback=args.identity_fallback,outcome_noise=outcome_noise,seed=args.seed,cases=args.cases,conditions={})
    for land in (None,'Sacred Foundry','Inspiring Vantage'):
        for life in (None,12):
            name=f'{land or "unseen Elegant Parlor"}; life={life or "heldout17-20"}'
            result['conditions'][name]=evaluate([counterfactual(s,life=life,land=land) for s in states],policy.encoder,model,forward,outcome_noise=outcome_noise)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for name,groups in result['conditions'].items():print(name,{k:round(v['success_probability'],5) for k,v in groups.items()})
