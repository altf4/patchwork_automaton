"""Inspect numerical saturation of saved policies on development scenarios."""
from examples.train_scenarios import batch_for
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from patchwork_automaton.neural.ppo import normalized_advantages
from patchwork_automaton.neural.scenarios import TASKS, sample
from patchwork_automaton.neural.policy import NeuralPolicy, untrained_policy


def diagnose(policy):
    model=policy.model
    rng=np.random.default_rng(123456)
    scenarios=[sample(task,rng) for task in TASKS for _ in range(64)]
    inputs=batch_for(scenarios,policy.encoder,model.memory_size)
    captured={};originals={}
    for name in ('candidate_projection','query_projection','score_hidden','value_head'):
        layer=getattr(model,name);originals[name]=layer.call
        def capture(x,*args,_name=name,_call=layer.call,**kwargs):
            y=_call(x,*args,**kwargs)
            captured[_name]=(np.asarray(x),np.asarray(y))
            return y
        layer.call=capture
    try:output=model(inputs)
    finally:
        for name,original in originals.items():getattr(model,name).call=original
    result={}
    for name,(x,y) in captured.items():
        result[name]=dict(input_rms=float(np.sqrt(np.mean(x*x))),output_rms=float(np.sqrt(np.mean(y*y))),
                          max_abs=float(np.abs(y).max()),fraction_abs_above_099=float(np.mean(np.abs(y)>.99)))
    with tf.GradientTape(persistent=True) as tape:
        out=model(inputs)
        logp=tf.nn.log_softmax(out['logits'])
        probabilities=np.asarray(tf.exp(logp),np.float64)
        actions=np.asarray([rng.choice(len(p),p=p/p.sum()) for p in probabilities],np.int32)
        rewards=np.asarray([s.reward(s.observation.decision.options[a].id) for s,a in zip(scenarios,actions)],np.float32)
        advantage=normalized_advantages(rewards,np.asarray(out['value'])[:,0],np.ones(len(scenarios),bool),.1)
        actor=-tf.reduce_mean(tf.gather(logp,actions,axis=1,batch_dims=1)*advantage)
        critic=.5*tf.reduce_mean(tf.square(out['value'][:,0]-rewards))
    actor_grads=tape.gradient(actor,model.trainable_variables)
    critic_grads=tape.gradient(critic,model.trainable_variables)
    shared=[(tf.convert_to_tensor(a),tf.convert_to_tensor(c)) for a,c in zip(actor_grads,critic_grads) if a is not None and c is not None]
    an=float(tf.linalg.global_norm([a for a,c in shared]));cn=float(tf.linalg.global_norm([c for a,c in shared]))
    result['shared_gradient_probe']=dict(actor_norm=an,critic_norm=cn,cosine=float(sum(tf.reduce_sum(a*c) for a,c in shared))/(an*cn+1e-12),sample_reward=float(rewards.mean()))
    result['value_by_task']={task:np.asarray(output['value'])[i*64:(i+1)*64,0].tolist() for i,task in enumerate(TASKS)}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoints',nargs='*',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    results={'initial':diagnose(untrained_policy(seed=41,candidate_interactions=True,player_readout=True))}
    for path in args.checkpoints:results[str(path)]=diagnose(NeuralPolicy.load(path))
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    for name,r in results.items():print(name,json.dumps({k:v for k,v in r.items() if k!='value_by_task'}))
