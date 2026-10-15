"""CPU reward-only learning diagnostics on held-out synthetic Burn scenarios."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('KERAS_BACKEND','tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import argparse
import hashlib
import importlib
from functools import partial
import json
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
from patchwork_automaton.neural.scenarios import TASKS,sample,sample_rewards
from patchwork_automaton.neural.policy import untrained_policy,NeuralPolicy
from patchwork_automaton.neural.features import Encoder,collate
from patchwork_automaton.neural.experiment import StreamingTrainer,atomic_json
from patchwork_automaton.neural.batching import example_batch,input_signature
from patchwork_automaton.neural.ppo import PPOTrainer
from patchwork_automaton.neural.augmentation import mask_identities


def batch_for(scenarios,encoder,memory,*,runtime_history=False):
    items=[encoder.encode(s.observation).arrays for s in scenarios]
    if runtime_history:
        from patchwork_automaton.neural.history import History
        for s,item in zip(scenarios,items):
            history=History(encoder.registry);history.observe(s.observation);item.update(history.take(s.observation))
    arrays=collate(items)
    arrays.update(memory_in=np.zeros((len(scenarios),memory),np.float32),advance_memory=np.ones((len(scenarios),1),np.float32))
    return arrays


def evaluate(scenarios,encoder,model,forward,*,outcome_noise=0.,runtime_history=False):
    groups={}
    for start in range(0,len(scenarios),48):
        chunk=scenarios[start:start+48];inputs=batch_for(chunk,encoder,model.memory_size,runtime_history=runtime_history)
        output=forward(inputs);probs=np.asarray(tf.nn.softmax(output['logits']));values=np.asarray(output['value'])[:,0]
        for s,p,value in zip(chunk,probs,values):
            options=s.observation.decision.options
            winners=[i for i,o in enumerate(options) if o.id in s.winning_options]
            row=groups.setdefault(s.variant,dict(count=0,greedy=0,success_probability=0.,random_probability=0.,value_mean=0.,value_calibration_mse=0.))
            row['count']+=1;row['greedy']+=int(np.argmax(p) in winners)
            success=float(p[winners].sum())
            row['success_probability']+=success;row['random_probability']+=len(winners)/len(options)
            expected_value=2*(outcome_noise+(1-2*outcome_noise)*success)-1
            row['value_mean']+=float(value);row['value_calibration_mse']+=(float(value)-expected_value)**2
    for row in groups.values():
        for key in ('greedy','success_probability','random_probability','value_mean','value_calibration_mse'):row[key]/=row['count']
        if outcome_noise:row['expected_win_probability']=outcome_noise+(1-2*outcome_noise)*row['success_probability']
    return groups


def passes(groups):
    return bool(groups) and all(v['success_probability']>=.9 and v['greedy']>=.95 for v in groups.values())


def run(args):
    draw=partial(sample,distribution=args.distribution)
    args.directory.mkdir(parents=True,exist_ok=True)
    if (args.directory/'report.json').exists():raise FileExistsError('Use a new diagnostic directory')
    sources={name:hashlib.sha256(Path(importlib.import_module('patchwork_automaton.neural.'+name).__file__).read_bytes()).hexdigest()
             for name in ('features','model','ppo','scenarios','augmentation','policy','native_layout')}
    sources['runner']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in ('baselines.burn_mulligan','backends.xmage_prompts','backends.xmage_choices','backends.burn_actions'):
        sources[name]=hashlib.sha256(Path(importlib.import_module('patchwork_automaton.'+name).__file__).read_bytes()).hexdigest()
    sources['native_layout_data']=hashlib.sha256(Path('src/patchwork_automaton/data/burn-opening-layout.json').read_bytes()).hexdigest()
    report=dict(source_sha256=sources,synthetic=True,objective='supervised capacity control' if args.supervised_control else 'one-decision terminal task success; not full-game wins',config=vars(args)|{'directory':str(args.directory),'load':str(args.load) if args.load else None},runs=[])
    for seed in args.seeds:
        start=time.monotonic();rng=np.random.default_rng(seed)
        policy=NeuralPolicy.load(args.load,seed=seed) if args.load else untrained_policy(seed=seed,candidate_interactions=args.candidate_interactions,
                                player_readout=args.player_readout,stable_heads=args.stable_heads,standardize_inputs=args.standardize_inputs,isolate_value=args.isolate_value,
                                fact_readout=args.fact_readout);model=policy.model;encoder=Encoder()
        if args.load:
            for flag in ('candidate_interactions','player_readout','stable_heads','standardize_inputs','isolate_value','fact_readout'):
                if getattr(model,flag)!=getattr(args,flag):raise ValueError('Loaded model flag mismatch: '+flag)
        if args.standardize_inputs and not args.load:
            calibration_rng=np.random.default_rng(50000+seed)
            # Only training-distribution observations; no rewards, labels, or
            # held-out states contribute to the frozen feature statistics.
            calibration=[encoder.encode(draw(args.tasks[i%len(args.tasks)],calibration_rng).observation).arrays for i in range(512)]
            for key,layer in (('entities',model.entity_standardization),('candidates',model.candidate_standardization)):
                mask_key='entity_mask' if key=='entities' else 'candidate_mask'
                layer.adapt(np.concatenate([a[key][a[mask_key]] for a in calibration]))
            if args.fact_readout:
                model.context_standardization.adapt(np.stack([a['context'] for a in calibration]))
        trainer=StreamingTrainer(model,learning_rate=args.learning_rate,batch_size=args.batch)
        trainer.compile()
        ppo=PPOTrainer(model,learning_rate=args.learning_rate,epochs=args.ppo_epochs,batch_size=args.batch,
                       clip_ratio=args.ppo_clip,target_kl=args.target_kl,advantage_scale_floor=args.advantage_scale_floor,trust_guard=args.trust_guard,max_state_kl=args.max_state_kl) if args.algorithm=='ppo' else None
        optimizer_rng=np.random.default_rng(60000+seed)
        outcome_rng=np.random.default_rng(40000+seed)
        augmentation_rng=np.random.default_rng(30000+seed)
        forward=tf.function(model.call,input_signature=[input_signature(example_batch(model.memory_size))],autograph=False)
        @tf.function(input_signature=[input_signature(example_batch(model.memory_size)),tf.TensorSpec([None,None],tf.bool)])
        def supervised_gradients(inputs,targets):
            with tf.GradientTape() as tape:
                logp=tf.nn.log_softmax(model(inputs)['logits'])
                # Equivalent copies are all correct; do not teach an arbitrary
                # UUID/order tie-break when several bottom choices succeed.
                loss=-tf.reduce_mean(tf.reduce_logsumexp(tf.where(targets,logp,tf.fill(tf.shape(logp),-1.e9)),axis=1))
            grads=tape.gradient(loss,model.trainable_variables)
            return [tf.zeros_like(v) if g is None else tf.convert_to_tensor(g) for g,v in zip(grads,model.trainable_variables)],tf.stack([loss])
        test_rng=np.random.default_rng((91000 if args.development_only else 90000)+seed)
        test=[draw(task,test_rng,heldout=not args.development_only) for task in args.tasks for _ in range(args.test_cases)]
        monitor_rng=np.random.default_rng(70000+seed)
        monitor=[draw(task,monitor_rng) for task in args.tasks for _ in range(40)]
        score=partial(evaluate,encoder=encoder,model=model,forward=forward,outcome_noise=args.outcome_noise,runtime_history=args.runtime_history)
        initial=score(test)
        history=[];optimization_steps=0;early_stops=0
        guard_stats=dict(rejected_epochs=0,zero_step_rollouts=0,max_accepted_kl=0.,max_accepted_state_kl=0.)
        for update in range(args.updates):
            progress=update/max(args.updates-1,1)
            final_rate=args.learning_rate if args.final_learning_rate is None else args.final_learning_rate
            scheduled_rate=args.learning_rate+(final_rate-args.learning_rate)*progress
            if ppo is not None:
                ppo.learning_rate=scheduled_rate
                ppo.optimizer.learning_rate.assign(scheduled_rate)
            else:
                trainer.optimizer.learning_rate.assign(scheduled_rate)
            scenarios=[draw(args.tasks[(update*args.batch+i)%len(args.tasks)],rng) for i in range(args.batch)]
            inputs=mask_identities(batch_for(scenarios,encoder,model.memory_size,runtime_history=args.runtime_history),augmentation_rng,args.identity_dropout)
            output=forward(inputs);probs=np.asarray(tf.nn.softmax(output['logits']),dtype=np.float64)
            actions=np.asarray([rng.choice(len(p),p=p/p.sum()) for p in probs],np.int32)
            rewards=sample_rewards(scenarios,actions,outcome_rng,args.outcome_noise)
            ppo_metrics=None
            if ppo is not None:
                logp=np.asarray(tf.nn.log_softmax(output['logits']))[np.arange(len(actions)),actions]
                ppo_metrics=ppo.update(inputs,actions,rewards,logp,np.asarray(output['value'])[:,0],rng=optimizer_rng,old_log_probs=np.asarray(tf.nn.log_softmax(output['logits'])))
                metrics=[ppo_metrics['loss']];norm=ppo_metrics['gradient_norm']
                optimization_steps+=ppo_metrics['steps'];early_stops+=int(ppo_metrics['early_stop'])
                guard_stats['rejected_epochs']+=ppo_metrics['rejected_epochs']
                guard_stats['zero_step_rollouts']+=int(ppo_metrics['steps']==0)
                guard_stats['max_accepted_kl']=max(guard_stats['max_accepted_kl'],ppo_metrics['largest_accepted_kl'])
                guard_stats['max_accepted_state_kl']=max(guard_stats['max_accepted_state_kl'],ppo_metrics['largest_accepted_state_kl'])
            elif args.supervised_control:
                targets=np.zeros_like(inputs['candidate_mask'])
                for row,s in enumerate(scenarios):
                    for column,o in enumerate(s.observation.decision.options):targets[row,column]=o.id in s.winning_options
                gradients,metrics=supervised_gradients(inputs,targets)
            else:
                gradients,metrics=trainer.gradient_fn(inputs,actions,rewards,np.asarray(output['value'])[:,0])
            if ppo is None:
                for g in gradients:tf.debugging.assert_all_finite(g,'Nonfinite scenario gradient')
                clipped,norm=tf.clip_by_global_norm(gradients,trainer.clip_norm)
                trainer.optimizer.apply_gradients(zip(clipped,model.trainable_variables))
                optimization_steps+=1
            if (update+1)%100==0 or update==0:
                row=dict(update=update+1,scheduled_learning_rate=scheduled_rate,reward=float(rewards.mean()),loss=float(metrics[0]),gradient_norm=float(norm))
                if ppo_metrics is not None:row['ppo']=ppo_metrics
                row['monitor']=score(monitor)
                history.append(row);print(json.dumps(dict(seed=seed,**row)),flush=True)
                atomic_json(args.directory/f'progress-{seed}.json',dict(seed=seed,history=history))
        final=score(test)
        iid_rng=np.random.default_rng(80000+seed)
        iid=score([draw(task,iid_rng) for task in args.tasks for _ in range(args.test_cases)])
        path=args.directory/f'seed-{seed}.keras';policy.save(path)
        restored=NeuralPolicy.load(path).model
        check=batch_for(test[:8],encoder,model.memory_size,runtime_history=args.runtime_history)
        np.testing.assert_allclose(np.asarray(model(check)['logits']),np.asarray(restored(check)['logits']),atol=1e-5)
        result=dict(seed=seed,initial=initial,final=final,final_iid=iid,history=history,
                    optimization_steps=optimization_steps,kl_stopped_rollouts=early_stops,guard=guard_stats,
                    calibration_observations=512 if args.standardize_inputs and not args.load else 0,
                    training_examples=args.updates*args.batch,parameters=model.count_params(),seconds=time.monotonic()-start,
                    final_passed=passes(final),
                    iid_passed=passes(iid),
                    retention_passed=(not args.retention_checks or (len(history)>=args.retention_checks and all(passes(h['monitor']) for h in history[-args.retention_checks:]))))
        result['passed']=result['final_passed'] and result['iid_passed'] and result['retention_passed']
        report['runs'].append(result);atomic_json(args.directory/'report.json',report)
        print(json.dumps(result),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=Path('.local/training/scenarios-v4'))
    p.add_argument('--load',type=Path,help='Warm start a compatible model, preserving its frozen feature calibration')
    p.add_argument('--runtime-history',action='store_true',help='Use the same initial observation events as NeuralPolicy.act')
    p.add_argument('--seeds',type=int,nargs='+',default=[41,42,43])
    p.add_argument('--tasks',choices=TASKS,nargs='+',default=list(TASKS))
    p.add_argument('--updates',type=int,default=600)
    p.add_argument('--batch',type=int,default=48)
    p.add_argument('--test-cases',type=int,default=200)
    p.add_argument('--development-only',action='store_true',help='Evaluate training-distribution development states, not held-out states')
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--final-learning-rate',type=float,default=None,help='Linearly decay to this positive rate; default keeps the rate constant')
    p.add_argument('--candidate-interactions',action='store_true')
    p.add_argument('--player-readout',action='store_true')
    p.add_argument('--fact-readout',action='store_true',help='Direct numeric context and referenced-entity facts in the scorer')
    p.add_argument('--stable-heads',action='store_true')
    p.add_argument('--standardize-inputs',action='store_true')
    p.add_argument('--distribution',choices=('legacy','diverse','native'),default='legacy')
    p.add_argument('--identity-dropout',type=float,default=0.)
    p.add_argument('--isolate-value',action='store_true')
    p.add_argument('--outcome-noise',type=float,default=0.,help='Independently flip terminal outcomes with this probability')
    p.add_argument('--advantage-scale-floor',type=float,default=.1)
    p.add_argument('--require-pass',action='store_true')
    p.add_argument('--retention-checks',type=int,default=0,help='Require the final N development checkpoints to pass too')
    p.add_argument('--algorithm',choices=('actor-critic','ppo'),default='actor-critic')
    p.add_argument('--ppo-epochs',type=int,default=4)
    p.add_argument('--ppo-clip',type=float,default=.2)
    p.add_argument('--target-kl',type=float,default=.02)
    p.add_argument('--trust-guard',action='store_true')
    p.add_argument('--max-state-kl',type=float,default=.1)
    p.add_argument('--supervised-control',action='store_true',help='Capacity control with oracle action labels; not RL')
    args=p.parse_args()
    if min(args.updates,args.batch,args.test_cases,args.learning_rate)<=0:p.error('Counts and learning rate must be positive')
    if args.supervised_control and args.algorithm=='ppo':p.error('Supervised control cannot be combined with PPO')
    if args.final_learning_rate is not None and not 0<args.final_learning_rate<=args.learning_rate:p.error('Final learning rate must be positive and no larger than the initial rate')
    if args.retention_checks<0:p.error('Retention checks cannot be negative')
    if args.trust_guard and args.algorithm!='ppo':p.error('Trust guard requires PPO')
    if not 0<=args.identity_dropout<=1:p.error('Identity dropout must be in [0, 1]')
    if not 0<=args.outcome_noise<.5:p.error('Outcome noise must be in [0, .5)')
    if args.outcome_noise and args.supervised_control:p.error('Outcome noise applies only to RL')
    report=run(args)
    if args.require_pass and not all(r['passed'] for r in report['runs']):raise SystemExit(1)
