"""Guarded PPO over a bounded, interned sparse rollout on the learner device.

The loss, Adam updates, finite-gradient checks, clipping and exact KL predicate
are shared with PPOTrainer. This changes the input/model architecture, not
returns, frozen advantages, canonical scans, RNG order or rollback semantics.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import time
import numpy as np
import tensorflow as tf
from .batching import model_example_batch
from .compact_standard import PackedBuilder, VERSION
from .compact_standard_model import TensorStore, CompactActorCritic
from .ppo import PPOTrainer, normalized_advantages, value_prediction_diagnostics
from .rollout_records import read_trajectory
from .streaming_ppo import _guard_status, _json_number
from .update_progress import UpdateProgress


class CompactPPOTrainer(PPOTrainer):
    def __init__(self,model,*,seed=0,input_budget_bytes=512*1024**2,concurrent_updates=True,**kwargs):
        if not isinstance(model,CompactActorCritic):raise ValueError('Compact PPO requires the compact actor/critic architecture')
        if type(input_budget_bytes) is not int or input_budget_bytes<=0:raise ValueError('Compact input budget must be positive integer bytes')
        if kwargs.get('critic_unroll',1)!=1:raise ValueError('Stateless compact PPO requires critic_unroll=1')
        model(model_example_batch(model),training=False)
        super().__init__(model,**kwargs)
        self.input_budget_bytes=input_budget_bytes
        self.concurrent_updates=bool(concurrent_updates)
        self.rng=np.random.default_rng(seed)
        self.prefetch_batches=0;self.measurement_workers=1
        self.device=getattr(model.trainable_variables[0].value,'device','/CPU:0')
        if {id(v) for v in model.actor.variables}&{id(v) for v in model.critic.variables}:
            raise ValueError('Compact actor and critic must not share variables')

    def compile(self):
        b=model_example_batch(self.model);z=np.zeros(1,np.float32);a=np.zeros(1,np.int32);old=np.zeros((1,1),np.float32)
        self.measure(b,a,z,z,z);self.drift(b,old)
        self.measure_with_drift(b,a,z,z,z,old);self.measure_actor_with_drift(b,a,z,z,z,old)

    def update_files(self,trajectories):
        started=time.monotonic();trajectories=list(trajectories)
        progress=UpdateProgress(getattr(self,'progress_callback',None),timeout_seconds=getattr(self,'update_timeout_seconds',None))
        progress.emit('packing',force=True,samples=0,total=sum(t['samples'] for t in trajectories))
        from .compact_dispatch import codec,neural
        builder=codec(self.model).PackedBuilder(self.input_budget_bytes)
        values=[];returns=[];advancing=[];old=[]
        for t in trajectories:
            if t['reward'] not in (-1.,0.,1.):raise ValueError('Expected terminal win/loss/draw')
            seen=0
            for sample in read_trajectory(t['path']):
                if '_compact' not in sample.arrays or np.size(sample.memory):
                    raise ValueError('Compact PPO requires fresh stateless compact trajectories; legacy rollouts cannot be adopted')
                r=sample.arrays['_compact'];mask=sample.arrays['candidate_mask'];p=sample.log_probs
                if (p is None or np.shape(p)!=np.shape(mask) or len(mask)!=len(r.roles) or not mask.all()
                        or not np.isfinite(p).all() or not np.isfinite(sample.value)
                        or not np.isclose(np.exp(p).sum(),1.,atol=1e-5)
                        or not 0<=sample.action<len(mask)):
                    raise ValueError('Invalid frozen compact behavior/action')
                builder.append(r,action=sample.action)
                values.append(sample.value);returns.append(t['reward']);advancing.append(sample.advance)
                old.append(np.asarray(p,np.float32).copy());seen+=1
                if len(values)%10000==0:progress.emit('packing',samples=len(values))
            if seen!=t['samples']:raise ValueError('Incomplete compact trajectory')
        arrays=builder.finish();del builder
        values=np.asarray(values,np.float32);returns=np.asarray(returns,np.float32)
        advancing=np.asarray(advancing,np.float32);count=len(values)
        choices=np.diff(arrays['action_splits'])>1
        baseline=values if self.advantage_baseline=='critic' else np.zeros_like(values)
        adv=normalized_advantages(returns,baseline,choices,self.advantage_scale_floor)
        arrays.update(behavior_log_probs=np.concatenate(old),returns=returns,adv=adv,advancing=advancing)
        input_bytes=sum(a.nbytes for a in arrays.values())
        if input_bytes>self.input_budget_bytes:raise ValueError('Compact rollout exceeds its declared input budget')
        with tf.device(self.device):store=neural(self.model).TensorStore(arrays)
        del arrays,old
        packing_seconds=time.monotonic()-started
        @tf.function(input_signature=[tf.TensorSpec([None],tf.int32)],autograph=False)
        def prepare(ids):
            with tf.device(self.device):
                b=store.batch(ids);b['advance_memory']=tf.gather(store.data['advancing'],ids)[:,None]
                full=CompactActorCritic.dense_logits(tf.gather(store.data['behavior_log_probs'],b['action_indices']),b)
                actions=tf.gather(store.data['action'],ids)
                chosen=tf.gather(full,actions,axis=1,batch_dims=1)
                return b,actions,tf.gather(store.data['returns'],ids),chosen,tf.gather(store.data['adv'],ids),full
        actor_seconds=wait_seconds=measurement_seconds=0.;scans=0
        def chunks(order):
            for i in range(0,count,self.batch_size):yield np.asarray(order[i:i+self.batch_size],np.int32)
        def measure(actor_only=False):
            nonlocal measurement_seconds,scans
            scans+=1;begin=time.monotonic();sums=np.zeros(5);choices_total=values_total=0.;exact_sum=worst=0.
            forward=self.measure_actor_with_drift if actor_only else self.measure_with_drift
            progress.emit('measurement',force=True,scan=scans,actor_only=actor_only,batches_done=0)
            for j,ids in enumerate(chunks(np.arange(count))):
                b,a,r,p,v,full=prepare(ids)
                metrics,drift=map(np.asarray,forward(b,a,r,p,v,full))
                n=int(choices[ids].sum());nv=float(advancing[ids].sum())
                sums+=metrics[1:]*np.asarray([n,nv,n,n,n]);choices_total+=n;values_total+=nv
                exact_sum+=float(drift[0])*n;worst=max(worst,float(drift[1]))
                progress.emit('measurement',scan=scans,actor_only=actor_only,batches_done=j+1,
                    batches_total=(count+self.batch_size-1)//self.batch_size)
            denom=np.asarray([choices_total,values_total,choices_total,choices_total,choices_total])
            terms=np.divide(sums,denom,out=np.zeros_like(sums),where=denom>0)
            loss=terms[0]+self.value_weight*terms[1]-self.entropy_weight*terms[2]
            measurement_seconds+=time.monotonic()-begin
            return np.r_[loss,terms],np.asarray([exact_sum/max(choices_total,1),worst])
        metrics,exact=measure();value_loss_before=float(metrics[2])
        # Reserve identical critic permutations before the actor touches RNG,
        # including epochs not consumed because the critic hits its step cap.
        orders=iter([self.rng.permutation(count) for _ in range(self.epochs)])
        def critic_batches():
            for ids in chunks(next(orders)):
                b,_,r,_,_,_=prepare(ids);yield b,r
        def fit_values():
            with tf.device(self.device):return self.fit_critic(critic_batches)
        steps=rejected=0;norm=0.;stopped=False;largest=np.zeros(2);proposals=[]
        pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='compact-critic') if self.concurrent_updates else nullcontext()
        # Every reader is joined before the store is released or a checkpoint
        # can publish. Frozen advantages/probabilities are never refreshed.
        with pool as executor:
            future=executor.submit(fit_values) if executor else None
            critic_stats={} if future else fit_values()
            for epoch in range(self.epochs):
                saved=self.snapshot();rate=self.learning_rate;order=self.rng.permutation(count);accepted=False
                for attempt in range(self.backtracks+1):
                    self.optimizer.learning_rate.assign(rate*.5**attempt);epoch_steps=0
                    progress.emit('actor_epoch',force=True,epoch=epoch+1,attempt=attempt+1,batches_done=0,
                        batches_total=(count+self.batch_size-1)//self.batch_size)
                    for ids in chunks(order):
                        before=time.monotonic();b,a,r,p,v,_=prepare(ids);wait_seconds+=time.monotonic()-before
                        if future and future.done():future.result()
                        before=time.monotonic();norm=float(self.step(b,a,r,p,v));actor_seconds+=time.monotonic()-before
                        epoch_steps+=1
                        progress.emit('actor_epoch',epoch=epoch+1,attempt=attempt+1,batches_done=epoch_steps,
                            batches_total=(count+self.batch_size-1)//self.batch_size,
                            actor_step_seconds=actor_seconds,actor_input_wait_seconds=wait_seconds)
                    proposed,drift=measure(actor_only=True)
                    accepted,finite,reasons=_guard_status(proposed,drift,self.target_kl,self.max_state_kl)
                    event=dict(epoch=epoch+1,attempt=attempt,accepted=accepted,finite=finite,rejection_reasons=reasons,
                        mean_kl=_json_number(drift[0]),max_state_kl=_json_number(drift[1]),effective_learning_rate=float(self.optimizer.learning_rate.numpy()))
                    proposals.append(event)
                    progress.emit('guard_result',force=True,**event)
                    if accepted:
                        metrics,exact=proposed,drift;largest=np.maximum(largest,drift);steps+=epoch_steps;break
                    self.restore(saved);rejected+=1
                if not accepted:stopped=True;break
            if future:critic_stats=future.result()
        metrics,exact=measure()
        if not np.isfinite(metrics).all() or not np.isfinite(exact).all():raise FloatingPointError('Nonfinite final compact PPO metrics')
        result=dict(zip(('loss','policy_loss','value_loss','entropy','approx_kl','clip_fraction'),map(float,metrics)),
            samples=count,optimizer_steps=int(self.optimizer.iterations),accepted_steps=steps,rejected_epochs=rejected,
            early_stop=stopped,gradient_norm=norm,exact_kl=float(exact[0]),max_state_kl=float(exact[1]),
            largest_accepted_kl=float(largest[0]),largest_accepted_state_kl=float(largest[1]),
            reward_std=float(returns.std()),advantage_std=float(adv.std()),advantage_baseline=self.advantage_baseline,
            value_loss_before=value_loss_before,behavior_critic=value_prediction_diagnostics(returns,values,advancing.astype(bool)),
            sample_order='global_shuffle_v1',update_schedule='concurrent' if self.concurrent_updates else 'sequential',
            recurrent_updates='stateless current observation',terminal_rewards_only=True,guard_proposals=proposals,
            pipeline=dict(profile=self.model.input_format,input_bytes=input_bytes,input_budget_bytes=self.input_budget_bytes,
                prefetch_batches=0,measurement_workers=1,device=self.device,
                timing=dict(packing_seconds=packing_seconds,actor_step_seconds=actor_seconds,
                    actor_input_wait_seconds=wait_seconds,measurement_seconds=measurement_seconds,
                    measurement_scans=scans,elapsed_seconds=time.monotonic()-started)),**critic_stats)
        progress.emit('complete',force=True,accepted_steps=steps,rejected_epochs=rejected,timing=result['pipeline']['timing'])
        return result
