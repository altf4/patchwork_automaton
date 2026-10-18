"""Streaming on-policy updates and atomic round-boundary checkpoints."""
import gzip
import itertools
import json
from pathlib import Path
import pickle
import shutil
import numpy as np
import tensorflow as tf

from .batching import example_batch, input_signature
from .features import Encoder, collate
from .policy import NeuralPolicy
from .training import TerminalActorCritic


class TrajectoryWriter:
    def __init__(self,path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file = gzip.open(self.path,'wb',compresslevel=1)
        self.count = 0
    def write(self,sample):
        # Only locally generated trusted rollouts are read back by this module.
        pickle.dump(sample,self.file,protocol=5)
        self.count += 1
    def close(self): self.file.close()


def read_trajectory(path):
    with gzip.open(path,'rb') as stream:
        while True:
            try: yield pickle.load(stream)
            except EOFError: return


class StreamingTrainer(TerminalActorCritic):
    def compile(self):
        batch = example_batch(self.model.memory_size, self.model.feature_spec)
        self.model(batch,training=False)
        self.optimizer.build(self.model.trainable_variables)
        signature=[input_signature(batch),tf.TensorSpec([None],tf.int32),
                   tf.TensorSpec([None],tf.float32),tf.TensorSpec([None],tf.float32)]
        self.gradient_fn=tf.function(self._gradients,input_signature=signature,autograph=False)
        self.gradient_fn(batch,np.zeros(1,np.int32),np.ones(1,np.float32),np.zeros(1,np.float32))

    def _gradients(self,inputs,actions,returns,old_values):
        with tf.GradientTape() as tape:
            output=self.model(inputs,training=True)
            logp=tf.nn.log_softmax(output['logits'])
            chosen=tf.gather(logp,actions,axis=1,batch_dims=1)
            entropy=-tf.reduce_sum(tf.exp(logp)*logp,axis=1)
            actor=-tf.reduce_mean(chosen*tf.stop_gradient(returns-old_values))
            critic=tf.reduce_mean(tf.square(tf.squeeze(output['value'],-1)-returns)*inputs['advance_memory'][:,0])
            exploration=tf.reduce_mean(entropy)
            loss=actor+self.value_weight*critic-self.entropy_weight*exploration
        gradients=tape.gradient(loss,self.model.trainable_variables)
        gradients=[tf.zeros_like(v) if g is None else tf.convert_to_tensor(g)
                   for g,v in zip(gradients,self.model.trainable_variables)]
        return gradients,tf.stack([loss,actor,critic,exploration])

    def update_files(self,trajectories):
        total=sum(t['samples'] for t in trajectories)
        if not total: raise ValueError('Empty training round')
        def stream():
            for trajectory in trajectories:
                if trajectory['reward'] not in (-1.,0.,1.): raise ValueError('Invalid terminal reward')
                seen=0
                for sample in read_trajectory(trajectory['path']):
                    seen+=1
                    yield sample,trajectory['reward']
                if seen!=trajectory['samples']: raise ValueError('Incomplete trajectory file')
        iterator=stream(); accumulated=[tf.zeros_like(v) for v in self.model.trainable_variables]
        totals=np.zeros(4,np.float64); seen=0
        while chunk:=list(itertools.islice(iterator,self.batch_size)):
            samples,returns=zip(*chunk)
            inputs=collate([s.arrays for s in samples])
            inputs['memory_in']=np.stack([s.memory for s in samples])
            inputs['advance_memory']=np.asarray([[float(s.advance)] for s in samples],np.float32)
            gradients,metrics=self.gradient_fn(inputs,np.asarray([s.action for s in samples],np.int32),
                np.asarray(returns,np.float32),np.asarray([s.value for s in samples],np.float32))
            weight=len(samples)/total
            accumulated=[a+weight*g for a,g in zip(accumulated,gradients)]
            totals+=weight*np.asarray(metrics); seen+=len(samples)
        if seen!=total: raise ValueError('Trajectory count mismatch')
        for gradient in accumulated:tf.debugging.assert_all_finite(gradient,'Nonfinite gradient')
        clipped,norm=tf.clip_by_global_norm(accumulated,self.clip_norm)
        self.optimizer.apply_gradients(zip(clipped,self.model.trainable_variables))
        return dict(zip(('loss','policy_loss','value_loss','entropy'),totals.tolist()),
                    samples=total,gradient_norm=float(norm),optimizer_steps=int(self.optimizer.iterations))


def atomic_json(path,value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def save_checkpoint(directory,model,trainer,state,*,policy_type=NeuralPolicy):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    name=f"round-{state['round']:04d}"
    destination=directory/name
    if destination.exists(): raise FileExistsError(destination)
    temporary=directory/(name+'.tmp')
    if temporary.exists():shutil.rmtree(temporary)
    temporary.mkdir()
    policy_type(model).save(temporary/'policy.keras')
    np.savez(temporary/'optimizer.npz',**{f'v{i}':np.asarray(v) for i,v in enumerate(trainer.optimizer.variables)})
    saved_state=dict(state,optimizer_steps=int(trainer.optimizer.iterations))
    if hasattr(trainer, 'critic_optimizer'):
        np.savez(temporary/'critic-optimizer.npz', **{f'v{i}':np.asarray(v) for i,v in enumerate(trainer.critic_optimizer.variables)})
        saved_state['critic_optimizer_steps'] = int(trainer.critic_optimizer.iterations)
    if hasattr(trainer,'rng'):saved_state['trainer_rng_state']=trainer.rng.bit_generator.state
    atomic_json(temporary/'state.json',saved_state)
    temporary.rename(destination)
    atomic_json(directory/'latest.json',{'checkpoint':name})
    return destination


def load_checkpoint(directory):
    directory=Path(directory)
    name=json.loads((directory/'latest.json').read_text())['checkpoint']
    if Path(name).name!=name:raise ValueError('Invalid checkpoint path')
    root=directory/name
    state=json.loads((root/'state.json').read_text())
    manifest=json.loads((root/'policy.manifest.json').read_text())
    if manifest.get('backend') == 'rustmage':
        from .rustmage_policy import RustMagePolicy
        model=RustMagePolicy.load(root/'policy.keras').model
    else:
        model=NeuralPolicy.load(root/'policy.keras').model
    config=state['config']
    trainer=make_trainer(model,config)
    trainer.compile()
    with np.load(root/'optimizer.npz',allow_pickle=False) as archive:
        if len(archive.files)!=len(trainer.optimizer.variables):raise ValueError('Optimizer variable mismatch')
        for i,variable in enumerate(trainer.optimizer.variables):
            value=archive[f'v{i}']
            if tuple(value.shape)!=tuple(variable.shape):raise ValueError('Optimizer shape mismatch')
            variable.assign(value)
    if int(trainer.optimizer.iterations)!=state.get('optimizer_steps',state['round']):raise ValueError('Optimizer/checkpoint mismatch')
    if hasattr(trainer, 'critic_optimizer'):
        with np.load(root/'critic-optimizer.npz', allow_pickle=False) as archive:
            if len(archive.files) != len(trainer.critic_optimizer.variables): raise ValueError('Critic optimizer variable mismatch')
            for i, variable in enumerate(trainer.critic_optimizer.variables):
                value = archive[f'v{i}']
                if tuple(value.shape) != tuple(variable.shape): raise ValueError('Critic optimizer shape mismatch')
                variable.assign(value)
        if int(trainer.critic_optimizer.iterations) != state['critic_optimizer_steps']:
            raise ValueError('Critic optimizer/checkpoint mismatch')
    if hasattr(trainer,'rng') and 'trainer_rng_state' in state:
        trainer.rng.bit_generator.state=state['trainer_rng_state']
    return model,trainer,state


def make_trainer(model,config):
    if config.get('algorithm','actor-critic')=='ppo':
        from .streaming_ppo import StreamingPPOTrainer
        return StreamingPPOTrainer(model,learning_rate=config['learning_rate'],batch_size=config['train_batch'],
            seed=config.get('seed',0)+5000,epochs=config.get('ppo_epochs',2),clip_ratio=config.get('ppo_clip',.2),
            target_kl=config.get('target_kl',.01),max_state_kl=config.get('max_state_kl',.05),
            advantage_scale_floor=config.get('advantage_scale_floor',1.),trust_guard=True)
    if config.get('algorithm','actor-critic')!='actor-critic':raise ValueError('Unknown training algorithm')
    return StreamingTrainer(model,learning_rate=config['learning_rate'],batch_size=config['train_batch'])
