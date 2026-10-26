"""Strict cuDNN equivalence checks on frozen, training-only PPO records.

No optimizer steps, games, or checkpoint writes. The separate replay driver
benchmarks full updates only after these output/gradient checks pass.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time


def compare_arrays(reference, candidate, *, atol, rtol):
    import numpy as np
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape:
        raise ValueError(f'Shape mismatch {a.shape} != {b.shape}')
    delta = b.astype('float64') - a.astype('float64')
    denominator = float(np.linalg.norm(a.astype('float64')))
    return dict(shape=list(a.shape), finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
        max_absolute=float(np.max(np.abs(delta), initial=0.)),
        relative_l2=float(np.linalg.norm(delta)/max(denominator, 1e-12)),
        close=bool(np.allclose(a, b, atol=atol, rtol=rtol)))


def graph_ops(function):
    from collections import Counter
    graph = function.get_concrete_function().graph.as_graph_def()
    counts = Counter(n.op for n in graph.node)
    for library in graph.library.function:
        counts.update(n.op for n in library.node_def)
    return dict(sorted(counts.items()))


def run(args):
    for k,v in {'KERAS_BACKEND':'tensorflow','CUDA_VISIBLE_DEVICES':'0',
                'TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'4',
                'TF_ENABLE_ONEDNN_OPTS':'0','TF_CPP_MIN_LOG_LEVEL':'2'}.items():os.environ.setdefault(k,v)
    import keras
    import numpy as np
    import tensorflow as tf
    from patchwork_automaton.neural.devices import configure_runtime, device_scope, assert_model_device
    from patchwork_automaton.neural.experiment import load_checkpoint, read_trajectory
    from patchwork_automaton.neural.batching import input_signature
    from patchwork_automaton.neural.features import collate
    from patchwork_automaton.neural.recurrent import configure_gru_backend, masked_gru
    from examples.profile_ppo_update import checkpoint_view, select_training, sha, write
    from examples.train_rustmage import array_digest
    configure_runtime('gpu')
    args.output.mkdir(exist_ok=False, parents=True)
    source = args.source.resolve()
    report = json.loads((source/'training/report.json').read_text())
    row, trajectories = select_training(report, 40)
    checkpoint = source/'training/checkpoints/round-0039'
    protected = {str(p):sha(p) for p in checkpoint.iterdir() if p.is_file()}
    protected[str(source/'training/report.json')] = sha(source/'training/report.json')
    keras.utils.set_random_seed(report['config']['seed'])
    with device_scope('gpu'), checkpoint_view(checkpoint) as view:
        model, trainer, state = load_checkpoint(view)
    initial_weights = array_digest(model.weights)
    initial_optimizer = array_digest(trainer.optimizer.variables)
    initial_critic_optimizer = array_digest(trainer.critic_optimizer.variables)
    initial_rng = copy.deepcopy(trainer.rng.bit_generator.state)
    results = dict(status='running',tolerances=dict(outputs=dict(atol=2e-5,rtol=2e-4),
        gradients=dict(atol=2e-5,rtol=2e-3),gradient_global_relative_l2=1e-3),
        training_games=800,heldout_games_excluded=200,source_hashes=protected,
        placements=assert_model_device(model,'gpu'),initial_weights=initial_weights,
        gru_backend=args.gru_backend,cases=[])
    write(args.output/'report.json',results)
    # Deterministic, outcome-independent trajectory selection spanning the round.
    # Each case retains real detached memories, actions and frozen probabilities.
    batches=[]
    for offset in (0, len(trajectories)//3, 2*len(trajectories)//3):
        records=[]; rewards=[]; paths=[]
        for trajectory in trajectories[offset:]:
            paths.append(trajectory['path'])
            for sample in read_trajectory(trajectory['path']):
                records.append(sample); rewards.append(trajectory['reward'])
                if len(records)==512:break
            if len(records)==512:break
        if len(records)!=512:raise ValueError('Incomplete parity batch')
        x=collate([s.arrays for s in records]);x['memory_in']=np.stack([s.memory for s in records])
        x['advance_memory']=np.array([[s.advance] for s in records],np.float32)
        returns=np.array(rewards,np.float32);actions=np.array([s.action for s in records],np.int32)
        old=np.array([s.log_probs[s.action] for s in records],np.float32)
        advantages=returns-np.array([s.value for s in records],np.float32)
        batches.append((f'real-{offset}', x, actions, returns, old, advantages,
                        {p:sha(p) for p in paths}))
    # Explicit all-empty histories and semantics, with nonzero incoming memory;
    # also test selection-only decisions that must preserve that memory exactly.
    edge=[np.copy(v) for v in batches[0][1].values()]
    x=dict(zip(batches[0][1],edge));x['semantic_mask'][:]=False;x['history_mask'][:]=False
    x['advance_memory'][::2]=0
    batches.append(('all-empty-mixed-advance',x,*batches[0][2:]))
    failures=[]
    for label, x, actions, returns, old, advantages, paths in batches:
        for key in ('semantic_mask','history_mask'):
            mask=x[key].reshape(-1,x[key].shape[-1])
            if np.any(mask[:,1:] & ~mask[:,:-1]):raise ValueError('Non-right-padded real mask')
        def make_function():
            @tf.function(input_signature=[input_signature(x),tf.TensorSpec([None],tf.int32)] +
                         [tf.TensorSpec([None],tf.float32)]*3,autograph=False)
            def evaluate(inputs, actions, returns, old, advantages):
                # cuDNN's backward consumes its forward reserve space. Request
                # every derivative in ONE traversal; persistent-tape multiple
                # gradient() calls over a shared forward can corrupt results.
                # Separate input identities let this one traversal expose actor
                # and critic input derivatives independently.
                actor_inputs={k:tf.identity(v) for k,v in inputs.items()}
                critic_inputs={k:tf.identity(v) for k,v in inputs.items()}
                watched=[mapping[k] for mapping in (actor_inputs,critic_inputs)
                         for k in ('entities','semantics','history','memory_in')]
                with tf.GradientTape() as tape:
                    tape.watch(watched)
                    actor=model.actor_output(actor_inputs,training=True)
                    critic=model.value_output(critic_inputs,training=True)
                    terms=trainer._terms_from_output(actor,tf.nn.log_softmax(actor['logits']),
                        inputs,actions,returns,old,advantages)
                    actor_loss=terms[1]-trainer.entropy_weight*terms[3]
                    mask=inputs['advance_memory'][:,0]
                    critic_loss=tf.math.divide_no_nan(tf.reduce_sum(tf.square(critic['value'][:,0]-returns)*mask),tf.reduce_sum(mask))
                    loss=actor_loss+critic_loss
                av=model.actor.trainable_variables;cv=model.critic.trainable_variables
                variables=av+cv+watched
                gradients=tuple(tf.zeros_like(v) if g is None else tf.convert_to_tensor(g)
                                for g,v in zip(tape.gradient(loss,variables),variables))
                na,nc=len(av),len(cv)
                return (dict(logits=actor['logits'],probabilities=tf.nn.softmax(actor['logits']),
                        actor_memory=actor['memory_out'],value=critic['value'],critic_memory=critic['memory_out'],
                        actor_loss=actor_loss,critic_loss=critic_loss),
                        gradients[:na],gradients[na:na+nc],
                        gradients[na+nc:na+nc+4],gradients[na+nc+4:])
            return evaluate
        with device_scope('gpu'):
            configure_gru_backend(model,'unfused'); a=make_function(); left=a(x,actions,returns,old,advantages)
            configure_gru_backend(model,args.gru_backend); b=make_function(); right=b(x,actions,returns,old,advantages)
        ops=graph_ops(b)
        expected = 4 if args.gru_backend == 'cudnn-memory' else 6
        if ops.get('CudnnRNNV3',0)!=expected or ops.get('CudnnRNNBackpropV3',0)!=expected:
            raise ValueError('Unexpected fused GRU forward/backward count')
        case=dict(label=label,batch_size=512,input_shapes={k:list(v.shape) for k,v in x.items()},
                  empty_semantic_rows=int((~x['semantic_mask'].any(axis=-1)).sum()),
                  empty_histories=int((~x['history_mask'].any(axis=-1)).sum()),
                  input_hashes=paths,outputs={},gradients={},cudnn_graph_ops={k:v for k,v in ops.items() if 'Cudnn' in k})
        for key,value in left[0].items():
            case['outputs'][key]=compare_arrays(value,right[0][key],**results['tolerances']['outputs'])
        for i,name in enumerate(('actor_weights','critic_weights','actor_inputs','critic_inputs'),1):
            stats=[compare_arrays(a,b,**results['tolerances']['gradients']) for a,b in zip(left[i],right[i])]
            aa=np.concatenate([np.asarray(v).ravel() for v in left[i]])
            bb=np.concatenate([np.asarray(v).ravel() for v in right[i]])
            total=compare_arrays(aa,bb,**results['tolerances']['gradients'])
            case['gradients'][name]=dict(variables=stats,total=total)
            if total['relative_l2']>1e-3:failures.append(f'{label}:{name}:relative_l2')
            if not all(s['finite'] and s['close'] for s in stats):failures.append(f'{label}:{name}:allclose')
        if not all(s['finite'] and s['close'] for s in case['outputs'].values()):failures.append(f'{label}:outputs')
        # Inference and training use the same stateless GRU forward equations.
        with device_scope('gpu'):
            configure_gru_backend(model,'unfused'); inference_a=model(x,training=False)
            configure_gru_backend(model,args.gru_backend); inference_b=model(x,training=False)
        case['inference']={k:compare_arrays(v,inference_b[k],**results['tolerances']['outputs']) for k,v in inference_a.items()}
        if not all(s['finite'] and s['close'] for s in case['inference'].values()):failures.append(f'{label}:inference')
        results['cases'].append(case);write(args.output/'report.json',results)
        print(json.dumps(dict(label=label,failures=failures,cudnn=case['cudnn_graph_ops'])),flush=True)
        del a,b
    results.update(status='passed' if not failures else 'failed',failures=failures,
        weights_unchanged=initial_weights==array_digest(model.weights),
        optimizers_unchanged=initial_optimizer==array_digest(trainer.optimizer.variables) and initial_critic_optimizer==array_digest(trainer.critic_optimizer.variables),
        rng_unchanged=initial_rng==trainer.rng.bit_generator.state,
        sources_unchanged=all(sha(p)==h for p,h in protected.items()))
    write(args.output/'report.json',results)
    if failures or not all(results[k] for k in ('weights_unchanged','optimizers_unchanged','rng_unchanged','sources_unchanged')):
        raise RuntimeError('Fused GRU parity failed; see report')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gru-backend',choices=['cudnn','cudnn-memory'],default='cudnn')
    run(p.parse_args())
