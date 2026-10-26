"""Verify numerical packing and fused output/gradient parity on a frozen round.

All canonical batches and six seeded shuffled batches must be byte-for-byte
identical to the original collation. Three real batches also compare GPU outputs
and a single joint backward traversal. No optimizer steps or generated games.
"""
import argparse
import copy
import itertools
import os
from pathlib import Path
import json
from examples.profile_ppo_update import checkpoint_view, select_training, sha, write


def run(args):
    for key,value in {'KERAS_BACKEND':'tensorflow','CUDA_VISIBLE_DEVICES':'0','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'4','TF_ENABLE_ONEDNN_OPTS':'0','TF_CPP_MIN_LOG_LEVEL':'2'}.items():os.environ.setdefault(key,value)
    import numpy as np
    import keras
    import tensorflow as tf
    from patchwork_automaton.neural.devices import configure_runtime,device_scope
    from patchwork_automaton.neural.experiment import load_checkpoint,read_trajectory
    from patchwork_automaton.neural.features import collate
    from patchwork_automaton.neural.packed_rollouts import NumericRollout,PolicyBatchCache
    from patchwork_automaton.neural.rollout_cache import IndexedTrajectories,RolloutCache
    from patchwork_automaton.neural.ppo import normalized_advantages
    from patchwork_automaton.neural.batching import input_signature
    from examples.benchmark_fused_gru import compare_arrays,graph_ops
    from examples.train_rustmage import array_digest
    import tempfile
    configure_runtime('gpu');args.output.mkdir(parents=True,exist_ok=False)
    source=args.source.resolve();report_path=source/'training/report.json';saved=json.loads(report_path.read_text());row,paths=select_training(saved,40)
    checkpoint=source/'training/checkpoints/round-0039';protected={str(p):sha(p) for p in checkpoint.iterdir() if p.is_file()};protected[str(report_path)]=sha(report_path)
    inputs={t['path']:sha(t['path']) for t in paths}
    keras.utils.set_random_seed(saved['config']['seed'])
    with device_scope('gpu'),checkpoint_view(checkpoint) as view:
        model,trainer,state=load_checkpoint(view,gru_backend='cudnn-memory')
    initial=[array_digest(v) for v in (model.weights,trainer.optimizer.variables,trainer.critic_optimizer.variables)];rng=copy.deepcopy(trainer.rng.bit_generator.state)
    result=dict(status='running',training_games=len(row['split']['training_indices']),heldout_games_excluded=len(row['split']['validation_indices']),source_hashes=protected,input_hashes=inputs,canonical_batches=0,shuffled_batches=0,cases=[])
    write(args.output/'report.json',result)
    store=NumericRollout();cache=PolicyBatchCache();values=[];returns=[];choices=[]
    def records():
        for t in paths:
            for sample in read_trajectory(t['path']):yield sample,t['reward']
    def reference(samples):
        x=collate([s.arrays for s in samples]);x['memory_in']=np.stack([s.memory for s in samples]);x['advance_memory']=np.asarray([[float(s.advance)] for s in samples],np.float32)
        old=np.full(x['candidate_mask'].shape,-1.e9,np.float32)
        for i,s in enumerate(samples):old[i,:len(s.log_probs)]=s.log_probs
        return x,np.asarray([s.action for s in samples],np.int32),old
    def candidate(indices):
        x=store.gather(indices);old=x.pop('_old');actions=x.pop('_action')[:,0];return x,actions,old
    def equal(a,b):
        aa=[*a[0].values(),*a[1:]];bb=[b[0][k] for k in a[0]]+list(b[1:])
        assert len(aa)==len(bb)
        for u,v in zip(aa,bb):
            assert u.dtype==v.dtype and u.shape==v.shape
            np.testing.assert_array_equal(u,v)
    try:
        for sample,reward in records():
            store.append(sample);values.append(sample.value);returns.append(reward);choices.append(sample.arrays['candidate_mask'].sum()>1)
        store.finish();returns=np.asarray(returns,np.float32);advantages=normalized_advantages(returns,values,np.asarray(choices),trainer.advantage_scale_floor)
        stream=iter(records());start=0;gpu_batches=[]
        while chunk:=list(itertools.islice(stream,512)):
            indices=np.arange(start,start+len(chunk));samples=[s for s,_ in chunk];a=reference(samples);b=candidate(indices);equal(a,b)
            cache.put(tuple(indices),(b[0],b[1],returns[indices],b[2][np.arange(len(chunk)),b[1]],advantages[indices],b[2]))
            if start==0 or start==512*(store.count//1024) or start+len(chunk)==store.count:gpu_batches.append((indices,a))
            start+=len(chunk);result['canonical_batches']+=1
        assert start==store.count;cache.finish()
        with tempfile.TemporaryFile() as spill:
            indexed=IndexedTrajectories(paths,RolloutCache(0),read_trajectory,spill);offsets=np.cumsum([0]+[t['samples'] for t in paths])
            try:
                order=np.random.default_rng(538126).permutation(store.count)
                for i in range(6):
                    indices=order[i*512:(i+1)*512];samples=[indexed.get(int(np.searchsorted(offsets,p,side='right')-1),int(p-offsets[np.searchsorted(offsets,p,side='right')-1])) for p in indices]
                    equal(reference(samples),candidate(indices));result['shuffled_batches']+=1
            finally:indexed.close()
        # Build the same joint forward/backward graph once. Both operands use
        # selective cuDNN, identical weights, local padding and FP32 arithmetic.
        signature=input_signature(gpu_batches[0][1][0])
        @tf.function(input_signature=[signature,tf.TensorSpec([None],tf.int32),*[tf.TensorSpec([None],tf.float32)]*3],autograph=False)
        def evaluate(x,actions,rewards,old,adv):
            ax={k:tf.identity(v) for k,v in x.items()};cx={k:tf.identity(v) for k,v in x.items()}
            watched=[mapping[k] for mapping in (ax,cx) for k in ('entities','semantics','history','memory_in')]
            with tf.GradientTape() as tape:
                tape.watch(watched);actor=model.actor_output(ax,training=True);critic=model.value_output(cx,training=True)
                terms=trainer._terms_from_output(actor,tf.nn.log_softmax(actor['logits']),x,actions,rewards,old,adv)
                al=terms[1]-trainer.entropy_weight*terms[3];mask=x['advance_memory'][:,0]
                cl=tf.math.divide_no_nan(tf.reduce_sum(tf.square(critic['value'][:,0]-rewards)*mask),tf.reduce_sum(mask));loss=al+cl
            variables=model.actor.trainable_variables+model.critic.trainable_variables+watched
            grads=tuple(tf.zeros_like(v) if g is None else tf.convert_to_tensor(g) for g,v in zip(tape.gradient(loss,variables),variables))
            return (dict(logits=actor['logits'],probabilities=tf.nn.softmax(actor['logits']),value=critic['value'],actor_memory=actor['memory_out'],critic_memory=critic['memory_out'],actor_loss=al,critic_loss=cl),grads)
        with device_scope('gpu'):
            for indices,a in gpu_batches:
                b=cache.get(tuple(indices));equal(a,(b[0],b[1],b[-1]))
                x,actions,old=a;chosen=old[np.arange(len(indices)),actions]
                left=evaluate(x,actions,returns[indices],chosen,advantages[indices]);right=evaluate(*b[:5])
                outputs={k:compare_arrays(v,right[0][k],atol=2e-5,rtol=2e-4) for k,v in left[0].items()}
                gradients=[compare_arrays(u,v,atol=2e-5,rtol=2e-3) for u,v in zip(left[1],right[1])]
                all_a=np.concatenate([np.asarray(g).ravel() for g in left[1]]);all_b=np.concatenate([np.asarray(g).ravel() for g in right[1]])
                total=compare_arrays(all_a,all_b,atol=2e-5,rtol=2e-3)
                assert all(s['finite'] and s['close'] for s in [*outputs.values(),*gradients]) and total['relative_l2']<=1e-3
                result['cases'].append(dict(first_index=int(indices[0]),count=len(indices),outputs=outputs,gradient_max_absolute=max(s['max_absolute'] for s in gradients),gradient_relative_l2=total['relative_l2']))
        ops=graph_ops(evaluate);assert ops.get('CudnnRNNV3')==ops.get('CudnnRNNBackpropV3')==4
        result.update(status='passed',samples=store.count,numeric_disk_bytes=store.disk_bytes,policy_cache_disk_bytes=cache.disk_bytes,cudnn_ops={k:v for k,v in ops.items() if 'Cudnn' in k},weights_and_optimizers_unchanged=initial==[array_digest(v) for v in (model.weights,trainer.optimizer.variables,trainer.critic_optimizer.variables)],rng_unchanged=rng==trainer.rng.bit_generator.state,source_unchanged=all(sha(p)==h for p,h in {**protected,**inputs}.items()))
        assert all(result[k] for k in ('weights_and_optimizers_unchanged','rng_unchanged','source_unchanged'))
        write(args.output/'report.json',result)
    finally:
        # Drop last cached mapped input views before releasing temporary files.
        cache.close();store.close()
    print(json.dumps({k:result[k] for k in ['status','samples','canonical_batches','shuffled_batches','numeric_disk_bytes','policy_cache_disk_bytes']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
