"""Read-only per-layer GRU timing on captured, real round-40 tensors.

Diagnostic only: no optimizer, new games, checkpoint writes, or RL changes.
The forward-plus-backward cases request every derivative in one traversal.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def run(args):
    for k,v in {'KERAS_BACKEND':'tensorflow','CUDA_VISIBLE_DEVICES':'0',
                'TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'4',
                'TF_ENABLE_ONEDNN_OPTS':'0','TF_CPP_MIN_LOG_LEVEL':'2'}.items():os.environ.setdefault(k,v)
    import keras
    import numpy as np
    import tensorflow as tf
    from patchwork_automaton.neural.devices import configure_runtime,device_scope
    from patchwork_automaton.neural.experiment import load_checkpoint,read_trajectory
    from patchwork_automaton.neural.features import collate
    from patchwork_automaton.neural.recurrent import masked_gru
    from examples.profile_ppo_update import checkpoint_view,select_training,write,sha
    from examples.train_rustmage import array_digest
    configure_runtime('gpu');args.output.mkdir(exist_ok=False,parents=True)
    saved=json.loads((args.source/'training/report.json').read_text());_,trajectories=select_training(saved,40)
    checkpoint=args.source/'training/checkpoints/round-0039';protected={str(p):sha(p) for p in checkpoint.iterdir() if p.is_file()}
    keras.utils.set_random_seed(saved['config']['seed'])
    with device_scope('gpu'),checkpoint_view(checkpoint) as view:model,trainer,_=load_checkpoint(view)
    before=array_digest(model.weights)
    records=[];paths=[]
    for t in trajectories:
        paths.append(t['path'])
        records.extend(read_trajectory(t['path']))
        if len(records)>=512:records=records[:512];break
    x=collate([s.arrays for s in records]);x['memory_in']=np.stack([s.memory for s in records]);x['advance_memory']=np.array([[s.advance] for s in records],np.float32)
    layers=[(side+'.'+name,getattr(getattr(model,side),name)) for side in ['actor','critic'] for name in ['semantic_gru','history_memory','memory']]
    captured={};originals=[]
    def observe(label,layer,original):
        def call(sequences,initial_state=None,mask=None,training=False):
            initial=initial_state[0] if isinstance(initial_state,(list,tuple)) else initial_state
            captured[label]=(np.asarray(sequences),None if mask is None else np.asarray(mask),
                             np.zeros((len(sequences),layer.units),np.float32) if initial is None else np.asarray(initial))
            return original(sequences,initial_state=initial_state,mask=mask,training=training)
        return call
    try:
        for label,layer in layers:
            originals.append((layer,layer.call));layer.call=observe(label,layer,layer.call)
        with device_scope('gpu'):model(x,training=True)
    finally:
        for layer,original in originals:layer.call=original
    result=dict(status='running',batch_size=512,steps_per_repeat=20,repeats=3,order=['unfused','cudnn','cudnn','unfused'],
                input_hashes={p:sha(p) for p in paths},cases=[])
    write(args.output/'report.json',result)
    for label,layer in layers:
        values,mask,initial=captured[label];has_mask=mask is not None
        if mask is None:mask=np.ones(values.shape[:2],bool)
        with device_scope('gpu'):
            inputs=(tf.constant(values),tf.constant(mask),tf.constant(initial))
        for backward in [False,True]:
            for backend in result['order']:
                layer.use_cudnn=backend=='cudnn';layer.supports_jit=False
                def function():
                    @tf.function(input_signature=[tf.TensorSpec([None,None,values.shape[-1]],tf.float32),
                        tf.TensorSpec([None,None],tf.bool),tf.TensorSpec([None,layer.units],tf.float32)],autograph=False)
                    def evaluate(v,m,s):
                        with tf.GradientTape() as tape:
                            tape.watch([v,s])
                            if has_mask:y=masked_gru(layer,v,mask=m,initial_state=s,training=True)
                            else:y=layer(v,initial_state=s,training=True)
                            if layer.return_state:y=y[1]
                            loss=tf.reduce_mean(tf.square(y))
                        if backward:
                            gradients=tape.gradient(loss,layer.trainable_variables+[v,s])
                            return tf.add_n([tf.reduce_sum(tf.square(tf.convert_to_tensor(g))) for g in gradients if g is not None])
                        return tf.reduce_sum(y)
                    return evaluate
                with device_scope('gpu'):
                    fn=function()
                    for _ in range(5):fn(*inputs).numpy()
                    timings=[]
                    for _ in range(result['repeats']):
                        started=time.perf_counter()
                        for _ in range(result['steps_per_repeat']):fn(*inputs).numpy()
                        timings.append(1000*(time.perf_counter()-started)/result['steps_per_repeat'])
                case=dict(layer=label,backend=backend,operation='forward_backward' if backward else 'forward',
                          shape=list(values.shape),empty_rows=int((~mask.any(axis=1)).sum()),
                          repeat_ms=timings,median_ms=statistics.median(timings))
                result['cases'].append(case);write(args.output/'report.json',result)
                print(json.dumps(case),flush=True)
    result.update(status='complete',weights_unchanged=before==array_digest(model.weights),
                  source_unchanged=all(sha(p)==h for p,h in protected.items()))
    write(args.output/'report.json',result)
    if not result['weights_unchanged'] or not result['source_unchanged']:raise RuntimeError('Read-only layer benchmark changed weights')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
