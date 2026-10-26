"""Replay a frozen full PPO update with an opt-in GRU execution backend.

Run benchmark_fused_gru.py first for output/gradient equivalence. This driver
restores the exact model, both Adam states and PCG from the preceding checkpoint;
only the runtime GRU backend changes. No games or source checkpoint writes.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from examples.profile_ppo_update import (Probe, instrument, checkpoint_view,
    select_training, sha, write)


def run(args):
    for key,value in {'KERAS_BACKEND':'tensorflow','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'4','TF_ENABLE_ONEDNN_OPTS':'0','TF_CPP_MIN_LOG_LEVEL':'2','CUDA_VISIBLE_DEVICES':'0' if args.device=='gpu' else '-1'}.items():os.environ.setdefault(key,value)
    from patchwork_automaton.neural.devices import configure_runtime,device_scope,assert_model_device
    configure_runtime(args.device,hide_gpu=args.device=='cpu')
    import keras
    import numpy as np
    from examples.train_rustmage import place_trainer,neural_call,array_digest
    from patchwork_automaton.neural.experiment import load_checkpoint
    source=args.source.resolve();output=args.output.resolve()
    if source==output or source in output.parents or output in source.parents:raise ValueError('Output must be outside the source study')
    output.mkdir(parents=True,exist_ok=False)
    report_path=source/'training/report.json';saved=json.loads(report_path.read_text());row,trajectories=select_training(saved,args.round)
    checkpoint=source/f'training/checkpoints/round-{args.round-1:04d}'
    sources={str(report_path):sha(report_path),**{str(p):sha(p) for p in checkpoint.iterdir() if p.is_file()}}
    inputs={t['path']:sha(t['path']) for t in trajectories}
    keras.utils.set_random_seed(saved['config']['seed'])
    with device_scope(args.device),checkpoint_view(checkpoint) as view:
        model,trainer,state=load_checkpoint(view)
        from patchwork_automaton.neural.recurrent import configure_gru_backend
        from patchwork_automaton.neural.experiment import make_trainer
        import copy
        backend_layers=configure_gru_backend(model,args.gru_backend)
        original=trainer
        trainer=make_trainer(model,state['config']);trainer.compile()
        for left,right in [(original.optimizer.variables,trainer.optimizer.variables),(original.critic_optimizer.variables,trainer.critic_optimizer.variables)]:
            assert len(left)==len(right)
            for a,b in zip(left,right):b.assign(a)
        trainer.rng.bit_generator.state=copy.deepcopy(original.rng.bit_generator.state)
        from examples.benchmark_fused_gru import graph_ops
        kernel_ops={name:graph_ops(getattr(trainer,name)) for name in ('step','critic_step','measure_actor_with_drift')}
        if args.gru_backend.startswith('cudnn') and not all(any('CudnnRNN' in k for k in d) for d in kernel_ops.values()):raise RuntimeError('cuDNN graph path missing')
        placements=assert_model_device(model,args.device);traces=place_trainer(trainer,args.device)
    assert state['round']==args.round-1 and state['config']==saved['config']
    probe=Probe(trace_dir=output/'tf-trace' if args.instrumented and args.trace_seconds else None,trace_seconds=args.trace_seconds)
    if args.instrumented:(output/'instrumented-method.py.txt').write_text(instrument(trainer,probe))
    initial_rng=trainer.rng.bit_generator.state
    info={'status':'running','mode':args.gru_backend,'gru_layers':backend_layers,'kernel_ops':kernel_ops,'device':args.device,'source':str(source),'source_round':args.round,
          'source_hashes':sources,'input_hashes':inputs,'training_trajectories':len(trajectories),'training_games':len(row['split']['training_indices']),
          'heldout_games_excluded':len(row['split']['validation_indices']),'config':state['config'],'variable_placements':placements,'graph_traces':traces,
          'initial_rng':initial_rng,'initial_model_sha256':array_digest(model.weights),'initial_actor_optimizer_steps':int(trainer.optimizer.iterations),
          'initial_critic_optimizer_steps':int(trainer.critic_optimizer.iterations),'runtime':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','TF_ENABLE_ONEDNN_OPTS']}}
    write(output/'report.json',info)
    metrics_file=(output/'nvidia-smi.csv').open('w');metrics_error=(output/'nvidia-smi.stderr.log').open('w')
    try:
        monitor=subprocess.Popen(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem','--format=csv,noheader,nounits','--loop-ms=500'],stdout=metrics_file,stderr=metrics_error)
    except FileNotFoundError:monitor=None
    started=time.perf_counter();info['update_started_wall_ns']=time.time_ns();cpu=time.process_time()
    try:
        with probe.span('full_update'):
            result=neural_call(args.device,trainer.update_files,trajectories)
        elapsed=time.perf_counter()-started;cpu_elapsed=time.process_time()-cpu
        final_arrays={}
        for label,values in [('model',model.weights),('actor_optimizer',trainer.optimizer.variables),('critic_optimizer',trainer.critic_optimizer.variables)]:
            final_arrays.update({f'{label}_{i}':np.asarray(v) for i,v in enumerate(values)})
        np.savez(output/'result-state.npz',**final_arrays)
        info.update(status='complete',update_seconds=elapsed,process_cpu_seconds=cpu_elapsed,result=result,final_rng=trainer.rng.bit_generator.state,
                    final_model_sha256=array_digest(model.weights),final_actor_optimizer_steps=int(trainer.optimizer.iterations),final_critic_optimizer_steps=int(trainer.critic_optimizer.iterations))
    except BaseException as e:
        info.update(status='error',error=repr(e));raise
    finally:
        if monitor:
            monitor.terminate()
            try:monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:monitor.kill();monitor.wait()
        metrics_file.close();metrics_error.close()
        info['profiling']=probe.finish(output)
        info['source_unchanged']=all(sha(p)==h for p,h in sources.items()) and all(sha(p)==h for p,h in inputs.items())
        info['finished_utc']=datetime.now(timezone.utc).isoformat();write(output/'report.json',info)
    if not info['source_unchanged']:raise ValueError('Read-only source changed during profiling')
    print(json.dumps({k:info[k] for k in ['status','mode','update_seconds','process_cpu_seconds','final_model_sha256','source_unchanged']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gru-backend',choices=['unfused','cudnn','cudnn-memory'],required=True);p.add_argument('--round',type=int,default=40);p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--instrumented',action='store_true');p.add_argument('--trace-seconds',type=float,default=30)
    run(p.parse_args())
