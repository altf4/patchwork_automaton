"""Replay a frozen PPO update with timing-only probes; never generate games.

The production source is not edited. An isolated method copy gains timing
contexts around preparation and queue waits. Existing Tensor -> float/NumPy
synchronizations are observed at the same caller boundaries. No batch size,
optimizer, guard, RNG, model, data order, or storage setting is overridden.
"""
import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import functools
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import threading
import time
import types


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def interval_union(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if end<start:raise ValueError('Reversed interval')
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return sum(b-a for a,b in merged)


class Probe:
    def __init__(self,*,trace_dir=None,trace_after_steps=20,trace_seconds=30):
        self.events=[];self.event_lock=threading.Lock();self.trace_lock=threading.Lock()
        self.t0=time.perf_counter_ns();self.wall_t0=time.time_ns()
        self.trace_dir=Path(trace_dir) if trace_dir else None
        self.trace_after_steps=trace_after_steps;self.trace_seconds=trace_seconds
        self.actor_steps=0;self.trace_active=threading.Event();self.trace_timer=None
        self.trace_started=False;self.trace_info={};self.proposals=[];self.tf=None

    @contextmanager
    def span(self,name):
        trace=None
        if self.trace_active.is_set():
            trace=self.tf.profiler.experimental.Trace(name);trace.__enter__()
        wall=time.perf_counter_ns();cpu=time.thread_time_ns()
        try:yield
        finally:
            end=time.perf_counter_ns();cpu_end=time.thread_time_ns()
            if trace:trace.__exit__(None,None,None)
            event=(name,threading.get_ident(),threading.current_thread().name,wall,end,cpu_end-cpu)
            with self.event_lock:self.events.append(event)

    def timed(self,name):
        def decorate(fn):
            @functools.wraps(fn)
            def wrapped(*a,**kw):
                with self.span(name):return fn(*a,**kw)
            return wrapped
        return decorate

    @contextmanager
    def locked(self,lock):
        with self.span('packing_lock_wait'):lock.acquire()
        try:
            with self.span('packing_lock_held'):yield
        finally:lock.release()

    def iterate(self,iterable):
        it=iter(iterable)
        while True:
            name='critic_batch_wait' if threading.current_thread().name.startswith('ppo-critic') else 'actor_batch_wait'
            try:
                with self.span(name):item=next(it)
            except StopIteration:return
            yield item

    def maybe_trace(self):
        self.actor_steps+=1
        if self.trace_dir is None or self.trace_started or self.actor_steps!=self.trace_after_steps:return
        self.trace_started=True
        import tensorflow as tf
        self.tf=tf;self.trace_dir.mkdir(parents=True,exist_ok=True)
        try:
            tf.profiler.experimental.start(str(self.trace_dir),options=tf.profiler.experimental.ProfilerOptions(
                host_tracer_level=2,python_tracer_level=0,device_tracer_level=1))
            self.trace_info.update(start_perf_ns=time.perf_counter_ns(),start_wall_ns=time.time_ns(),requested_seconds=self.trace_seconds)
            self.trace_active.set();self.trace_timer=threading.Timer(self.trace_seconds,self.stop_trace)
            self.trace_timer.daemon=True;self.trace_timer.start()
        except Exception as e:self.trace_info['error']=repr(e)

    def stop_trace(self):
        with self.trace_lock:
            if not self.trace_active.is_set():return
            self.trace_active.clear()
            try:self.tf.profiler.experimental.stop()
            except Exception as e:self.trace_info['stop_error']=repr(e)
            self.trace_info.update(stop_perf_ns=time.perf_counter_ns(),stop_wall_ns=time.time_ns())

    def finish(self,output):
        if self.trace_timer:self.trace_timer.cancel()
        self.stop_trace()
        grouped=defaultdict(list)
        for e in self.events:grouped[e[0]].append(e)
        summary={name:{'calls':len(items),'thread_wall_seconds':sum(e[4]-e[3] for e in items)/1e9,
                       'wall_union_seconds':interval_union((e[3],e[4]) for e in items)/1e9,
                       'thread_cpu_seconds':sum(e[5] for e in items)/1e9}
                 for name,items in grouped.items()}
        with gzip.open(Path(output)/'host-spans.json.gz','wt') as f:
            json.dump({'t0_perf_ns':self.t0,'t0_wall_ns':self.wall_t0,
                       'fields':['name','thread_id','thread_name','start_perf_ns','end_perf_ns','thread_cpu_ns'],
                       'events':self.events},f)
        return {'spans':summary,'trace':self.trace_info,'guard_proposals':self.proposals,
                'accounting':'Spans overlap/nest across actor, critic and readers. Thread sums are not additive wall-time shares.'}


class TimingTransformer(ast.NodeTransformer):
    def with_span(self,label,body,node):
        out=ast.With(items=[ast.withitem(context_expr=ast.Call(
            func=ast.Attribute(ast.Name('_probe',ast.Load()),'span',ast.Load()),
            args=[ast.Constant(label)],keywords=[]),optional_vars=None)],body=body,type_comment=None)
        return ast.copy_location(out,node)

    def visit_FunctionDef(self,node):
        node=self.generic_visit(node)
        if node.name in ('pack','measure','fit_values'):
            label={'pack':'batch_pack','measure':'guard_scan','fit_values':'critic_fit'}[node.name]
            node.decorator_list.append(ast.Call(ast.Attribute(ast.Name('_probe',ast.Load()),'timed',ast.Load()),[ast.Constant(label)],[]))
        if node.name=='_update_files':
            start=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='records_cache' for t in n.targets))
            end=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='packing_lock' for t in n.targets))
            node.body[start:end+1]=[self.with_span('initial_rollout_scan',node.body[start:end+1],node.body[start])]
        return node

    def visit_With(self,node):
        is_batch=any(isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Name) and i.context_expr.func.id=='batches' for i in node.items)
        node=self.generic_visit(node)
        for item in node.items:
            if isinstance(item.context_expr,ast.Name) and item.context_expr.id=='packing_lock':
                item.context_expr=ast.Call(ast.Attribute(ast.Name('_probe',ast.Load()),'locked',ast.Load()),[ast.Name('packing_lock',ast.Load())],[])
        # Both optimizer generators own this context until their pass finishes.
        return self.with_span('optimizer_pass', [node], node) if is_batch else node

    def visit_Assign(self,node):
        node=self.generic_visit(node)
        names={x.id for x in node.targets if isinstance(x,ast.Name)}
        if 'records' in names and isinstance(node.value,ast.ListComp):return self.with_span('record_gather', [node], node)
        if 'inputs' in names and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='collate':return self.with_span('collate', [node], node)
        return node

    def visit_Yield(self,node):
        node=self.generic_visit(node)
        # Only the two iterator yields in batches(); tuple yields from the critic stay untouched.
        if isinstance(node.value,ast.Name) and node.value.id=='prepared' or isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='map':
            node.value=ast.Call(ast.Attribute(ast.Name('_probe',ast.Load()),'iterate',ast.Load()),[node.value],[])
        return node


def instrument(trainer,probe):
    original=trainer._update_files.__func__
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)));tree=TimingTransformer().visit(tree);ast.fix_missing_locations(tree)
    env=dict(original.__globals__,_probe=probe)
    guard=env['_guard_status']
    def observed_guard(*a,**kw):
        result=guard(*a,**kw)
        probe.proposals.append({'accepted':result[0],'reasons':result[2],'mean_kl':float(a[1][0]),'max_state_kl':float(a[1][1])})
        return result
    env['_guard_status']=observed_guard
    exec(compile(tree,inspect.getsourcefile(original),'exec'),env)
    trainer._update_files=types.MethodType(env['_update_files'],trainer)
    for name in ('step','critic_step','measure_with_drift','measure_actor_with_drift','snapshot','restore'):
        fn=getattr(trainer,name,None)
        if fn is None:continue
        def wrapped(*a,_fn=fn,_name=name,**kw):
            if _name=='step':probe.maybe_trace()
            with probe.span(_name+'_call'):value=_fn(*a,**kw)
            if _name in ('step','critic_step'):
                with probe.span(_name+'_result_sync'):return float(value)
            if _name in ('measure_with_drift','measure_actor_with_drift'):
                import numpy as np
                with probe.span(_name+'_result_sync'):return tuple(np.asarray(v) for v in value)
            return value
        setattr(trainer,name,wrapped)
    return ast.unparse(tree)


@contextmanager
def checkpoint_view(path):
    path=Path(path).resolve()
    with tempfile.TemporaryDirectory(prefix='ppo-profile-checkpoint-') as temp:
        root=Path(temp);(root/path.name).symlink_to(path,target_is_directory=True)
        (root/'latest.json').write_text(json.dumps({'checkpoint':path.name}));yield root


def select_training(report,round_id):
    row=next(r for r in report['rounds'] if r['round']==round_id)
    train=set(row['split']['training_indices']);held=set(row['split']['validation_indices'])
    ids={g['index'] for g in row['games']}
    if train&held or train|held!=ids or len(ids)!=len(row['games']):raise ValueError('Invalid whole-game split')
    trajectories=[dict(t) for g in row['games'] if g['index'] in train for t in g.get('trajectories',())]
    if not trajectories:raise ValueError('No training trajectories')
    return row,trajectories


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
        model,trainer,state=load_checkpoint(view);placements=assert_model_device(model,args.device);traces=place_trainer(trainer,args.device)
    assert state['round']==args.round-1 and state['config']==saved['config']
    probe=Probe(trace_dir=output/'tf-trace' if args.instrumented and args.trace_seconds else None,trace_seconds=args.trace_seconds)
    if args.instrumented:(output/'instrumented-method.py.txt').write_text(instrument(trainer,probe))
    initial_rng=trainer.rng.bit_generator.state
    info={'status':'running','mode':'profile' if args.instrumented else 'control','device':args.device,'source':str(source),'source_round':args.round,
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
    p.add_argument('--round',type=int,default=40);p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--instrumented',action='store_true');p.add_argument('--trace-seconds',type=float,default=30)
    run(p.parse_args())
