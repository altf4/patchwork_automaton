"""Isolated three-deck compact-input imitation and matched timing experiment."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes as C
from datetime import datetime, timezone
import gzip
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

os.environ.setdefault('KERAS_BACKEND','tensorflow')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

import numpy as np
from patchwork_automaton.baselines.standard_decks import STANDARD_DECKS, standard_deck
from patchwork_automaton.neural.compact_standard import encode, pack, restore, VERSION, SPEC

DECKS = tuple(STANDARD_DECKS)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temp.replace(path)


def emit(phase, **values):
    print(json.dumps(dict(phase=phase,**values)),flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory():
    mem = {k:int(v.split()[0])*1024 for k,v in
        (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        if k in ('MemAvailable','SwapFree')}
    mem['rss_bytes'] = int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines()
        if x.startswith('VmRSS:')))*1024
    return mem


class Game:
    def __init__(self, row):
        from patchwork_automaton.neural.direct_standard import library, check
        from patchwork_automaton.baselines.standard import StandardAggro
        self.lib, self.check, self.row = library(), check, row
        cards = [[c.name for c in standard_deck(d).cards for _ in range(c.quantity)] for d in row['decks']]
        self.handle = check(self.lib,self.lib.ds_new(json.dumps(dict(decks=cards,seed=row['seed'],
            starting_player=row['starting_player'],deck_rules='Constructed')).encode()))
        self.scripts = [StandardAggro(d) for d in row['decks']]
        self.started = time.monotonic(); self.decisions = 0; self.pending = None

    def observe(self):
        from patchwork_automaton.neural.direct_standard import Metadata
        view = self.check(self.lib,self.lib.ds_observe(self.handle,0))
        meta = Metadata(); self.check(self.lib,self.lib.ds_metadata(view,C.byref(meta)))
        if meta.outcome != -1 or meta.turn > 50 or self.decisions >= 6000 or time.monotonic()-self.started > 120:
            self.lib.ds_view_free(view)
            return None,meta
        if meta.actor != 0:
            self.lib.ds_view_free(view)
            view = self.check(self.lib,self.lib.ds_observe(self.handle,meta.actor))
        return view,meta

    def script(self, view, actor):
        from patchwork_automaton.neural.direct_standard import string
        from patchwork_automaton.neural.rustmage_observation import learning_observation
        from patchwork_automaton.backends.rustmage import domain_action, _plain
        from patchwork_automaton.core import Handle
        raw = json.loads(string(self.lib,self.lib.ds_debug_json(view)))
        fake = SimpleNamespace(session='compact-experiment',handle=Handle('rustmage','game',str(self.row['index'])),_raw=lambda _:raw)
        obs = learning_observation(fake,f'P{actor}')
        response = self.scripts[actor].choose(obs)
        option = next(o for o in obs.decision.options if o.id==response.option)
        return _plain(option.constraints['action'] if option.kind=='finite' else domain_action(option,response.value))

    def size(self, view):
        size = C.c_size_t(); self.check(self.lib,self.lib.ds_prepare(view,False,C.byref(size)))
        return size.value

    def record(self, view, *, action=0, actor=0):
        from patchwork_automaton.neural.direct_standard import arrays
        return encode(arrays(self.lib,self.lib.ds_encode(view)),action=action,deck=DECKS.index(self.row['decks'][actor]))

    def apply(self, view, *, index=None, action=None):
        try:
            if index is None:
                self.check(self.lib,self.lib.ds_apply_script(self.handle,view,json.dumps(action,separators=(',',':')).encode()))
            else:
                self.check(self.lib,self.lib.ds_apply_choice(self.handle,view,int(index)))
            self.decisions += 1
        finally:
            self.lib.ds_view_free(view)

    def finish(self, meta):
        winner = None if meta.outcome in (-1,2) else meta.outcome
        return dict(self.row,decisions=self.decisions,turn=int(meta.turn),winner=winner,
            adjudicated=meta.outcome==-1,seconds=time.monotonic()-self.started,
            reason='decision_limit' if self.decisions>=6000 else 'turn_limit' if meta.turn>50 else
                'wall_clock_timeout' if meta.outcome==-1 else 'terminal')

    def close(self):
        if self.pending is not None:self.lib.ds_view_free(self.pending);self.pending=None
        if self.handle:self.lib.ds_free(self.handle);self.handle=None


def collect_game(task):
    row, folder = task
    path = Path(folder)/f"game-{row['index']:05d}.pkl.gz"
    game = Game(row); records = []
    try:
        from patchwork_automaton.neural.direct_standard import string
        while True:
            view,meta = game.observe()
            if view is None:break
            action = game.script(view,meta.actor)
            if game.size(view)>1:
                options = [json.loads(string(game.lib,game.lib.ds_choice_debug_json(view,i))) for i in range(game.size(view))]
                if action not in options:raise ValueError('Teacher action outside retained candidate set')
                records.append(game.record(view,action=options.index(action),actor=meta.actor))
            game.apply(view,action=action)
        result = game.finish(meta)
        with gzip.open(path,'wb',compresslevel=1) as f:pickle.dump(records,f,protocol=5)
        result.update(path=str(path),sha256=sha(path),samples=len(records),
            compact_bytes=sum(r.nbytes for r in records),dense_bytes=sum(r.dense_bytes for r in records),
            stored_bytes=path.stat().st_size)
        return result
    finally:game.close()


def schedule(games, seed):
    if games%180:raise ValueError('Game count must be a positive multiple of 180 for balanced whole-game splits')
    out=[]
    for i in range(games):
        pair=i%9;stratum=(i//9)%4;cycle=i//36
        out.append(dict(index=i,seed=seed+i,decks=[DECKS[pair//3],DECKS[pair%3]],
            starting_player=f'P{stratum%2}',partition='validation' if cycle%5==4 else 'training'))
    return out


def collect(args):
    folder=args.output/'games';folder.mkdir(parents=True,exist_ok=True)
    path=args.output/'dataset.json'
    expected=schedule(args.games,args.seed)
    if path.exists():
        data=json.loads(path.read_text())
        if data['schedule']!=expected or data['representation']!=VERSION:raise ValueError('Dataset resume mismatch')
        return data
    started=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        tasks=[pool.submit(collect_game,(row,str(folder))) for row in expected]
        for future in as_completed(tasks):
            rows.append(future.result())
            if len(rows)%36==0:emit('collection',games=len(rows),total=args.games,seconds=time.monotonic()-started)
    rows.sort(key=lambda r:r['index'])
    data=dict(representation=VERSION,schedule=expected,games=rows,seconds=time.monotonic()-started,
        samples=sum(r['samples'] for r in rows),compact_bytes=sum(r['compact_bytes'] for r in rows),
        dense_bytes=sum(r['dense_bytes'] for r in rows),stored_bytes=sum(r['stored_bytes'] for r in rows),
        adjudications=sum(r['adjudicated'] for r in rows))
    write(path,data);return data


def load_records(data):
    train=[];validation=[]
    for row in data['games']:
        if sha(row['path'])!=row['sha256']:raise ValueError('Demonstration changed')
        with gzip.open(row['path'],'rb') as f:records=pickle.load(f)
        if len(records)!=row['samples']:raise ValueError('Sample count changed')
        (train if row['partition']=='training' else validation).extend(records)
    return train,validation


def chunks(order,size):
    for i in range(0,len(order),size):yield order[i:i+size]


def metric_summary(loss,correct,decks,kinds):
    def one(ids):
        return dict(samples=int(ids.sum()),cross_entropy=float(loss[ids].mean()),accuracy=float(correct[ids].mean()))
    return dict(overall=one(np.ones(len(loss),bool)),
        decks={name:one(decks==i) for i,name in enumerate(DECKS) if np.any(decks==i)},
        action_kinds={str(i):one(kinds==i) for i in np.unique(kinds)},
        macro_action_kind_accuracy=float(np.mean([correct[kinds==i].mean() for i in np.unique(kinds)])))


def setup_tf():
    from patchwork_automaton.neural.devices import configure_runtime
    configure_runtime('gpu')
    import tensorflow as tf
    import keras
    return tf,keras


def train_compact(args, train, validation, report):
    tf,keras=setup_tf()
    from patchwork_automaton.neural.compact_standard_model import TensorStore,CompactStandardPolicy,objective
    keras.utils.set_random_seed(args.seed % 2**32)
    training_count,validation_count=len(train),len(validation)
    with tf.device('/GPU:0'):
        preparation=time.monotonic()
        training_arrays=pack(train);validation_arrays=pack(validation)
        decks=validation_arrays['deck'].copy();kinds=validation_arrays['kind'].copy()
        store=TensorStore(training_arrays);held=TensorStore(validation_arrays)
        report['compact_store']=dict(train_bytes=store.bytes,validation_bytes=held.bytes,
            packing_and_upload_seconds=time.monotonic()-preparation,packing='interned-sparse-rows-v2',
            unique_training_nodes=len(training_arrays['nodes_splits'])-1,
            unique_training_local_actions=len(training_arrays['local_splits'])-1)
        train.clear();validation.clear();del training_arrays,validation_arrays;gc.collect()
        from patchwork_automaton.neural.host_memory import HostMemoryReclaimer
        HostMemoryReclaimer().release('compact_host_inputs_uploaded')
        model=CompactStandardPolicy(args.width)
        model(store.batch(tf.constant([0],tf.int32)))
        optimizer=keras.optimizers.Adam(args.learning_rate)
        optimizer.build(model.trainable_variables)
    from patchwork_automaton.neural.devices import assert_model_device
    assert_model_device(model,'gpu')
    report['compact_parameters']=int(model.count_params())
    def evaluator(dataset):
        @tf.function(input_signature=[tf.TensorSpec([None],tf.int32)])
        def evaluate(ids):
            b=dataset.batch(ids);loss,correct,_=objective(model(b),b)
            return loss,correct
        return evaluate
    held_eval=evaluator(held)
    def evaluate():
        losses=[];correct=[]
        for ids in chunks(np.arange(validation_count,dtype=np.int32),args.batch):
            a,b=held_eval(ids);losses.extend(a.numpy());correct.extend(b.numpy())
        return metric_summary(np.asarray(losses),np.asarray(correct),decks,kinds)
    @tf.function(input_signature=[tf.TensorSpec([None],tf.int32)])
    def step(ids):
        b=store.batch(ids)
        with tf.GradientTape() as tape:
            loss,_,_=objective(model(b),b);value=tf.reduce_mean(loss)
        grads=tape.gradient(value,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        return value
    report['compact_initial']=evaluate()
    model.save_weights(args.output/'compact-initial.weights.h5')
    emit('compact_initial',metrics=report['compact_initial'],parameters=model.count_params(),memory=memory())
    history=[];rng=np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        order=rng.permutation(training_count).astype(np.int32)
        started=time.monotonic();losses=[];step_times=[]
        for ids in chunks(order,args.batch):
            before=time.monotonic();losses.append(float(step(ids)));step_times.append(time.monotonic()-before)
        seconds=time.monotonic()-started
        row=dict(epoch=epoch+1,seconds=seconds,decisions_per_second=training_count/seconds,
            training_cross_entropy=float(np.mean(losses)),
            steady_step_median_seconds=float(np.median(step_times[3:])),
            memory=memory(),validation=evaluate())
        history.append(row);report['compact_epochs']=history;write(args.output/'report.json',report)
        emit('compact_epoch',**row)
    model.save_weights(args.output/'compact-final.weights.h5')
    write(args.output/'compact-model.json',dict(representation=VERSION,width=args.width,seed=args.seed,
        parameters=model.count_params(),history=False,card_identity_embeddings=False,feature_spec=SPEC))
    return model


def dense_control(args,train,validation,report):
    """Same labels, shuffle, optimizer and batch size; legacy stateless actor."""
    tf,keras=setup_tf()
    from patchwork_automaton.neural.model import SeparateActorCritic
    from patchwork_automaton.neural.features import collate
    from patchwork_automaton.neural.batching import example_batch
    from patchwork_automaton.neural.recurrent import configure_gru_backend
    from patchwork_automaton.neural.host_memory import HostMemoryReclaimer
    keras.utils.set_random_seed(args.seed % 2**32)
    with tf.device('/GPU:0'):
        container=SeparateActorCritic(1,SPEC)
        container(example_batch(container.memory_size,SPEC),training=False)
        configure_gru_backend(container,'cudnn-memory')
        model=container.actor
        optimizer=keras.optimizers.Adam(args.learning_rate);optimizer.build(model.trainable_variables)
    def batch(ids,records):
        arrays=collate([restore(records[int(i)]) for i in ids])
        arrays['memory_in']=np.zeros((len(ids),model.memory_size),np.float32)
        arrays['advance_memory']=np.ones((len(ids),1),np.float32)
        labels=np.asarray([records[int(i)].action for i in ids],np.int32)
        return arrays,labels
    @tf.function(reduce_retracing=True)
    def step(arrays,labels):
        with tf.GradientTape() as tape:
            logits=model(arrays,training=True)['logits']
            loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))
        grads=tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables));return loss
    @tf.function(reduce_retracing=True)
    def infer(arrays,labels):
        logits=model(arrays,training=False)['logits']
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits),tf.cast(tf.argmax(logits,axis=-1,output_type=tf.int32)==labels,tf.float32)
    def evaluate():
        losses=[];correct=[]
        for ids in chunks(np.arange(len(validation),dtype=np.int32),args.batch):
            a,b=batch(ids,validation);loss,hit=infer(a,b);losses.extend(loss.numpy());correct.extend(hit.numpy())
            del a,b
        return metric_summary(np.asarray(losses),np.asarray(correct),np.asarray([r.deck for r in validation]),np.asarray([r.kind for r in validation]))
    report['dense_initial']=evaluate();history=[];rng=np.random.default_rng(args.seed)
    emit('dense_initial',metrics=report['dense_initial'],memory=memory())
    for epoch in range(args.control_epochs):
        order=rng.permutation(len(train)).astype(np.int32);started=time.monotonic();prep=compute=0.;losses=[]
        for j,ids in enumerate(chunks(order,args.batch)):
            before=time.monotonic();a,b=batch(ids,train);prep+=time.monotonic()-before
            before=time.monotonic();losses.append(float(step(a,b)));compute+=time.monotonic()-before
            del a,b
            if j%50==0:emit('dense_progress',epoch=epoch+1,batch=j,seconds=time.monotonic()-started,memory=memory())
        seconds=time.monotonic()-started
        row=dict(epoch=epoch+1,seconds=seconds,decisions_per_second=len(train)/seconds,
            preparation_seconds=prep,optimizer_seconds=compute,training_cross_entropy=float(np.mean(losses)),
            memory=memory(),validation=evaluate())
        history.append(row);report['dense_epochs']=history;write(args.output/'report.json',report)
        emit('dense_epoch',**row)
        HostMemoryReclaimer().release('dense_epoch')
    model.save_weights(args.output/'dense-control-final.weights.h5')
    return model


def evaluate_games(args,model,report,stage):
    import tensorflow as tf
    from patchwork_automaton.neural.compact_standard_model import TensorStore,objective
    @tf.function(reduce_retracing=True)
    def infer(batch):
        logits=model(batch);_,_,pred=objective(logits,batch)
        starts=tf.math.unsorted_segment_min(tf.range(tf.size(logits)),batch['action_owner'],batch['n_examples'])
        return pred-starts
    schedule_rows=[]
    for learner in range(3):
        for opponent in range(3):
            for i in range(args.eval_games):
                seat=i%2;decks=[DECKS[opponent]]*2;decks[seat]=DECKS[learner]
                schedule_rows.append(dict(index=len(schedule_rows),seed=args.eval_seed+len(schedule_rows),
                    decks=decks,starting_player=f'P{(i//2)%2}',learner_seat=seat,
                    learner_deck=DECKS[learner],opponent_deck=DECKS[opponent]))
    started=time.monotonic();pending=list(schedule_rows);active=[];results=[];neural=0;encode_seconds=infer_seconds=0.
    try:
        while pending or active:
            while pending and len(active)<args.eval_parallel:active.append(Game(pending.pop(0)))
            ready=[];records=[]
            for game in list(active):
                while True:
                    view,meta=game.observe()
                    if view is None:
                        result=game.finish(meta);result['learner_won']=result['winner']==game.row['learner_seat']
                        results.append(result);game.close();active.remove(game);break
                    if meta.actor!=game.row['learner_seat']:
                        game.apply(view,action=game.script(view,meta.actor));continue
                    if game.size(view)==1:game.apply(view,index=0);continue
                    before=time.monotonic();records.append(game.record(view,actor=meta.actor));encode_seconds+=time.monotonic()-before
                    game.pending=view;ready.append(game);break
            if ready:
                before=time.monotonic()
                with tf.device('/GPU:0'):
                    store=TensorStore(pack(records));batch=store.batch(tf.range(len(records),dtype=tf.int32))
                    selected=infer(batch).numpy()
                infer_seconds+=time.monotonic()-before;neural+=len(ready)
                for game,index in zip(ready,selected):
                    view=game.pending;game.pending=None;game.apply(view,index=index)
            if len(results) and len(results)%36==0:
                emit('game_evaluation',stage=stage,completed=len(results),total=len(schedule_rows),seconds=time.monotonic()-started)
        def summarize(rows):
            wins=sum(r['learner_won'] for r in rows);draws=sum(r['winner'] is None for r in rows)
            return dict(games=len(rows),wins=wins,draws=draws,score_rate=(wins+.5*draws)/len(rows),
                adjudications=sum(r['adjudicated'] for r in rows))
        summary=dict(overall=summarize(results),decks={d:summarize([r for r in results if r['learner_deck']==d]) for d in DECKS},
            matchups={a+' vs '+b:summarize([r for r in results if r['learner_deck']==a and r['opponent_deck']==b]) for a in DECKS for b in DECKS},
            seconds=time.monotonic()-started,neural_decisions=neural,encode_seconds=encode_seconds,inference_seconds=infer_seconds)
        write(args.output/f'evaluation-{stage}.json',dict(summary=summary,games=results))
        report.setdefault('game_evaluations',{})[stage]=summary;write(args.output/'report.json',report)
        emit('evaluation_complete',stage=stage,**summary)
    finally:
        for game in active:game.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('collect','compact','dense','evaluate'),required=True)
    p.add_argument('--games',type=int,default=900)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--control-epochs',type=int,default=2)
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--seed',type=int,default=1826110100000)
    p.add_argument('--eval-seed',type=int,default=2826110100000)
    p.add_argument('--eval-games',type=int,default=24)
    p.add_argument('--eval-parallel',type=int,default=24)
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    if args.eval_games%4 or min(args.games,args.epochs,args.batch,args.width)<1:raise ValueError('Invalid balanced experiment settings')
    start=time.monotonic();report_path=args.output/'report.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else dict(status='running',
        created_at=datetime.now(timezone.utc).isoformat(),config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='phase'},
        representation=VERSION,limitations=['Stateless policy: recurrent memory, event history and previous action are excluded.',
            'Imitation training is not a full PPO update; matched dense imitation is the representation control.',
            'Native per-decision encoding is still dense before compact conversion; training data and minibatches stay sparse.'])
    report['status']='running';report.pop('error',None)
    try:
        if args.phase=='collect':
            data=collect(args);report['dataset']={k:v for k,v in data.items() if k not in ('games','schedule')}
            report['dataset']['sha256']=sha(args.output/'dataset.json');write(report_path,report);return
        data=json.loads((args.output/'dataset.json').read_text());train,validation=load_records(data)
        report['partitions']=dict(training_games=sum(r['partition']=='training' for r in data['games']),
            validation_games=sum(r['partition']=='validation' for r in data['games']),training_decisions=len(train),validation_decisions=len(validation))
        report['random_reference']=dict(accuracy=float(np.mean([1/len(r.roles) for r in validation])),
            cross_entropy=float(np.mean([np.log(len(r.roles)) for r in validation])))
        if args.phase=='compact':train_compact(args,train,validation,report)
        elif args.phase=='dense':dense_control(args,train,validation,report)
        else:
            tf,_=setup_tf()
            from patchwork_automaton.neural.compact_standard_model import TensorStore,CompactStandardPolicy
            with tf.device('/GPU:0'):
                model=CompactStandardPolicy(args.width);model(TensorStore(pack(train[:1])).batch(tf.constant([0],tf.int32)))
            train.clear();validation.clear();gc.collect()
            from patchwork_automaton.neural.host_memory import HostMemoryReclaimer
            HostMemoryReclaimer().release('before_live_evaluation')
            for stage in ('initial','final'):
                model.load_weights(args.output/f'compact-{stage}.weights.h5');evaluate_games(args,model,report,stage)
        report.setdefault('phase_seconds',{})[args.phase]=time.monotonic()-start
        if all(k in report for k in ('compact_epochs','dense_epochs','game_evaluations')):report['status']='complete'
        write(report_path,report)
    except BaseException as e:
        report['status']='failed';report['error']=repr(e);write(report_path,report);raise


if __name__=='__main__':main()
