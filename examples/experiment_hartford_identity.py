"""Isolated identity-input discrimination probe and bounded pipeline acceptance.

This does not continue the production run or measure improvement in game strength.
The probe labels implement an explicit immediate-mana/life objective, not optimal
Magic play. All new checkpoints and trajectories belong to this experiment.
"""
import argparse
import asyncio
from dataclasses import dataclass,replace
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/compact-ppo-mpl')

import numpy as np
from patchwork_automaton.core import Handle,Response
from patchwork_automaton.backends.rustmage import OBSERVATION_SCHEMA
from patchwork_automaton.neural.rustmage_observation import _Object,_Player,PLAYERS
from patchwork_automaton.neural.compact_identity import IdentityEncoder,pack


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class ProbeObservation:
    backend:str
    actor:object
    objects:dict
    players:tuple
    native:dict
    decision:object


def land_probe(fast,shock,lands,life,reverse,permutation):
    # Both cards are in hand. Pick an untapped land, minimizing life paid.
    # The label assumes a subsequent pay-two-life choice for the shockland.
    names=(shock,fast) if reverse else (fast,shock)
    handles=(981+permutation*10,985+permutation*10)
    objects={}
    for i,(name,identifier) in enumerate(zip((*names,*(['Island']*lands)),(*handles,*range(100,100+lands)))):
        handle=Handle('rustmage','object',str(identifier));zone='Hand' if i<2 else 'Battlefield'
        p=dict(id=identifier,name=name,card=name,zone=zone,controller='P0',kinds=2,colors=0,
            keywords=0,power=None,toughness=None,counters={},tapped=False)
        objects[handle]=_Object(handle,name,zone.lower(),PLAYERS['P0'],p)
    if permutation%2:objects=dict(reversed(list(objects.items())))
    players=tuple(_Player(PLAYERS[f'P{i}'],f'P{i}',life,2,40,
        dict(life=life,hand_count=2,library_count=40,mana=[0]*6)) for i in range(2))
    raw=dict(schema=OBSERVATION_SCHEMA,decision='priority',step='Main1',active='P0',turn=lands+1,
        stack=[],waiting_triggers=[],objects=[o.properties for o in objects.values()])
    obs=ProbeObservation('rustmage',PLAYERS['P0'],objects,players,raw,SimpleNamespace(kind='priority',token=None))
    choices=tuple((Response(str(i)),dict(kind='PlayLand',card=h)) for i,h in enumerate(handles))
    record=IdentityEncoder().encode(obs,choices=choices).arrays['_compact']
    label=names.index(fast if lands<=2 else shock)
    return replace(record,action=label)


def probe(source,output,steps):
    import keras
    import tensorflow as tf
    from patchwork_automaton.neural.compact_identity_model import transfer_v1,TensorStore
    from patchwork_automaton.neural.compact_standard import pack as old_pack
    from patchwork_automaton.neural.compact_standard_model import TensorStore as OldStore
    pairs=[('Blooming Marsh','Overgrown Tomb'),('Concealed Courtyard','Godless Shrine')]
    def dataset(held):
        return [land_probe(a,b,n,life,reverse,100+i if held else i)
            for i,(a,b,n,life,reverse) in enumerate((a,b,n,life,reverse)
                for a,b in pairs for n in range(6) for life in range(7 if held else 8,21,2)
                for reverse in (False,True))]
    train,test=dataset(False),dataset(True)
    training=TensorStore(pack(train));validation=TensorStore(pack(test))
    tb=training.batch(tf.range(len(train)));vb=validation.batch(tf.range(len(test)))
    labels=tf.constant([r.action for r in train]);truth=np.asarray([r.action for r in test])
    old_b=OldStore(old_pack(test)).batch(tf.range(len(test)))
    initial=source(old_b)
    report=dict(task='Choose untapped mana with minimum life payment among two specified lands',
        limitation='Synthetic representation probe with hand-authored local labels; not game-strength evidence.',
        training_examples=len(train),heldout_examples=len(test),steps=steps,
        heldout='Odd life totals and fresh object IDs/order; both action orders balanced in both splits.',
        unchanged_200k_accuracy=float(np.mean(np.asarray(initial['logits']).argmax(1)==truth)),arms={})
    for name,identity,rules in [('capacity_control',False,False),('identity_only',True,False),('identity_plus_rules_ablation',True,True)]:
        keras.utils.set_random_seed(413)
        model=transfer_v1(source,use_identity=identity,use_rules=rules)
        # Keep the completed policy fixed during this representation probe.
        variables=model.actor.residual.trainable_variables
        opt=keras.optimizers.Adam(.003);opt.build(variables)
        before=model(vb)
        for k in ('logits','value'):np.testing.assert_array_equal(before[k],initial[k])
        @tf.function(autograph=False)
        def step():
            with tf.GradientTape() as tape:
                logits=model.dense_logits(model.actor(tb),tb)
                loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))
            grads=tape.gradient(loss,variables)
            opt.apply_gradients((g,v) for g,v in zip(grads,variables) if g is not None)
            return loss
        start=time.monotonic()
        for _ in range(steps):loss=step()
        logits=np.asarray(model(vb)['logits']);prediction=logits.argmax(1)
        report['arms'][name]=dict(accuracy=float(np.mean(prediction==truth)),training_loss=float(loss),
            seconds=time.monotonic()-start,parameters=int(model.count_params()),
            per_land_pair={a:float(np.mean(prediction[j*len(test)//2:(j+1)*len(test)//2]==truth[j*len(test)//2:(j+1)*len(test)//2]))
                for j,(a,b) in enumerate(pairs)})
        write(output/'probe.json',report)
    return report


async def acceptance(source,output):
    import tensorflow as tf
    from examples import train_hartford_ppo as driver
    from patchwork_automaton.baselines.standard_decks import STANDARD_DECKS
    from patchwork_automaton.neural.compact_identity_model import transfer_v1
    from patchwork_automaton.neural.batching import InferenceBatcher
    from patchwork_automaton.neural.experiment import make_trainer,save_checkpoint,load_checkpoint
    from patchwork_automaton.neural.standard_policy import StandardPolicy
    from patchwork_automaton.neural.validation import GameValidator,training_normalization
    from patchwork_automaton.rl.standard_direct import play
    decks=tuple(STANDARD_DECKS);model=transfer_v1(source)
    config=driver.configuration(driver.parser().parse_args([]))
    config.update(architecture='compact-identity-standard-v2',concurrent_updates=False,
        experiment='identity-input-pipeline-acceptance',initialization='transfer_from_compact_200k')
    trainer=make_trainer(model,config);trainer.compile()
    config_path=output/'acceptance-config.json';write(config_path,config)
    schedule=[dict(index=i,seed=713020000+i,decks=[decks[(i//3)%3],decks[i%3]],
        starting_player=f'P{i%2}',learner_seat=i%2,partition='holdout' if i%3==2 else 'training') for i in range(12)]
    report=dict(purpose='Bounded pipeline acceptance; not a strength estimate or production continuation',
        schedule=schedule,arms={})
    loop=asyncio.get_running_loop();running=True
    def tick():
        if running:loop.call_later(.01,tick)
    loop.call_later(.01,tick)
    try:
        for name,current in [('unchanged_200k',source),('identity_transfer',model)]:
            service=InferenceBatcher(current,max_batch=16,wait_ms=0,device='cpu');await service.start()
            rows=[];begin=time.monotonic()
            try:
                for game in schedule:
                    row=await asyncio.to_thread(play,game,output=output/name,model=current,batcher=service,
                        learner_seat=game['learner_seat'],record_ppo=True,game_timeout=20,max_turns=50,max_decisions=6000)
                    rows.append(row)
                    write(output/f'{name}-games.json',rows)
                    if row['status'] not in ('win','loss','draw'):raise RuntimeError(row.get('error',row))
            finally:await service.close()
            report['arms'][name]=dict(games=len(rows),seconds=time.monotonic()-begin,
                game_seconds=sum(r['seconds'] for r in rows),outcomes=rows)
            write(output/'acceptance.json',report)
        training=[r for r in rows if r['partition']=='training'];held=[r for r in rows if r['partition']=='holdout']
        report['ppo']=trainer.update_files([t for r in training for t in r['trajectories']])
        before=driver.training_state(model,trainer)
        report['holdout']=GameValidator(model,batch_size=config['train_batch']).evaluate(held,
            training_normalization(training,config['advantage_scale_floor']))
        assert driver.training_state(model,trainer)==before
        checkpoint=save_checkpoint(output/'checkpoints',model,trainer,dict(round=1,config=config),policy_type=StandardPolicy)
        restored,restored_trainer,_=load_checkpoint(output/'checkpoints',policy_type=StandardPolicy)
        assert driver.training_state(restored,restored_trainer)==before
        report.update(checkpoint=str(checkpoint),optimizer_rng_resume_verified=True,holdout_read_only=True)
        write(output/'acceptance.json',report)
        return report
    finally:running=False


async def with_executor_shutdown(coroutine):
    """Keep bounded wakeups through executor shutdown in restricted sandboxes."""
    loop=asyncio.get_running_loop();running=True
    def tick():
        if running:loop.call_later(.01,tick)
    loop.call_later(.01,tick)
    try:
        result=await coroutine
        await loop.shutdown_default_executor(timeout=10)
        return result
    finally:running=False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=300);p.add_argument('--mode',choices=('probe','acceptance','both'),default='both')
    args=p.parse_args()
    if args.steps<1:raise ValueError('Positive steps required')
    args.output.mkdir(parents=True,exist_ok=False)
    from patchwork_automaton.neural.standard_policy import StandardPolicy
    digest=sha(args.source);source=StandardPolicy.load(args.source).model
    write(args.output/'provenance.json',dict(source=str(args.source.resolve()),source_sha256=digest,mode=args.mode,
        default_rules_enabled=False,production_run_modified=False))
    if args.mode in ('probe','both'):probe(source,args.output,args.steps)
    if args.mode in ('acceptance','both'):asyncio.run(with_executor_shutdown(acceptance(source,args.output)))
    if sha(args.source)!=digest:raise RuntimeError('Source checkpoint changed')
    print(json.dumps(dict(output=str(args.output),source_unchanged=True)),flush=True)


if __name__=='__main__':main()
