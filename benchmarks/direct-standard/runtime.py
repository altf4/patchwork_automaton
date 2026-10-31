"""Experimental Standard self-play worker using typed Rust game/tensor handles.

The model and process-based inference service are unchanged. This module is an
isolated benchmark candidate, not a production default. No per-decision game
observation or action is serialized as JSON; progress files are diagnostic.
"""
from collections import Counter
import ctypes as C
from functools import lru_cache
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np

from probe_direct import load, Metadata
from verify_tensors import configure, View
from patchwork_automaton.baselines.standard_decks import standard_deck
from patchwork_automaton.neural.native_pipeline import KEYS
from patchwork_automaton.neural.rollout_records import Sample, TrajectoryWriter


class Owner:
    def __init__(self,lib,pointer):self.lib,self.pointer=lib,pointer
    def __del__(self):
        if self.pointer:self.lib.ds_arrays_free(self.pointer);self.pointer=None


@lru_cache(maxsize=1)
def native():
    lib=load(Path(os.environ['DIRECT_STANDARD_LIBRARY']).resolve(strict=True))
    configure(lib)
    lib.ds_choice_kind.argtypes=[C.c_void_p,C.c_size_t];lib.ds_choice_kind.restype=C.c_void_p
    return lib


def check(lib,result):
    if not result:raise RuntimeError(lib.ds_error().decode())
    return result


def numpy_arrays(lib,pointer):
    owner=Owner(lib,check(lib,pointer));count=C.c_size_t()
    views=check(lib,lib.ds_arrays_views(owner.pointer,C.byref(count))); result={}
    for v in views[:count.value]:
        ctype=(C.c_float,C.c_int32,C.c_uint8)[v.dtype]
        storage=(ctype*v.length).from_address(v.data);storage._allocation_owner=owner
        arr=np.ctypeslib.as_array(storage)
        if v.dtype==2:arr=arr.view(np.bool_)
        result[KEYS[v.key]]=arr.reshape(tuple(v.dims[:v.rank]))
    return result


def label(lib,view,index):
    ptr=check(lib,lib.ds_choice_kind(view,index))
    try:return C.string_at(ptr).decode()
    finally:lib.ds_string_free(ptr)


def play(game, *, output=None, model=None, batcher=None, learner_seat=None, greedy=False, version=0,
         opponent_model=None, opponent_batcher=None, opponent_version=None, opponent_sha256=None,
         max_decisions=6000,max_turns=50,game_timeout=60,progress_path=None,record_ppo=False,**options):
    """Play a frozen-neural Standard game with private typed observations."""
    started=time.monotonic();row=dict(game,status='error',decisions=0,trajectories=[])
    lib=native();handle=None;writer=None;counts=Counter();last_turn=None
    try:
        if game.get('opponent_kind')!='frozen_neural' or learner_seat not in (0,1):
            raise ValueError('Experimental path supports frozen neural self-play only')
        if opponent_model is None or opponent_batcher is None or game['opponent_version']!=opponent_version or game['opponent_checkpoint_sha256']!=opponent_sha256:
            raise ValueError('Frozen opponent identity mismatch')
        if model.feature_spec!=opponent_model.feature_spec:raise ValueError('Mismatched model features')
        if output is not None and record_ppo:
            writer=TrajectoryWriter(Path(output)/f"game-{game['index']:06d}-p{learner_seat}.pkl.gz")
        decks=[standard_deck(d) for d in game['decks']]
        config=dict(decks=[[c.name for c in d.cards for _ in range(c.quantity)] for d in decks],
            seed=game['seed'],starting_player=game['starting_player'],deck_rules='Constructed')
        handle=check(lib,lib.ds_new(json.dumps(config,separators=(',',':')).encode()))
        rng=[np.random.default_rng(game['seed']*2+p) for p in range(2)]
        states=[None,None];previous=[None,None]
        while True:
            views=[]
            try:
                metadata=[]
                for player in range(2):
                    view=check(lib,lib.ds_observe(handle,player));views.append(view);meta=Metadata()
                    check(lib,lib.ds_metadata(view,C.byref(meta)));check(lib,lib.ds_history_observe(handle,view))
                    metadata.append(meta)
                meta=metadata[0];row.update(turn=meta.turn,final_life=[meta.life0,meta.life1])
                if progress_path is not None and (row['decisions']%100==0 or row['turn']!=last_turn):
                    path=Path(progress_path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(row));tmp.replace(path)
                last_turn=row['turn']
                if row['turn']>max_turns:
                    row.update(status='draw',reason='turn_limit',adjudicated=True,turn_limit=max_turns,draw=True,winner_seat=None,winner_deck=None,
                        learner_seat=learner_seat,learner_won=False);break
                if meta.outcome!=-1:
                    if metadata[1].outcome!=meta.outcome:raise RuntimeError('Inconsistent private outcomes')
                    winner=None if meta.outcome==2 else meta.outcome
                    row.update(status='draw' if winner is None else 'win' if winner==0 else 'loss',
                        winner_seat=winner,winner_deck=None if winner is None else game['decks'][winner],
                        draw=winner is None,learner_seat=learner_seat,learner_won=winner==learner_seat)
                    if writer is not None:
                        if not writer.count:raise RuntimeError('Empty trajectory')
                        row['trajectories'].append(dict(path=str(writer.path.resolve()),samples=writer.count,
                            player=learner_seat,deck=game['decks'][learner_seat]))
                    break
                if time.monotonic()-started>=game_timeout:
                    row.update(status='draw',reason='wall_clock_timeout',adjudicated=True,timeout_seconds=game_timeout,
                        draw=True,winner_seat=None,winner_deck=None,learner_seat=learner_seat,learner_won=False);break
                if row['decisions']>=max_decisions:row.update(status='cutoff',reason='decision_limit');break
                actor=meta.actor
                if actor not in (0,1) or metadata[1].actor!=actor:raise RuntimeError('Expected one prompted player')
                view=views[actor];size=C.c_size_t();check(lib,lib.ds_prepare(view,False,C.byref(size)))
                if size.value==1:index=0
                else:
                    data=numpy_arrays(lib,lib.ds_encode(view));data.update(numpy_arrays(lib,lib.ds_history_take(handle,view)))
                    side_model=model if actor==learner_seat else opponent_model
                    state=states[actor] if states[actor] is not None else np.zeros(side_model.memory_size,np.float32)
                    if previous[actor] is not None:data['selection']=previous[actor]
                    service=batcher if actor==learner_seat else opponent_batcher
                    role_version=version if actor==learner_seat else opponent_version
                    result=service.predict(dict(data,memory_in=state,advance_memory=np.ones(1,np.float32)),
                        version=role_version,identity=(game['seed'],actor),unbatched=True)
                    logits=np.asarray(result['logits'])[0].astype(np.float64)
                    if not np.isfinite(logits).all():raise ValueError('Nonfinite model scores')
                    logp=logits-np.max(logits);logp-=np.log(np.exp(logp).sum())
                    index=int(np.argmax(logits) if greedy else rng[actor].choice(len(logp),p=np.exp(logp)))
                    if writer is not None and actor==learner_seat:
                        value=float(np.asarray(result['value'])[0,0])
                        if not np.isfinite(value):raise ValueError('Nonfinite behavior value')
                        writer.write(Sample(data,state.copy(),True,index,value,logp.astype(np.float32)))
                    states[actor]=np.asarray(result['memory_out'])[0].copy();previous[actor]=data['candidates'][index].copy()
                action=label(lib,view,index)
                check(lib,lib.ds_apply_choice(handle,view,index));counts[f'p{actor}:{action}']+=1;row['decisions']+=1
            finally:
                for view in views:lib.ds_view_free(view)
    except Exception:
        row.update(status='error',error=traceback.format_exc(),trajectories=[])
    finally:
        if handle:lib.ds_free(handle)
        if writer is not None:
            writer.close()
            if row.get('adjudicated') and writer.count:
                row['trajectories'].append(dict(path=str(writer.path.resolve()),samples=writer.count,
                    player=learner_seat,deck=game['decks'][learner_seat],partial=True))
        row.update(seconds=time.monotonic()-started,actions=dict(counts))
        if record_ppo and row['status'] in ('win','loss','draw') and writer is not None:
            for trajectory in row['trajectories']:
                trajectory.update(reward=0. if row['status']=='draw' else 1. if learner_seat==row['winner_seat'] else -1.,
                    value_sum=writer.value_sum,value_square_sum=writer.value_square_sum)
    return row
