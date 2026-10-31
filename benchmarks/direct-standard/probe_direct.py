"""Diagnostic lockstep probe of the experimental typed RustMage bridge.

The debug JSON endpoint is used only here to compare redacted observations;
normal bridge observation operations return typed in-memory handles.
"""
import argparse
import ctypes as C
import json
from pathlib import Path

from patchwork_automaton.backends.rustmage import RustMageMatch
from patchwork_automaton.baselines.standard import StandardAggro
from patchwork_automaton.baselines.standard_decks import STANDARD_DECKS, standard_deck
from patchwork_automaton.neural.rustmage_observation import learning_observation

class Metadata(C.Structure):
    _fields_=[('revision',C.c_uint64),('turn',C.c_uint32),('viewer',C.c_uint32),('actor',C.c_int32),('active',C.c_uint32),('outcome',C.c_int32),('actions',C.c_size_t),('objects',C.c_size_t),('life0',C.c_int64),('life1',C.c_int64)]

def load(path):
    lib=C.CDLL(str(path))
    for name,args,result in [
        ('ds_version',[],C.c_uint32),('ds_error',[],C.c_char_p),('ds_new',[C.c_char_p],C.c_void_p),
        ('ds_free',[C.c_void_p],None),('ds_observe',[C.c_void_p,C.c_uint32],C.c_void_p),
        ('ds_view_free',[C.c_void_p],None),('ds_metadata',[C.c_void_p,C.POINTER(Metadata)],C.c_bool),
        ('ds_step',[C.c_void_p,C.c_uint32,C.c_uint64,C.c_size_t],C.c_bool),
        ('ds_debug_json',[C.c_void_p],C.c_void_p),('ds_string_free',[C.c_void_p],None)]:
        fn=getattr(lib,name);fn.argtypes=args;fn.restype=result
    if lib.ds_version()!=1:raise RuntimeError('Unexpected direct bridge ABI')
    return lib

def run(args):
    lib=load(args.library.resolve(strict=True));games=[];total_views=total_steps=0
    for pair in [(a,b) for a in STANDARD_DECKS for b in STANDARD_DECKS]:
        decks=[standard_deck(d) for d in pair]
        expanded=[[c.name for c in deck.cards for _ in range(c.quantity)] for deck in decks]
        for repeat in range(args.games_per_pair):
            seed=args.seed+len(games); start=f'P{repeat%2}'
            config=json.dumps(dict(decks=expanded,seed=seed,starting_player=start,deck_rules='Constructed')).encode()
            handle=lib.ds_new(config)
            if not handle:raise RuntimeError(lib.ds_error().decode())
            try:
                with RustMageMatch(*decks,seed=seed,starting_player=start) as reference:
                    policies=[StandardAggro(d) for d in pair]
                    for turn in range(args.max_decisions):
                        views=[]
                        for viewer in (0,1):
                            obs=learning_observation(reference,f'P{viewer}');view=lib.ds_observe(handle,viewer)
                            if not view:raise RuntimeError(lib.ds_error().decode())
                            try:
                                meta=Metadata()
                                if not lib.ds_metadata(view,C.byref(meta)):raise RuntimeError(lib.ds_error().decode())
                                pointer=lib.ds_debug_json(view)
                                if not pointer:raise RuntimeError(lib.ds_error().decode())
                                try: raw=json.loads(C.string_at(pointer))
                                finally:lib.ds_string_free(pointer)
                                if raw!=obs.native:raise AssertionError(f'Private view mismatch seed={seed} viewer={viewer} decision={turn}')
                                if meta.revision!=obs.revision or meta.turn!=obs.turn or meta.viewer!=viewer:
                                    raise AssertionError('Typed metadata mismatch')
                                total_views+=1;views.append(obs)
                            finally:lib.ds_view_free(view)
                        if views[0].terminal:
                            games.append(dict(seed=seed,decks=pair,status='terminal',decisions=turn));break
                        actor=next(i for i,obs in enumerate(views) if obs.decision is not None)
                        response=policies[actor].choose(views[actor]);option=next(o for o in views[actor].decision.options if o.id==response.option)
                        if option.kind!='finite':
                            games.append(dict(seed=seed,decks=pair,status='unsupported_numeric_domain',decisions=turn));break
                        raw=reference._raw(f'P{actor}');index=int(option.id.split(':')[1])
                        if not lib.ds_step(handle,actor,raw['revision'],index):raise RuntimeError(lib.ds_error().decode())
                        updated=reference.native_game.step(raw,index);reference._after_action(updated,f'P{actor}',raw['revision'])
                        total_steps+=1
                    else: raise RuntimeError('Decision limit exceeded')
            finally:lib.ds_free(handle)
    result=dict(status='complete',games=games,views=total_views,steps=total_steps,
                note='Diagnostic JSON export compares private typed observations against the baseline; no model tensors or timing measured.')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','views','steps')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--games-per-pair',type=int,default=2);p.add_argument('--max-decisions',type=int,default=6000);p.add_argument('--seed',type=int,default=1222610310000)
    run(p.parse_args())
