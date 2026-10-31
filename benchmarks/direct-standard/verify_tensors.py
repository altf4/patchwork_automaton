"""Lockstep privacy, complete legal choice and Standard tensor comparison.

This diagnostic reads reference JSON and exports action JSON only to compare
against the established adapter. The direct encoder itself consumes typed Rust
observations and outputs numeric arrays; the final benchmark must exclude this
shadow checking and debug export.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
from patchwork_automaton.backends.rustmage import RustMageMatch, domain_action, _plain
from patchwork_automaton.baselines.standard import StandardAggro
from patchwork_automaton.baselines.standard_decks import STANDARD_DECKS,standard_deck
from patchwork_automaton.neural import card_traits
from patchwork_automaton.neural.standard_features import StandardEncoder
from patchwork_automaton.neural.history import History
from patchwork_automaton.neural.rustmage_observation import learning_observation
from patchwork_automaton.neural.native_pipeline import KEYS
from probe_direct import load, Metadata

class View(C.Structure):
    _fields_=[('data',C.c_void_p),('length',C.c_size_t),('rank',C.c_size_t),('dims',C.c_size_t*4),('dtype',C.c_uint32),('key',C.c_uint32)]


def configure(lib):
    definitions=[('ds_configure',[C.c_char_p],C.c_bool),('ds_history_observe',[C.c_void_p,C.c_void_p],C.c_bool),
        ('ds_history_take',[C.c_void_p,C.c_void_p],C.c_void_p),('ds_prepare',[C.c_void_p,C.c_bool,C.POINTER(C.c_size_t)],C.c_bool),
        ('ds_apply_choice',[C.c_void_p,C.c_void_p,C.c_size_t],C.c_bool),
        ('ds_choice_debug_json',[C.c_void_p,C.c_size_t],C.c_void_p),
        ('ds_encode',[C.c_void_p],C.c_void_p),('ds_arrays_views',[C.c_void_p,C.POINTER(C.c_size_t)],C.POINTER(View)),
        ('ds_arrays_free',[C.c_void_p],None)]
    for name,args,result in definitions:
        f=getattr(lib,name);f.argtypes=args;f.restype=result
    names=(*card_traits.CARDS,*card_traits.ALIASES)
    encoder=StandardEncoder()
    cfg=dict(fingerprint=encoder.manifest()['fingerprint'],actions=list(encoder.actions),
        mana_values={name:card_traits.FACTS[card_traits.ALIASES.get(name,name)]['mana_value'] for name in names},
        traits={name:card_traits.traits_for(name) for name in names},
        identities={name:card_traits.ALIASES.get(name,name) for name in names},unknown_traits=card_traits.UNKNOWN_TRAITS)
    if not lib.ds_configure(json.dumps(cfg,separators=(',',':')).encode()):raise RuntimeError(lib.ds_error().decode())
    return encoder


def cjson(lib,pointer):
    if not pointer:raise RuntimeError(lib.ds_error().decode())
    try:return json.loads(C.string_at(pointer))
    finally:lib.ds_string_free(pointer)


def take_arrays(lib,ptr):
    if not ptr:raise RuntimeError(lib.ds_error().decode())
    try:
        count=C.c_size_t();views=lib.ds_arrays_views(ptr,C.byref(count)); result={}
        if not views:raise RuntimeError(lib.ds_error().decode())
        for v in views[:count.value]:
            ct=(C.c_float,C.c_int32,C.c_uint8)[v.dtype]
            data=np.ctypeslib.as_array((ct*v.length).from_address(v.data))
            if v.dtype==2:data=data.view(np.bool_)
            result[KEYS[v.key]]=data.reshape(tuple(v.dims[:v.rank])).copy()
        return result
    finally:lib.ds_arrays_free(ptr)


def arrays(lib,handle):
    return take_arrays(lib,lib.ds_encode(handle))


def run(args):
    lib=load(args.library.resolve(strict=True));encoder=configure(lib);games=[];view_count=choice_count=tensor_count=0
    for pair in [(a,b) for a in STANDARD_DECKS for b in STANDARD_DECKS]:
        decks=[standard_deck(d) for d in pair]
        expanded=[[c.name for c in deck.cards for _ in range(c.quantity)] for deck in decks]
        for repeat in range(args.games_per_pair):
            seed=args.seed+len(games);start=f'P{repeat%2}'
            handle=lib.ds_new(json.dumps(dict(decks=expanded,seed=seed,starting_player=start,deck_rules='Constructed')).encode())
            if not handle:raise RuntimeError(lib.ds_error().decode())
            try:
                with RustMageMatch(*decks,seed=seed,starting_player=start) as reference:
                    policies=[StandardAggro(d) for d in pair]
                    histories=[History(encoder.registry) for _ in range(2)]
                    for decision in range(args.max_decisions):
                        refs=[];handles=[]
                        try:
                            for viewer in (0,1):
                                obs=learning_observation(reference,f'P{viewer}');v=lib.ds_observe(handle,viewer)
                                if not v:raise RuntimeError(lib.ds_error().decode())
                                handles.append(v);refs.append(obs)
                                if cjson(lib,lib.ds_debug_json(v))!=obs.native:
                                    raise AssertionError(f'private view mismatch seed={seed} viewer={viewer} decision={decision}')
                                histories[viewer].observe(obs)
                                if not lib.ds_history_observe(handle,v):raise RuntimeError(lib.ds_error().decode())
                                view_count+=1
                            if refs[0].terminal or refs[0].turn>50:
                                games.append(dict(seed=seed,decks=pair,status='terminal' if refs[0].terminal else 'turn_limit',decisions=decision));break
                            actor=next(i for i,x in enumerate(refs) if x.decision is not None);obs=refs[actor];v=handles[actor]
                            choices=encoder.expand_options(obs.decision);count=C.c_size_t()
                            if not lib.ds_prepare(v,False,C.byref(count)):raise RuntimeError(lib.ds_error().decode())
                            if count.value!=len(choices):raise AssertionError(f'choice count mismatch seed={seed} decision={decision}: {count.value} vs {len(choices)}')
                            for i,(_,action) in enumerate(choices):
                                actual=cjson(lib,lib.ds_choice_debug_json(v,i))
                                if actual!=_plain(action):raise AssertionError(f'choice {i} mismatch seed={seed} decision={decision}: {actual} vs {_plain(action)}')
                                choice_count+=1
                            native=arrays(lib,v);expected=encoder.encode(obs,choices=choices).arrays
                            if set(native)!=set(expected):raise AssertionError('tensor key mismatch')
                            for key,x in expected.items():
                                y=native[key]
                                if x.shape!=y.shape or x.dtype!=y.dtype or not np.array_equal(x,y):
                                    raise AssertionError(f'tensor mismatch seed={seed} decision={decision} key={key} shapes={x.shape}/{y.shape} maxdiff={np.max(np.abs(x.astype(float)-y.astype(float))) if x.shape==y.shape else "shape"}')
                            if len(choices)>1:
                                actual_history=take_arrays(lib,lib.ds_history_take(handle,v))
                                expected_history=histories[actor].take(obs)
                                for key,x in expected_history.items():
                                    y=actual_history[key]
                                    if x.shape!=y.shape or x.dtype!=y.dtype or not np.array_equal(x,y):
                                        raise AssertionError(f'history mismatch seed={seed} decision={decision} key={key} shapes={x.shape}/{y.shape}')
                            tensor_count+=1
                            response=policies[actor].choose(obs)
                            option=next(o for o in obs.decision.options if o.id==response.option)
                            chosen_action=option.constraints['action'] if option.kind=='finite' else domain_action(option,response.value)
                            selected=next((i for i,(_,action) in enumerate(choices) if _plain(action)==_plain(chosen_action)),None)
                            if selected is None:raise AssertionError('baseline action missing from native choices')
                            if not lib.ds_apply_choice(handle,v,selected):raise RuntimeError(lib.ds_error().decode())
                            raw=reference._raw(obs.actor.value)
                            updated=(reference.native_game.step(raw,int(option.id.split(':')[1])) if option.kind=='finite'
                                     else reference.native_game.act(raw,domain_action(option,response.value)))
                            reference._after_action(updated,obs.actor.value,raw['revision'])
                        finally:
                            for v in handles:lib.ds_view_free(v)
                    else:raise RuntimeError('decision limit exceeded')
            finally:lib.ds_free(handle)
            print(json.dumps(dict(seed=seed,games=len(games),views=view_count,choices=choice_count,tensors=tensor_count)),flush=True)
    args.output.write_text(json.dumps(dict(status='complete',games=games,views=view_count,choices=choice_count,tensor_decisions=tensor_count),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--games-per-pair',type=int,default=2);p.add_argument('--max-decisions',type=int,default=6000);p.add_argument('--seed',type=int,default=1322610310000)
    run(p.parse_args())
