"""Search matched scripted games for the 250K checkpoint's Harmonize adapter."""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
from patchwork_automaton.backends.rustmage import RustMageMatch, domain_action, _plain
from patchwork_automaton.baselines.standard import StandardAggro
from patchwork_automaton.baselines.standard_decks import standard_deck
from patchwork_automaton.neural.rustmage_observation import learning_observation
import verify_tensors as verify
from probe_direct import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--games',type=int,default=100)
    args=parser.parse_args()
    lib=load(args.library.resolve(strict=True));encoder=verify.configure(lib)
    decks=[standard_deck('izzet-prowess')]*2
    expanded=[[c.name for c in d.cards for _ in range(c.quantity)] for d in decks]
    for game in range(args.games):
        seed=1522610310000+game
        handle=lib.ds_new(json.dumps(dict(decks=expanded,seed=seed,starting_player=f'P{game%2}',deck_rules='Constructed')).encode())
        if not handle:raise RuntimeError(lib.ds_error().decode())
        try:
            with RustMageMatch(*decks,seed=seed,starting_player=f'P{game%2}') as ref:
                policies=[StandardAggro('izzet-prowess') for _ in range(2)]
                for step in range(6000):
                    observations=[learning_observation(ref,f'P{i}') for i in range(2)]
                    if observations[0].terminal or observations[0].turn>50:break
                    actor=next(i for i,o in enumerate(observations) if o.decision is not None)
                    obs=observations[actor]; view=lib.ds_observe(handle,actor)
                    if not view:raise RuntimeError(lib.ds_error().decode())
                    try:
                        choices=encoder.expand_options(obs.decision); count=C.c_size_t()
                        if not lib.ds_prepare(view,False,C.byref(count)):raise RuntimeError(lib.ds_error().decode())
                        if count.value!=len(choices):raise AssertionError('choice count mismatch')
                        kinds=[_plain(a).get('kind') for _,a in choices]
                        if obs.native['decision']=='harmonize' and len(choices)>1:
                            if verify.cjson(lib,lib.ds_debug_json(view))!=obs.native:raise AssertionError('Harmonize private view mismatch')
                            actual=verify.arrays(lib,view); expected=encoder.encode(obs,choices=choices).arrays
                            for key,x in expected.items():
                                y=actual[key]
                                if x.dtype!=y.dtype or x.shape!=y.shape or not np.array_equal(x,y):raise AssertionError('Harmonize tensor mismatch: '+key)
                            for i,(_,a) in enumerate(choices):
                                if verify.cjson(lib,lib.ds_choice_debug_json(view,i))!=_plain(a):raise AssertionError('Harmonize action mismatch')
                            chosen=next(i for i,(_,a) in enumerate(choices) if _plain(a)['object'] is not None)
                            response,_=choices[chosen]
                            option=next(o for o in obs.decision.options if o.id==response.option)
                            if not lib.ds_apply_choice(handle,view,chosen):raise RuntimeError(lib.ds_error().decode())
                            raw=ref._raw(obs.actor.value)
                            updated=ref.native_game.step(raw,int(option.id.split(':')[1]))
                            ref._after_action(updated,obs.actor.value,raw['revision'])
                            for viewer in (0,1):
                                post=lib.ds_observe(handle,viewer)
                                try:
                                    if verify.cjson(lib,lib.ds_debug_json(post))!=learning_observation(ref,f'P{viewer}').native:
                                        raise AssertionError('Post-Harmonize private view mismatch')
                                finally:lib.ds_view_free(post)
                            print(json.dumps(dict(status='complete',game=game,seed=seed,decision=step,choices=kinds)),flush=True)
                            return
                        # Prioritize Harmonize whenever it becomes available, and
                        # cast Wild Ride when possible so it enters the graveyard.
                        chosen=next((i for i,k in enumerate(kinds) if k=='AnnounceHarmonize'),None)
                        if chosen is None:
                            names={o['id']:o.get('name') for o in obs.native['objects']}
                            chosen=next((i for i,(_,a) in enumerate(choices)
                                if _plain(a).get('kind')=='AnnounceCast' and names.get(_plain(a).get('card'))=='Wild Ride'),None)
                        if chosen is None:
                            response=policies[actor].choose(obs)
                            option=next(o for o in obs.decision.options if o.id==response.option)
                            target=option.constraints['action'] if option.kind=='finite' else domain_action(option,response.value)
                            chosen=next(i for i,(_,a) in enumerate(choices) if _plain(a)==_plain(target))
                        response,action=choices[chosen]
                        option=next(o for o in obs.decision.options if o.id==response.option)
                        if not lib.ds_apply_choice(handle,view,chosen):raise RuntimeError(lib.ds_error().decode())
                        raw=ref._raw(obs.actor.value)
                        updated=(ref.native_game.step(raw,int(option.id.split(':')[1])) if option.kind=='finite'
                                 else ref.native_game.act(raw,domain_action(option,response.value)))
                        ref._after_action(updated,obs.actor.value,raw['revision'])
                    finally:lib.ds_view_free(view)
        finally:lib.ds_free(handle)
        print(json.dumps(dict(game=game,seed=seed,status='no_harmonize')),flush=True)
    raise RuntimeError('Harmonize decision not reached')

if __name__=='__main__':main()
