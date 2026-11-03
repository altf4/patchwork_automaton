"""Bounded plumbing acceptance on disjoint seeds; no optimizer updates or strength score."""
import argparse
import asyncio
import gc
import json
from pathlib import Path

from audit_training import read, sha, verify_static


async def run(launch):
    from examples import train_hartford_ppo as driver
    from patchwork_automaton.neural.devices import configure_runtime, device_scope, assert_model_device
    from patchwork_automaton.neural.experiment import load_checkpoint
    from patchwork_automaton.neural.inference_processes import weights_digest
    from patchwork_automaton.neural.standard_policy import StandardPolicy
    from patchwork_automaton.rl.standard_selfplay_processes import StandardSelfplayCollector
    configure_runtime('gpu')
    loop=asyncio.get_running_loop()
    def tick():
        if not loop.is_closed():loop.call_later(.002,tick)
    loop.call_later(.002,tick)
    base=Path(launch['output']);output=base/'acceptance-v1'
    if output.exists():raise FileExistsError('Inspect existing acceptance evidence; never rerun implicitly')
    output.mkdir();verify_static(launch)
    initial={name:driver.checkpoint_hashes(Path(a['output'])/'checkpoints/round-0000') for name,a in launch['arms'].items()}
    with device_scope('gpu'):
        model,trainer,state=load_checkpoint(Path(launch['arms']['control']['output'])/'checkpoints',policy_type=StandardPolicy)
        other,other_trainer,other_state=load_checkpoint(Path(launch['arms']['diverse']['output'])/'checkpoints',policy_type=StandardPolicy)
    assert_model_device(model,'gpu');assert_model_device(other,'gpu')
    before=driver.training_state(model,trainer)
    if before!=driver.training_state(other,other_trainer):raise ValueError('Loaded arms do not share exact model/optimizers/RNG')
    del other,other_trainer;gc.collect()
    config=dict(launch['arms']['diverse']['config'],seed=442611030000)
    planned=driver.planned_training(config,1)
    script=[g for g in planned if g['opponent_kind']=='scripted']
    selected=[]
    for deck in ('mono-green-landfall','izzet-prowess','bw-lifegain'):
        for seat in (0,1):
            for start in ('P0','P1'):
                selected.append(next(g for g in script if (g['opponent_deck'],g['learner_seat'],g['starting_player'])==(deck,seat,start)))
    versions=sorted({g['opponent_version'] for g in planned if g['opponent_kind']=='frozen_neural'})
    for version in versions:
        selected.extend([g for g in planned if g['opponent_kind']=='frozen_neural' and g['opponent_version']==version][:9])
    if len(selected)!=48 or len({g['index'] for g in selected})!=48:raise ValueError('Incomplete acceptance schedule')
    paths={0:Path(launch['arms']['diverse']['output'])/'checkpoints/round-0000/policy.keras'}
    paths.update({int(n):Path(v['path'])/'policy.keras' for n,v in config['fork']['external_opponents'].items()})
    for game in selected:
        if game['opponent_kind']=='frozen_neural':game['opponent_checkpoint_sha256']=sha(paths[game['opponent_version']])
    collectors=[]
    async def pin(numbers):
        weights={}
        with device_scope('cpu'):
            for number in numbers:
                opponent=StandardPolicy.load(paths[number]).model;weights[number]=opponent.get_weights()
            for _ in range(2):
                collectors.append(StandardSelfplayCollector(model,opponent,workers=24,inference_processes=1,opponent_inference_processes=1))
        await driver.settle_all(c.start(version=0,opponent_weights=weights) for c in collectors)
        return {n:weights_digest(w) for n,w in weights.items()}
    try:
        rows=await driver.collect_round(planned=selected,number=1,directory=output/'collection',
            collectors=collectors,pin_opponent=None,pin_opponents=pin,game_timeout=20,
            progress=lambda phase,**kw:print(json.dumps(dict(phase=phase,**kw)),flush=True))
        await driver.quiesce_collectors(collectors)
        reproduction=driver.verify_behavior(model,rows)
        if driver.training_state(model,trainer)!=before:raise ValueError('Acceptance changed model/optimizer/RNG')
        for name,arm in launch['arms'].items():
            if driver.checkpoint_hashes(Path(arm['output'])/'checkpoints/round-0000')!=initial[name]:raise ValueError('Prepared arm changed')
        verify_static(launch)
        result=dict(passed=True,games=len(rows),scripted_games=12,neural_games=36,
            production_games=0,optimizer_updates=0,policy_strength_evaluation=False,
            matching_loaded_model_optimizer_rng=True,gpu_placement=True,
            initial_checkpoints=initial,behavior_reproduction=reproduction,
            no_production_or_parent_artifacts_changed=True)
        driver.atomic_json(output/'result.json',result);print(json.dumps(result),flush=True)
    finally:
        await driver.settle_all(c.close() for c in collectors)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',type=Path,required=True)
    asyncio.run(run(read(parser.parse_args().launch)))
