"""Validate the targeted repair on the unchanged failed identity, no other games."""
import asyncio,hashlib,json,sys
from pathlib import Path
from patchwork_automaton.neural.standard_policy import StandardPolicy
from patchwork_automaton.neural.batching import InferenceBatcher
from patchwork_automaton.neural.inference_processes import weights_digest
from patchwork_automaton.neural.rollout_records import read_trajectory
from patchwork_automaton.rl.standard_direct import play
from patchwork_automaton.rl.standard_ppo import validate_collection
from examples.experiment_hartford_identity import with_executor_shutdown

async def main(root,dest):
    dest.mkdir(exist_ok=False)
    schedule=json.loads((root/'rollouts/round-0043/schedule.json').read_text())
    game=next(row for row in schedule if row['index']==42776)
    paths=[root/'checkpoints/round-0042/policy.keras',root/'checkpoints/round-0039/policy.keras']
    before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert before[1]==game['opponent_checkpoint_sha256']
    learner,opponent=[StandardPolicy.load(path).model for path in paths]
    weights=[weights_digest(m.get_weights()) for m in (learner,opponent)]
    services=[InferenceBatcher(m,max_batch=1,wait_ms=0,device='cpu') for m in (learner,opponent)]
    for service,version in zip(services,(42,39)):
        await service.start();service.set_version(version)
    try:
        row=await asyncio.to_thread(play,game,output=dest,model=learner,batcher=services[0],
            learner_seat=0,version=42,opponent_model=opponent,opponent_batcher=services[1],
            opponent_version=39,opponent_sha256=before[1],
            game_timeout=20,max_turns=50,max_decisions=6000,record_ppo=True)
    finally:
        for service in services:await service.close()
    for t in row.get('trajectories',[]):
        t['sha256']=hashlib.sha256(Path(t['path']).read_bytes()).hexdigest()
    (dest/'result.json').write_text(json.dumps(row,indent=2)+'\n')
    assert row['status']=='draw' and not row.get('adjudicated'),row
    assert row['decisions']<6000 and row['turn']==9
    validate_collection([row],[game],42)
    assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert weights==[weights_digest(m.get_weights()) for m in (learner,opponent)]
    original=list(read_trajectory(root/'rollouts/round-0043/trajectories/game-042776-p0.pkl.gz'))
    actual=list(read_trajectory(row['trajectories'][0]['path']))
    import numpy as np
    matched=0
    for old,new in zip(original,actual):
        if old.action!=new.action:break
        for key in ('identity_ids','identity_roles','identity_base_nodes','identity_context'):
            np.testing.assert_array_equal(getattr(old.arrays['_compact'],key),getattr(new.arrays['_compact'],key))
        matched+=1
    assert matched==len(actual)
    audit=dict(passed=True,failed_identity=game['index'],unchanged_seed=game['seed'],
        checkpoint_hashes=before,decisions=row['decisions'],turn=row['turn'],rules_draw=True,
        original_decisions=6000,matched_original_learner_prefix=matched,model_weights_unchanged=True,
        original_attempt_preserved=True,no_successful_game_replayed=True,policy_evaluation_games=0)
    (dest/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))

if __name__=='__main__':
    asyncio.run(with_executor_shutdown(main(Path(sys.argv[1]),Path(sys.argv[2]))))
