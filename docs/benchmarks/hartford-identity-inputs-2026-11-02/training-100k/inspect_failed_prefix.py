"""Forensic reconstruction of a preserved failed prefix; never a scheduled retry.

No production result or rollout is written. Learner actions are forced from its
original recording and compared with original inputs. Stop after twelve repeat
trigger choices, keeping a short raw engine trace for diagnosis.
"""
import asyncio,collections,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from patchwork_automaton.neural.rollout_records import read_trajectory
from patchwork_automaton.neural.standard_policy import StandardPolicy
from patchwork_automaton.neural.batching import InferenceBatcher
from patchwork_automaton.neural.direct_standard import library,string
from patchwork_automaton.neural.standard_features import StandardEncoder
from patchwork_automaton.rl import standard_direct
from examples.experiment_hartford_identity import with_executor_shutdown

async def main(root,dest):
    schedule=json.loads((root/'rollouts/round-0043/schedule.json').read_text())
    game=next(g for g in schedule if g['index']==42776)
    records=iter(read_trajectory(root/'rollouts/round-0043/trajectories/game-042776-p0.pkl.gz'))
    trace=collections.deque(maxlen=80);lib=library();seen=0;triggers=0
    class RecordedLearner:
        def predict(self,payload,**kwargs):
            nonlocal seen
            sample=next(records);seen+=1
            old=sample.arrays['_compact'];new=payload['_compact']
            for key in ('identity_ids','identity_base_nodes','identity_roles','identity_context','roles'):
                np.testing.assert_array_equal(getattr(old,key),getattr(new,key),err_msg=key)
            logits=np.full((1,len(old.roles)),-1000.,np.float32);logits[0,sample.action]=0.
            return dict(logits=logits,value=np.asarray([[sample.value]],np.float32),memory_out=np.zeros((1,0),np.float32))
    class TraceLibrary:
        def __getattr__(self,name):return getattr(lib,name)
        def ds_apply_choice(self,handle,view,index):
            nonlocal triggers
            action=json.loads(string(lib,lib.ds_choice_debug_json(view,index)))
            raw=json.loads(string(lib,lib.ds_debug_json(view)))
            trace.append(dict(action=action,raw=raw))
            if action['kind']=='PutTrigger':
                triggers+=1
                if triggers>=12:raise RuntimeError('Intentional forensic prefix stop; no production retry')
            return lib.ds_apply_choice(handle,view,index)
    standard_direct.library=lambda:TraceLibrary()
    opponent=StandardPolicy.load(root/'checkpoints/round-0039/policy.keras').model
    service=InferenceBatcher(opponent,max_batch=1,wait_ms=0,device='cpu');await service.start()
    service.set_version(39)
    metadata=SimpleNamespace(feature_spec=StandardEncoder.feature_spec,memory_size=0,input_format='sparse-identity-standard-v2')
    try:
        row=await asyncio.to_thread(standard_direct.play,game,model=metadata,batcher=RecordedLearner(),
            learner_seat=0,version=42,opponent_model=opponent,opponent_batcher=service,
            opponent_version=39,opponent_sha256=game['opponent_checkpoint_sha256'],
            game_timeout=20,max_turns=50,max_decisions=6000)
    finally:await service.close()
    result=dict(purpose='Original failed prefix reconstruction only; no successful games replayed',
        original_learner_inputs_matched=seen,trigger_choices=triggers,trace=list(trace),diagnostic_stop=row)
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(matched_samples=seen,trace_actions=len(trace),trigger_choices=triggers,
        diagnostic_error=row.get('error'),last_actions=[t['action'] for t in list(trace)[-25:]])))

if __name__=='__main__':
    asyncio.run(with_executor_shutdown(main(Path(sys.argv[1]),Path(sys.argv[2]))))
