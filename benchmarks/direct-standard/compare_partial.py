"""Compare the recorded prefix of an adjudicated game to its matched run."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from patchwork_automaton.neural.rollout_records import read_trajectory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--direct',type=Path,required=True)
    p.add_argument('--index',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    def row(path):
        return next(r for line in path.read_text().splitlines() if (r:=json.loads(line))['index']==a.index)
    left,right=row(a.baseline),row(a.direct)
    if not left.get('adjudicated') or right.get('adjudicated'):raise ValueError('Expected only baseline to be adjudicated')
    x=list(read_trajectory(left['trajectories'][0]['path']))
    y=list(read_trajectory(right['trajectories'][0]['path']))
    errors=[]
    for i,(u,v) in enumerate(itertools.zip_longest(x,y)):
        if u is None:break
        if v is None or u.action!=v.action or u.advance!=v.advance or set(u.arrays)!=set(v.arrays):
            errors.append(dict(record=i,reason='action or length'));continue
        for key in u.arrays:
            aa,bb=u.arrays[key],v.arrays[key]
            if aa.dtype!=bb.dtype or aa.shape!=bb.shape or not np.array_equal(aa,bb):
                errors.append(dict(record=i,reason=key));break
    result=dict(index=a.index,baseline_reason=left.get('reason'),baseline_records=len(x),direct_records=len(y),
        prefix_mismatch_count=len(errors),prefix_mismatches=errors[:20],direct_status=right['status'],
        direct_turn=right['turn'],direct_decisions=right['decisions'],direct_final_life=right['final_life'])
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if errors:raise AssertionError('Recorded prefix mismatch')

if __name__=='__main__':main()
