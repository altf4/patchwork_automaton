"""Audit both evaluation batches and graph all 30,000 outcomes."""
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from manage import HERE, ssh

AUDIT = r'''
import base64,gzip,hashlib,importlib.util,json,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
launch=json.loads(sys.stdin.read())
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
original_launch=json.loads((Path(launch['output'])/'original-launch.json').read_text())
original=load(original_launch['evaluation']['script'],'original_evaluation')
extension=load(launch['evaluation']['script'],'extension_evaluation')
if 'recovery' in launch:
    extension=load(launch['recovery']['script'],'publication_recovery')
batches=[original.audit(original_launch),extension.audit(launch)]
if batches[0]['manifest']['models']!=batches[1]['manifest']['models']:
    raise ValueError('Batch checkpoints differ')
if batches[0]['manifest']['engine']!=batches[1]['manifest']['engine']:
    raise ValueError('Batch engines differ')
groups=[json.loads(gzip.decompress(base64.b64decode(batch['games_gzip_b64']))) for batch in batches]
rows=sorted(groups[0]+groups[1],key=lambda row:row['index'])
if [len(group) for group in groups]!=[10000,20000] or len({row['index'] for row in rows})!=30000:
    raise ValueError('Wrong combined outcome count or duplicate identity')
if len({row['seed'] for row in rows})!=30000: raise ValueError('Repeated seed')
counts=Counter((row['learner_deck'],row['opponent_deck'],row['learner_seat'],row['starting_player']) for row in rows)
if len(counts)!=400 or set(counts.values())!={75}: raise ValueError('Combined schedule is unbalanced')
from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS
from patchwork_automaton.rl.hartford_evaluation import summarize
summary=summarize(rows,HARTFORD_DECKS)
payload=gzip.compress(original.canonical(rows),mtime=0)
result=dict(status='complete',games=30000,focal='diverse',opponent='control',training=False,
    completed_at=batches[1]['result']['completed_at'],summary=summary,
    batches=[dict(games=batch['result']['games'],summary=batch['result']['summary']) for batch in batches])
audit=dict(passed=True,audited_at=datetime.now(timezone.utc).isoformat(),games=30000,additional_games=20000,
    original_games=10000,unique_ids=True,unique_seeds=True,all_400_seat_start_strata_have_75_games=True,
    same_frozen_models_and_engine=True,original_results_preserved=True,
    games_gzip_sha256=hashlib.sha256(payload).hexdigest(),batch_audits=[batch['audit'] for batch in batches],
    interval_note=launch['extension']['statistical_note'])
print(json.dumps(dict(audit=audit,result=result,manifests=[batch['manifest'] for batch in batches],
    games_gzip_b64=base64.b64encode(payload).decode())))
'''


def main():
    launch = json.loads((HERE/'launch.json').read_text())
    value = ssh(launch,AUDIT,launch)
    if value['audit'].get('passed') is not True or value['audit']['games'] != 30000:
        raise ValueError('Combined audit failed')
    payload = base64.b64decode(value.pop('games_gzip_b64'),validate=True)
    if hashlib.sha256(payload).hexdigest() != value['audit']['games_gzip_sha256']:
        raise ValueError('Fetched outcome hash differs')
    output = HERE/'combined-30k'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Completed combined artifacts already exist')
    output.mkdir(exist_ok=True)
    (output/'games.json.gz').write_bytes(payload)
    for name,body in value.items():
        (output/(name+'.json')).write_text(json.dumps(body,indent=2,allow_nan=False)+'\n')
    spec=importlib.util.spec_from_file_location('ab_plot',HERE.parent/'finalize.py')
    plotting=importlib.util.module_from_spec(spec);spec.loader.exec_module(plotting)
    plotting.plot(value['result'],output)
    summary=value['result']['summary']
    (output/'README.md').write_text(f'''# Extended Hartford head-to-head

The diverse-opponent agent scored **{summary['score']:.2%}** over 30,000 games.
The nominal conservative 95% interval is {summary['score_ci95'][0]:.2%}–{summary['score_ci95'][1]:.2%}.
It recorded {summary['wins']:,} wins, {summary['losses']:,} losses and {summary['draws']:,} draws.
Score gives draws half a win.

![Combined results](head-to-head.png)

The original 10,000 outcomes and 20,000 fresh outcomes passed complete audits.
Both batches use the same final models, frozen gameplay source, engine, 48 workers,
temperature 1 and 20-second, 50-turn and 6,000-decision limits. Each of 100 ordered
matchups has 300 games, split evenly into its four seat/start combinations.
All original outcomes and adjudication diagnostics retain their pinned hashes.

The additional sample size was fixed after viewing the first 10K result.
Intervals are nominal fixed-sample Hoeffding bounds. One training run per recipe
contributes to this comparison.

Evidence accompanies this graph in audit.json, result.json, manifests.json and games.json.gz.
''')
    launch.update(status='complete',completed_at=value['result']['completed_at'],
        completion=dict(audit=str(output/'audit.json'),result=str(output/'result.json'),
            figure=str(output/'head-to-head.png'),games=30000,additional_games=20000,score=summary['score'],
            score_ci95=summary['score_ci95'],audited_at=value['audit']['audited_at']))
    (HERE/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch['completion'],indent=2))


if __name__ == '__main__':
    main()
