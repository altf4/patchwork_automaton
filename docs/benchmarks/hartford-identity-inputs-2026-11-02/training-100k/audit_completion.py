"""Read-only production completion audit; writes evidence only beside this file."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("identity_monitor", BASE / "monitor.py")
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)

EXTRA = r'''
from collections import Counter
from copy import deepcopy
import gzip, statistics
from examples import train_hartford_ppo as driver
from patchwork_automaton.rl.hartford_schedule import training_schedule
from patchwork_automaton.rl.standard_ppo import validate_collection
from patchwork_automaton.rl.hartford_preflight import hartford_preflight

assert not errors, errors
g=x['guard'];config=report['config']
assert g['status']=='exited' and g['child_exit_code']==g['guard_exit_code']==g['exit_code']==0
assert not x['owned_child_alive'] and g['attempts']==1
assert report['status']=='complete' and report['completed_games']==100000
assert len(report['rounds'])==100 and [v['round'] for v in report['rounds']]==list(range(1,101))
assert report['evaluation_games']==driver.validate_evaluation_completion(report,config)==0
assert config['skip_evaluations'] and not report['evaluations']
assert config['use_identity'] and not config['use_rules']
assert config['architecture']=='compact-identity-standard-v2'
assert '--skip-evaluations' in g['command']
assert set(report['checkpoints'])==set(map(str,range(101)))
assert read(root/'checkpoints/latest.json')=={'checkpoint':'round-0100'}

# The frozen history validator normally republishes latest.json and missing
# committed markers. Replace its two writers with equality-only assertions:
# this audit rejects publication gaps and performs no production writes.
def require_existing_json(path,value):
 assert Path(path).is_file() and read(Path(path))==value, str(path)
driver.atomic_json=require_existing_json
driver.pin_json=require_existing_json
history=deepcopy(report)
assert driver.restore_history(history,root/'checkpoints',config)==100
assert history==report

manifest=read(root/'manifest.json')
assert manifest['config']==config
assert driver.source_manifest()==manifest['pins']['source']
assert all(s['source_hashes'][name]==digest for name,digest in manifest['pins']['source'].items())
assert manifest['pins']['readiness']['engine']==s['engine']
readiness=hartford_preflight(playback='direct',checkout=s['engine']['checkout'],
 library_path=s['engine']['library'],build_manifest_path=s['engine']['build_manifest'])
assert readiness['ready'], readiness['errors']
assert readiness==manifest['pins']['readiness']
total=Counter();status=Counter();adjudications=Counter();ids=set();seeds=set()
schedule_digests={};finite_models=0;finite_optimizer_arrays=0;retained_trajectories=0
for number in range(101):
 cp=root/'checkpoints'/f'round-{number:04d}'
 state=read(cp/'state.json')
 assert 'trainer_rng_state' in state
 # Test saved generator restoration without consuming a random draw.
 rng=np.random.default_rng();rng.bit_generator.state=state['trainer_rng_state']
 for name in ('optimizer.npz','critic-optimizer.npz'):
  with np.load(cp/name,allow_pickle=False) as data:
   assert data.files and all(np.isfinite(data[k]).all() for k in data.files)
   finite_optimizer_arrays+=len(data.files)
 with zipfile.ZipFile(cp/'policy.keras') as archive:
  with h5py.File(io.BytesIO(archive.read('model.weights.h5')),'r') as h:
   numeric=[]
   def verify(name,item):
    if isinstance(item,h5py.Dataset) and np.issubdtype(item.dtype,np.number):
     assert np.isfinite(item[()]).all(), (number,name)
     numeric.append(name)
   h.visititems(verify)
   assert len(numeric)==41
   finite_models+=1
 if not number:continue
 round_result=report['rounds'][number-1]
 assert round_result['validation_read_only']
 rows=read(Path(round_result['games_file']))
 planned=training_schedule(config['training_decks'],round_id=number,seed=config['seed'])
 planned=[dict(v,opponent_checkpoint_sha256=report['checkpoints'][str(v['opponent_version'])]['policy.keras']) for v in planned]
 validate_collection(rows,planned,number-1)
 schedule_digests[str(number)]=hashlib.sha256(json.dumps(planned,sort_keys=True).encode()).hexdigest()
 assert Counter(v['partition'] for v in rows)=={'training':800,'validation':200}
 cells=Counter((v['learner_deck'],v['opponent_deck']) for v in rows)
 assert len(cells)==100 and set(cells.values())=={10}
 assert Counter(str(v['opponent_version']) for v in rows)==round_result['opponent_games']
 for row in rows:
  assert row['index'] not in ids and row['seed'] not in seeds
  ids.add(row['index']);seeds.add(row['seed']);total[row['partition']]+=1;status[row['status']]+=1
  if row.get('adjudicated'):adjudications[str(row.get('adjudication_reason',row.get('reason','recorded_adjudication')))]+=1
  for trajectory in row['trajectories']:
   path=Path(trajectory['path'])
   if path.exists():
    assert sha(path)==trajectory['sha256']
    with gzip.open(path,'rb') as f:
     while f.read(1024*1024):pass
    retained_trajectories+=1
 assert round_result['update']['rejected_epochs']>=0
assert ids==set(range(100000))
assert seeds==set(range(config['seed'],config['seed']+100000))
assert total=={'training':80000,'validation':20000}

archive=root/'decision-limit-investigation-v1'
amendment=read(archive/'runtime-amendment.json')
assert report['runtime_amendments']==[amendment]
assert amendment['kind']=='inert-entry-trigger-loop-recovery-v1' and amendment['through_checkpoint']==42
assert amendment['new_pins']==manifest['pins'] and amendment['config_unchanged']
old=read(archive/'manifest.json')
assert old['pins']==amendment['previous_pins'] and old['config']==config
oldproject=root.parent/'project-v1'
assert driver.source_manifest(oldproject)==old['pins']['source']
oldengine=old['pins']['readiness']['engine']
assert sha(Path(oldengine['library']))==oldengine['library_sha256']
preserved=read(archive/'preserved-hashes.json')
assert isinstance(preserved,dict)
for name,digest in preserved.items():assert sha(archive/name)==digest, name
for number,files in amendment['previous_checkpoint_hashes'].items():
 assert report['checkpoints'][number]==files
assert len(amendment['previous_checkpoint_hashes'])==43
archived_results={v['index']:v for v in [read(p) for p in (archive/'round-0043/results').glob('game-*.json')]}
committed={v['index']:v for v in read(root/'rounds/round-0043/games.json')}
assert len(archived_results)==len(committed)==1000
assert len(amendment['preserved_successful_results'])==999
for identity,digest in amendment['preserved_successful_results'].items():
 identity=int(identity)
 assert sha(archive/'round-0043/results'/f'game-{identity:06d}.json')==digest
 assert committed[identity]==archived_results[identity]
original=archive/'round-0043/results/game-042776.json'
assert sha(original)==amendment['original_failed_result_sha256']
assert archived_results[42776]['decisions']==6000
retry=read(archive/'retry-fixed-engine-v1/audit.json')
assert retry==amendment['retry_audit'] and retry['passed'] and retry['matched_original_learner_prefix']==20
assert retry['unchanged_seed']==committed[42776]['seed']==412610552776
assert committed[42776]['status']=='draw' and committed[42776]['decisions']==196
for t in committed[42776]['trajectories']:assert sha(Path(t['path']))==t['sha256']
assert report['rounds'][42]['collection_inference']==[]
assert report['rounds'][42]['collection_seconds']<1

guards=[]
for directory in (root.parent/'run-guard-v1',guard):
 gs=read(directory/'guard-status.json')
 samples=[json.loads(line) for line in (directory/'host-memory-samples.jsonl').read_text().splitlines()]
 assert max(z['low_memory_seconds'] for z in samples)==0
 guards.append(dict(path=str(directory),started_at=gs['started_at'],finished_at=gs.get('finished_at'),
  elapsed_seconds=gs.get('elapsed_seconds'),child_exit_code=gs['child_exit_code'],
  minimum_available_bytes=min(z['available_memory_bytes'] for z in samples)))
def metrics(value):
 return {k:v for k,v in value.items() if isinstance(v,(str,int,float,bool)) or v is None or (isinstance(v,dict) and k in ('critic','behavior_critic','value_before','value_after','value_mse','value_mse_change','policy_surrogate_gain','return_surrogate_gain'))}
timings={key:dict(total_seconds=sum(v[key] for v in report['rounds']),
 mean_seconds=statistics.mean(v[key] for v in report['rounds']),
 median_seconds=statistics.median(v[key] for v in report['rounds']))
 for key in ('collection_seconds','update_seconds','validation_seconds')}
result=dict(passed=True,audited_at=datetime.now(timezone.utc).isoformat(),
 completed_rounds=100,collection_games=100000,training_games=80000,holdout_games=20000,evaluation_games=0,
 parent_exposure_games=200000,total_policy_exposure_games=300000,policy_strength_evaluated=False,
 first_seed=min(seeds),last_seed=max(seeds),outcomes=dict(status),adjudications=dict(adjudications),
 source_files_verified=len(s['source_hashes']),runtime_readiness_verified=True,checkpoints_verified=101,finite_models_verified=finite_models,
 finite_optimizer_arrays_verified=finite_optimizer_arrays,rng_states_verified=101,
 retained_trajectory_hashes_and_gzip_verified=retained_trajectories,
 final_policy_sha256=report['checkpoints']['100']['policy.keras'],
 parent_checkpoint_hashes=s['parent_checkpoint_hashes'],current_engine=s['engine'],
 final_guard_exit_code=0,guards=guards,timings=timings,report_started_at=report['started_at'],report_finished_at=report['finished_at'],
 last_update=metrics(report['rounds'][-1]['update']),last_holdout=metrics(report['rounds'][-1]['validation']),
 total_rejected_epochs=sum(v['update']['rejected_epochs'] for v in report['rounds']),
 recovery=dict(original_archive_files_verified=len(preserved),original_checkpoints_preserved=43,
  successful_round43_outcomes_preserved=999,failed_identity=42776,original_attempt_preserved=True,
  retry_decisions=196,matched_original_prefix_samples=20,reused_round43_games=1000,
  successful_games_replayed=0,amendment_sha256=sha(archive/'runtime-amendment.json')),
 schedule_sha256=schedule_digests,
 production_report_sha256=sha(root/'report.json'),production_manifest_sha256=sha(root/'manifest.json'))
print(json.dumps(result,allow_nan=False))
'''


def main():
    launch = json.loads((BASE / 'launch.json').read_text())
    keys = ('output', 'guard', 'project', 'source_hashes', 'engine', 'control',
            'original_output', 'parent_checkpoint_hashes')
    config = {key: launch[key] for key in keys}
    program = monitor.REMOTE.rsplit('print(json.dumps(x))', 1)[0] + EXTRA
    command = ['env', *[k + '=' + v for k, v in launch['environment'].items()],
               launch['remote']['venv_python'], '-c', program, json.dumps(config)]
    completed = subprocess.run(launch['remote']['ssh_argv_prefix'] + [shlex.join(command)],
                               capture_output=True, timeout=180)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = BASE / 'completion-audit'
    folder.mkdir(exist_ok=True)
    (folder / (stamp + '-stdout.log')).write_bytes(completed.stdout)
    (folder / (stamp + '-stderr.log')).write_bytes(completed.stderr)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode())
    result = json.loads(completed.stdout)
    result['auditor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path = folder / (stamp + '-audit.json')
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(passed=result['passed'], audit=str(path),
                         collection_games=result['collection_games'],
                         evaluation_games=result['evaluation_games'])))


if __name__ == '__main__':
    main()
