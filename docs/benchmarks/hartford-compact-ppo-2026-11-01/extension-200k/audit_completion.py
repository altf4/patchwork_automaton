import json,pathlib,subprocess,shlex
BASE=pathlib.Path('/home/altf4/Code/mtgo-bot/patchwork_automaton/docs/benchmarks/hartford-compact-ppo-2026-11-01');d=json.loads((BASE/'launch.json').read_text())
code=r'''
import json,pathlib,sys,hashlib,collections,os,zipfile,io
from datetime import datetime,timezone
import numpy as np
import h5py
p=json.loads(sys.argv[1]);r=pathlib.Path(p['production']['output']);project=pathlib.Path(p['remote']['working_directory']);sys.path[:0]=[str(project),str(project/'src')]
from patchwork_automaton.rl.hartford_schedule import training_schedule,evaluation_schedule,previous_checkpoints
from patchwork_automaton.rl.standard_ppo import validate_collection
from patchwork_automaton.rl.hartford_evaluation import validate_row,summarize
from examples.train_hartford_ppo import checkpoint_configuration, evaluation_source_pins, required_evaluation_rounds
report=json.loads((r/'report.json').read_text());guard=json.loads(pathlib.Path(p['production']['guard_status']).read_text());manifest=json.loads((r/'manifest.json').read_text());cfg=report['config'];decks=tuple(cfg['training_decks'])
assert report['status']=='complete' and report['completed_games']==cfg['rounds']*1000 and len(report['rounds'])==cfg['rounds']
assert set(report['checkpoints'])=={str(n) for n in range(cfg['rounds']+1)}
assert set(report['evaluations'])=={str(n) for n in sorted(required_evaluation_rounds(cfg,report))}
assert guard['status']=='exited' and guard['guard_exit_code']==0 and guard['child_exit_code']==0
assert guard['windows_awake']['release']['graceful_release']
try:os.kill(guard['child_pid'],0)
except ProcessLookupError:pass
else:raise RuntimeError('Learner still alive')
def sha(path):
 with pathlib.Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for name,want in manifest['pins']['source'].items():assert sha(project/name)==want,name
engine=manifest['pins']['readiness']['engine'];assert sha(engine['library'])==engine['library_sha256']
np_arrays=0;h5_datasets=0
for n,files in report['checkpoints'].items():
 cp=r/'checkpoints'/('round-'+n.zfill(4))
 for name,want in files.items():assert sha(cp/name)==want,(n,name)
 state=json.loads((cp/'state.json').read_text());assert state['round']==int(n) and state['config']==checkpoint_configuration(report,cfg,int(n))
 for name in ('optimizer.npz','critic-optimizer.npz'):
  with np.load(cp/name,allow_pickle=False) as z:
   for key in z.files:assert np.isfinite(z[key]).all(),(n,name,key);np_arrays+=1
 with zipfile.ZipFile(cp/'policy.keras') as archive:
  with h5py.File(io.BytesIO(archive.read('model.weights.h5')),'r') as h:
   datasets=[]
   def visit(name,obj):
    if isinstance(obj,h5py.Dataset):datasets.append(name)
   h.visititems(visit)
   for name in datasets:
    values=h[name][()]
    if np.issubdtype(values.dtype,np.number):assert np.isfinite(values).all(),(n,name)
   h5_datasets+=len(datasets)
ids=set();seeds=set();training=heldout=0;collection_outcomes=collections.Counter();collection_adjudications=collections.Counter()
for number,row in enumerate(report['rounds'],1):
 assert row['round']==number and row['training_games']==800 and row['validation_games']==200 and row['validation_read_only']
 assert sha(row['games_file'])==row['games_sha256'];games=json.loads(pathlib.Path(row['games_file']).read_text());assert len(games)==1000
 planned=training_schedule(decks,round_id=number,seed=cfg['seed'])
 for game in planned:game['opponent_checkpoint_sha256']=report['checkpoints'][str(game['opponent_version'])]['policy.keras']
 validate_collection(games,planned,number-1)
 strata=collections.Counter((g['learner_deck'],g['opponent_deck'],g['partition']) for g in games);assert len(strata)==200 and all(c==(8 if s[2]=='training' else 2) for s,c in strata.items())
 assert collections.Counter(str(g['opponent_version']) for g in games)==collections.Counter(row['opponent_games'])
 for game in games:
  assert game['index'] not in ids and game['seed'] not in seeds;ids.add(game['index']);seeds.add(game['seed'])
  training+=game['partition']=='training';heldout+=game['partition']=='validation';collection_outcomes[game['status']]+=1
  if game.get('adjudicated'):collection_adjudications[game.get('reason','unknown')]+=1
assert len(ids)==len(seeds)==cfg['rounds']*1000 and training==cfg['rounds']*800 and heldout==cfg['rounds']*200
ev_count=0;ev_seeds=set();ev_outcomes=collections.Counter();ev_adjudications=collections.Counter()
for number in sorted(required_evaluation_rounds(cfg,report)):
 saved=report['evaluations'][str(number)];root=r/'evaluation'/('round-'+str(number).zfill(4));published=json.loads((root/'summary.json').read_text());assert published==saved and saved['complete']
 assert sha(root/'manifest.json')==saved['manifest_sha256'];assert json.loads((root/'manifest.json').read_text())['source_manifest']==evaluation_source_pins(report,manifest['pins'],number)
 assert saved['opponent_rounds']==list(previous_checkpoints(number));allrows=[]
 for opponent in previous_checkpoints(number):
  planned=evaluation_schedule(decks,checkpoint_round=number,opponent_round=opponent,games_per_cell=40,seed=cfg['eval_seed'])
  expected={g['index']:dict(g,opponent_checkpoint_sha256=report['checkpoints'][str(opponent)]['policy.keras']) for g in planned};seen=set()
  for path in (root/('opponent-'+str(opponent).zfill(4))/'games').glob('*.json'):
   game=json.loads(path.read_text());validate_row(game,expected[game['index']],checkpoint_round=number);assert game['index'] not in seen;seen.add(game['index']);assert game['seed'] not in ev_seeds and game['seed'] not in seeds;ev_seeds.add(game['seed']);allrows.append(game)
   ev_outcomes['draw' if game['draw'] else 'win' if game['learner_won'] else 'loss']+=1
   if game.get('adjudicated'):ev_adjudications[game.get('reason','unknown')]+=1
  assert seen==set(expected)
 assert len(allrows)==12000 and summarize(allrows,decks)==saved['aggregate'];ev_count+=len(allrows)
assert ev_count==report['evaluation_games']==len(required_evaluation_rounds(cfg,report))*12000
archive=r/'training-target-extension-200k-v1';amendment=json.loads((archive/'amendment.json').read_text());assert sha(archive/'preserved-artifact-hashes.json')==amendment['preserved_artifact_hashes_sha256']
for name,want in json.loads((archive/'preserved-artifact-hashes.json').read_text()).items():assert sha(r/name)==want,name
for name,want in amendment['previous_pins']['source'].items():assert sha(pathlib.Path(amendment['previous_source'])/name)==want,name
for recovery in report.get('runtime_amendments',[]):
 archive=pathlib.Path(recovery['archive']);assert sha(archive/'original-round-artifact-hashes.json')==recovery['original_round_hashes_sha256']
 for name,want in json.loads((archive/'original-round-artifact-hashes.json').read_text()).items():
  relative=pathlib.Path(name).relative_to('rollouts/round-0120');assert sha(archive/'round-0120-before'/relative)==want,name
 for name,want in recovery['failed_attempt_sha256'].items():assert sha(archive/'attempt-000'/name)==want,name
 committed={row['index']:row for row in json.loads((r/'rounds/round-0120/games.json').read_text())}
 preserved=0
 for path in (archive/'round-0120-before/results').glob('*.json'):
  original=json.loads(path.read_text())
  if original['status'] in ('win','loss','draw'):assert committed[original['index']]==original;preserved+=1
 assert preserved==recovery['preserved_successful_round120_games']==999
 retried=committed[119772];assert retried['seed']==recovery['failed_identity']['seed'] and retried['status']=='draw' and retried['decisions']<6000
 for number,files in recovery['preserved_checkpoints'].items():assert report['checkpoints'][number]==files
 for name,want in recovery['previous_pins']['source'].items():assert sha(pathlib.Path(recovery['previous_source'])/name)==want,name
disabled=report.get('evaluation_disable_amendment')
if disabled:
 archive=pathlib.Path(disabled['archive']);assert sha(archive/'preserved-artifact-hashes.json')==disabled['preserved_artifact_hashes_sha256']
 for name,want in json.loads((archive/'preserved-artifact-hashes.json').read_text()).items():assert sha(r/name)==want,name
 assert sorted(map(int,report['evaluations']))==disabled['retained_evaluation_rounds'] and ev_count==disabled['retained_evaluation_games']
 for name,want in disabled['previous_pins']['source'].items():assert sha(pathlib.Path(disabled['previous_source'])/name)==want,name
 partial=disabled['partial_collection']
 if partial:
  committed={g['index']:g for g in json.loads((r/'rounds'/f"round-{partial['round']:04d}"/'games.json').read_text())}
  for name,want in partial['files'].items():
   archived=archive/'partial-rollout-before'/name;assert sha(archived)==want,name
   if name.startswith('results/'):
    game=json.loads(archived.read_text());assert committed[game['index']]==game
samples=[json.loads(s) for s in pathlib.Path(p['production']['resource_samples']).open()]
minimum=min(z['available_memory_bytes'] for z in samples)
assert minimum>=1536*1024**2 and max(z['low_memory_seconds'] for z in samples)==0
result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),finished_at=report['finished_at'],training_rounds=cfg['rounds'],collection_games=cfg['rounds']*1000,training_partition_games=training,holdout_games=heldout,checkpoint_count=cfg['rounds']+1,evaluation_rounds=list(sorted(required_evaluation_rounds(cfg,report))),evaluation_games=ev_count,training_schedules_and_whole_game_partitions_verified=True,unique_collection_indices_and_seeds=True,unique_evaluation_seeds=True,all_checkpoint_file_hashes_verified=True,finite_optimizer_arrays_checked=np_arrays,finite_model_datasets_checked=h5_datasets,source_and_engine_hashes_verified=True,evaluation_summaries_recomputed=True,guard_exit_code=0,learner_stopped=True,windows_awake_released=True,minimum_available_gib_last_supervised_segment=minimum/1024**3,collection_outcomes=dict(collection_outcomes),collection_adjudications=dict(collection_adjudications),evaluation_outcomes=dict(ev_outcomes),evaluation_adjudications=dict(ev_adjudications),final_score=(report['evaluations'].get(str(cfg['rounds'])) or {}).get('aggregate',{}).get('score'),final_score_ci95=(report['evaluations'].get(str(cfg['rounds'])) or {}).get('aggregate',{}).get('score_ci95'),future_evaluations_disabled=bool(cfg.get('skip_evaluations')),last_evaluated_checkpoint=max(map(int,report['evaluations']),default=None),final_checkpoint=str(r/'checkpoints'/f"round-{cfg['rounds']:04d}"/'policy.keras'))
print(json.dumps(result))
'''
p=subprocess.run([*d['remote']['ssh_argv_prefix'],shlex.join(['/usr/bin/env','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',d['remote']['venv_python'],'-c',code,json.dumps(d)])],capture_output=True,text=True,timeout=180)
if p.returncode:print(p.stderr);raise SystemExit(p.returncode)
x=json.loads(p.stdout);(BASE/'extension-200k/completion-audit.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
