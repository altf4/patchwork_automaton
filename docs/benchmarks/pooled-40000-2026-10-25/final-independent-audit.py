"""Read-only independent audit of completed results; no model execution or games."""
import collections,datetime,hashlib,json,pathlib,random,zipfile
import numpy as np
B=pathlib.Path(__file__).resolve().parent
C=B/'live-backup-recovery2/current'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
checks=[]
def check(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
study=read(C/'study.json');report=read(C/'training/report.json');supervisor=read(B/'live-backup-recovery2/supervisor.json')
check(study['status']==report['status']==supervisor['status']=='complete','study, trainer and local supervisor complete')
check([r['round'] for r in report['rounds']]==list(range(1,41)),'40 consecutive rounds')
all_games=[];host=collections.Counter();fallback=0;round_summary=[]
for r in report['rounds']:
 n=r['round']; games=r['games']; ids={g['index'] for g in games}; split=r['split'];train=set(split['training_indices']);held=set(split['validation_indices'])
 check(len(games)==len(ids)==1000 and ids==set(range((n-1)*1000,n*1000)),f'round {n} full unique scheduled IDs')
 check(len(train)==800 and len(held)==200 and not train&held and train|held==ids,f'round {n} disjoint whole-game 800/200 split')
 check(all(g['partition']==('training' if g['index'] in train else 'validation') for g in games),f'round {n} recorded partitions match split')
 check(all(g['status'] in ('win','loss') and g['seed']==300262025+g['index'] and g['seat']==g['index']%2 and g['starting_player']==('P0' if g['index']%4<2 else 'P1') for g in games),f'round {n} terminal seed/seat/start schedule')
 check(r['collection_policy']['version']==n-1,f'round {n} behavior policy version')
 pool=[p for p in r['collection_resources']['pool']['rounds'] if p['round']==n]
 check(len(pool)==1,f'round {n} select matching pool record only')
 p=pool[0];counts=collections.Counter(g['collection_host'] for g in games)
 check(counts['laptop']==p['laptop_games']==r['collection_resources']['laptop_games'] and counts['raytracer']==p['raytracer_games']==r['collection_resources']['local_games'],f'round {n} actual host contributions match')
 check(p['laptop_games']+p['fallback_games']==128 and p['raytracer_games']==872+p['fallback_games'],f'round {n} fallback preserves planned exposure')
 host.update(counts);fallback+=p['fallback_games'];all_games.extend(games)
 round_summary.append({'round':n,'laptop':p['laptop_games'],'raytracer':p['raytracer_games'],'fallback':p['fallback_games'],'collection_seconds':r['collection_seconds'],'update_seconds':r['update_seconds']})
check(len(all_games)==len({g['index'] for g in all_games})==len({g['seed'] for g in all_games})==40000,'40000 unique committed schedules')
check(report['training_games_completed']==40000 and report['optimization_games_completed']==32000 and report['validation_games_completed']==8000,'40000 collected, 32000 optimized and 8000 held out')
# Reconcile the immutable checkpoint files against the original automatic backup manifests.
catalog={}
for backup in supervisor['backups']:
 if backup.get('status')!='verified':continue
 path=pathlib.Path(backup['snapshot'])/'backup-manifest.json'
 if not path.exists():continue
 for item in read(path)['members']:
  if item['path'].startswith('training/checkpoints/round-'):
   if item['path'] in catalog:check(catalog[item['path']]==item['sha256'],'same checkpoint hash across snapshots')
   catalog[item['path']]=item['sha256']
checkpoint_hashes={};previous=(-1,-1)
for n in range(41):
 d=C/f'training/checkpoints/round-{n:04d}';state=read(d/'state.json')
 check(state['round']==n and state['trainer_rng_state']['bit_generator']=='PCG64',f'checkpoint {n} round and RNG preserved')
 check(state['optimizer_steps']>=previous[0] and state['critic_optimizer_steps']>=previous[1],f'checkpoint {n} optimizer counters monotone');previous=state['optimizer_steps'],state['critic_optimizer_steps']
 for name in ('policy.keras','policy.manifest.json','optimizer.npz','critic-optimizer.npz','state.json'):
  p=d/name;key=p.relative_to(C).as_posix();checkpoint_hashes[key]=sha(p)
  check(catalog.get(key)==checkpoint_hashes[key],f'checkpoint {n} {name} matches captured backup')
 for filename,key in [('optimizer.npz','optimizer_steps'),('critic-optimizer.npz','critic_optimizer_steps')]:
  with np.load(d/filename,allow_pickle=False) as a:
   check(int(a['v0'])==state[key] and all(np.all(np.isfinite(a[k])) for k in a.files),f'checkpoint {n} {filename} finite and state counter matches')
 with zipfile.ZipFile(d/'policy.keras') as z:check(z.testzip() is None,f'checkpoint {n} model ZIP CRC')
source=read(B/'runtime-source-manifest.json')
for name,digest in source['files'].items():check(sha(B/'runtime'/name)==digest,'frozen local source '+name)
# Directly recount the fresh confirmation data without importing the study's comparison code.
labels=('imitation','round-4','round-40');evaluations={};configs=[];identities=[];summaries={}
for label in labels:
 d=read(C/f'confirmation-{label}.json');games=sorted(d['games'],key=lambda g:g['index']);evaluations[label]=games
 check(d['status']=='complete' and len(games)==5000,'complete final '+label)
 check([g['index'] for g in games]==list(range(5000)) and [g['seed'] for g in games]==list(range(330261025,330266025)),'fresh final scheduled IDs/seeds '+label)
 check(all(g['status'] in ('win','loss') and not g['trajectories'] for g in games),'terminal evaluation only, no training trajectories '+label)
 check(collections.Counter((g['seat'],g['starting_player']) for g in games)=={(a,b):1250 for a in (0,1) for b in ('P0','P1')},'balanced final seat/start '+label)
 check(not {g['seed'] for g in games}&{g['seed'] for g in all_games},'final/training seeds disjoint '+label)
 wins=sum(g['status']=='win' for g in games);check(wins==d['summary']['wins'] and d['summary']['errors']==d['summary']['cutoffs']==0,'recount final wins/denominators '+label)
 model=(B/'runtime/initialization/imitation/policy.keras') if label=='imitation' else C/f'training/checkpoints/round-{int(label.split("-")[1]):04d}/policy.keras'
 check(sha(model)==d['checkpoint_sha256'],'final checkpoint identity '+label)
 identities.append([(g['index'],g['seed'],g['seat'],g['starting_player'],g['mode']) for g in games]);configs.append(d['config'])
 summaries[label]={'wins':wins,'attempts':5000,'rate':wins/5000,'checkpoint_sha256':d['checkpoint_sha256']}
check(identities[0]==identities[1]==identities[2] and configs[0]==configs[1]==configs[2],'all final policies share exact scheduled cohort/config')
comparisons={}
for key,a,b in [('primary-40k-minus-4k','round-4','round-40'),('40k-minus-imitation','imitation','round-40'),('4k-minus-imitation','imitation','round-4')]:
 diffs=[int(y['status']=='win')-int(x['status']=='win') for x,y in zip(evaluations[a],evaluations[b])];saved=study['paired_comparisons'][key];counts=collections.Counter(diffs);delta=sum(diffs)/5000
 check(delta==saved['recorded_attempt_win_rate_change'] and counts[1]==saved['improved_recorded_win_attempts'] and counts[-1]==saved['worsened_recorded_win_attempts'],'independent paired recount '+key)
 rng=random.Random(9901);boot=sorted(sum(diffs[rng.randrange(5000)] for _ in range(5000))/5000 for _ in range(4000));ci=np.quantile(boot,[.025,.975]).tolist()
 check(np.allclose(ci,saved['recorded_attempt_win_rate_change_bootstrap_95_interval'],atol=1e-14,rtol=0),'independently reproduce prespecified paired bootstrap '+key)
 comparisons[key]={'difference':delta,'ci95':ci,'improved':counts[1],'worsened':counts[-1],'tied':counts[0],'unit':'matched_schedule','resamples':4000,'seed':9901}
last=supervisor['last_verified_backup'];check(sha(pathlib.Path(last['archive']))==last['archive_sha256'],'final automatic archive SHA')
check(last['terminal_cleanup_verified'] and supervisor['cleanup']['owned_children_remaining']==0 and not supervisor['cleanup']['errors'],'automatic final cleanup verified')
result={'status':'passed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'Separate read-only script; no learner, model execution, games, or imported study comparison routine.','checks_passed':len(checks),'checks':checks,'confirmation':summaries,'paired_comparisons':comparisons,'training':{'unique_committed_games':40000,'optimization_games':32000,'heldout_games':8000,'hosts':dict(host),'fallback_games':fallback,'rounds':round_summary,'checkpoint_files':checkpoint_hashes,'source_files_checked':len(source['files']),'last_actor_steps':previous[0],'last_critic_steps':previous[1]},'source_report_sha256':sha(C/'training/report.json'),'study_sha256':sha(C/'study.json'),'archive':last,'limitations':study['limitations']+['Physical executions exceed unique committed attempts due interrupted round 8 recollection and late/fallback work.','40k vs 4k compares two checkpoints from one trajectory with more rounds and optimizer steps; it does not isolate larger dataset size from additional optimization.','Comparisons are each policy versus the same baseline, not 40k-versus-4k head-to-head.']}
(B/'final-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'checks':len(checks),'confirmation':summaries,'primary':comparisons['primary-40k-minus-4k'],'hosts':dict(host),'fallback_games':fallback}))
