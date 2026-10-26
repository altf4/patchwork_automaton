"""Continue the exact completed100k RustMage PPO trajectory to140k attempts."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import sys
import time

import numpy as np

from . import run_rustmage_continuation as safe

FILES = safe.FILES
SOURCE_ROUND, TARGET_ROUND, GAMES_PER_ROUND = 100, 140, 1000
TRAINING_SEED = 300261025
MILESTONES = (101, 110, 120, 130, 140)
MONITORING_ROUNDS = (110, 120, 130, 140)
HEAD_TO_HEAD_NAMES = ('imitation', 'round-10', 'round-20', 'round-40',
                      'round-60', 'round-80', 'round-100')
SAFE_HELPER_SHA256 = 'd57b0351f68543ae4e1a34c897ec7b0a30f909a7e738f2355b23b66624dcf5a7'
SOURCE_ACTOR_STEPS, SOURCE_CRITIC_STEPS = 32822, 15857
RUNTIME_OPTIONS = dict(learner_gru_backend='cudnn-memory', update_pipeline='gpu',
                       policy_cache_mib=8192, actor_epoch_cache_mib=8192)
LEARNING_CONFIG = dict(games_per_round=1000, train_mode='versus_baseline',
    validation_fraction=.2, eval_games=200, eval_every_games=0, eval_horizon=25,
    train_horizon=60, workers=48, inference_processes=4, inference_wait_ms=0.,
    collector='processes', pooled_collector_config=None, observation_pipeline='python',
    architecture='separate', learner_device='gpu', inference_device='cpu',
    validation_device='cpu', validation_batch=32, baseline_eval_games=0,
    baseline_eval_seed=48261024, train_batch=512, learning_rate=.0001, ppo_epochs=2,
    entropy_weight=.01, advantage_baseline='critic', target_kl=.02, max_state_kl=.5,
    guard_audit=False, value_learning_rate=.0003, value_max_steps=157,
    critic_update_scope='all', critic_unroll=1, critic_warmup_games=1000,
    critic_warmup_max_steps=157, concurrent_updates=True, prefetch_batches=2,
    measurement_workers=4, rollout_cache_mib=512, rollout_storage='pickle',
    seed=TRAINING_SEED, algorithm='ppo', advantage_scale_floor=1., ppo_clip=.2)

# These helpers do not depend on the old source/target round constants. Reuse
# their reviewed implementations without altering any global or live source.
utc, read, publish = safe.utc, safe.read, safe.publish
verify_sources, verify_checkpoint = safe.verify_sources, safe.verify_checkpoint
process_identity, verify_terminal_owned_sessions = safe.process_identity, safe.verify_terminal_owned_sessions
run_child, native_guard_monitor = safe.run_child, safe.native_guard_monitor
publish_native_terminal_receipt = safe.publish_native_terminal_receipt


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_config(config):
    require(config == LEARNING_CONFIG and all(type(config.get(k)) is type(v)
        for k,v in LEARNING_CONFIG.items()), 'Exact100k learning recipe differs')
    return deepcopy(config)


def validate_manifest(manifest):
    expected = dict(source_round=100, target_round=140, target_games=140000,
        additional_training_games=40000, training_seed=TRAINING_SEED,
        milestones=list(MILESTONES), monitoring_rounds=list(MONITORING_ROUNDS),
        monitoring_seed=320261025, monitoring_games=2000,
        confirmation_seed=520261026, confirmation_games=10000,
        head_to_head_seed=530261026, head_to_head_games=5000,
        preflight_head_to_head_seed=540261026, preflight_head_to_head_games=8,
        head_to_head_opponents=list(HEAD_TO_HEAD_NAMES), recovery_epoch=1)
    for key,value in expected.items():
        require(manifest.get(key) == value and type(manifest.get(key)) is type(value),
            'Finite140k continuation design differs: '+key)
    for key in ('run_id','python','runtime','output','source_training','shared_lock',
                'native_liveness','native_terminal_receipt','imitation_checkpoint'):
        require(isinstance(manifest.get(key),str) and bool(manifest[key]), 'Missing manifest field: '+key)
    for key in ('python','runtime','output','source_training','shared_lock',
                'native_liveness','native_terminal_receipt','imitation_checkpoint'):
        require(Path(manifest[key]).is_absolute(), 'Absolute path required: '+key)
    source,output = Path(manifest['source_training']).resolve(),Path(manifest['output']).resolve()
    require(source != output and not source.is_relative_to(output) and not output.is_relative_to(source),
        'New output must be separate from immutable100k source')
    require(manifest.get('hard_deadline_utc') is None, 'No elapsed-time stop is authorized')
    require(manifest.get('runtime_options') == RUNTIME_OPTIONS
        and all(type(manifest['runtime_options'][k]) is type(v) for k,v in RUNTIME_OPTIONS.items()),
        'Validated fused/cache runtime profile differs')
    for key in ('source_sha256','source_artifacts_sha256'):
        require(isinstance(manifest.get(key),dict) and bool(manifest[key]), 'Missing source pins: '+key)
    require(manifest['source_sha256'].get('examples/run_rustmage_continuation.py') == SAFE_HELPER_SHA256
        and sha(safe.__file__) == SAFE_HELPER_SHA256, 'Unchanged lifecycle helper source pin differs')
    require(manifest['source_sha256'].get('examples/run_rustmage_continuation_140k.py') == sha(__file__),
        'New coordinator source pin differs')
    needed = [source/'report.json',source/'checkpoints/latest.json']
    needed += [source/'checkpoints'/f'round-{i:04d}'/name for i in range(101) for name in FILES]
    needed += [Path(manifest['imitation_checkpoint']),Path(manifest['imitation_checkpoint']).with_suffix('.manifest.json')]
    require(all(str(p) in manifest['source_artifacts_sha256'] for p in needed),
        'All101 source checkpoints/report/pointer/imitation must be pinned')
    require(manifest.get('recovery_coordinator_sha256') == sha(__file__), 'New terminal coordinator source pin differs')
    native = manifest.get('expected_native_owner')
    require(isinstance(native,dict) and set(native)=={'windows_pid','creation_utc'}
        and type(native['windows_pid']) is int and native['windows_pid']>0
        and isinstance(native['creation_utc'],str), 'Exact native owner required')
    return manifest


def design(manifest):
    keys = ('run_id','source_round','target_round','target_games','additional_training_games',
        'training_seed','milestones','monitoring_rounds','monitoring_seed','monitoring_games',
        'confirmation_seed','confirmation_games','head_to_head_seed','head_to_head_games',
        'preflight_head_to_head_seed','preflight_head_to_head_games','head_to_head_opponents','runtime_options')
    return {key:deepcopy(manifest[key]) for key in keys}


def _check_historical_copies(source, training):
    for n in range(101):
        source_cp,copy_cp = source/'checkpoints'/f'round-{n:04d}',training/'checkpoints'/f'round-{n:04d}'
        require({p.name for p in copy_cp.iterdir()}==set(FILES),'Historical checkpoint is partial')
        for name in FILES:
            require(sha(source_cp/name)==sha(copy_cp/name),'Historical checkpoint bytes differ: '+str(copy_cp/name))


def _report_snapshot(training, completed):
    report = read(training/'report.json')
    rows = report['rounds']
    require([r['round'] for r in rows]==list(range(1,len(rows)+1)), 'Missing or reordered training history')
    require(len(rows) in (completed,completed-1), 'Report is not at its latest checkpoint or recoverably one behind')
    if len(rows)==completed-1:
        state = read(training/'checkpoints'/f'round-{completed:04d}'/'state.json')
        require(state.get('round_result',{}).get('round')==completed,'Latest checkpoint cannot recover report')
        report = deepcopy(report)
        report['rounds'].append(state['round_result'])
    return report


def _reject_uncommitted_rollouts(training, completed):
    for path in (training/'rollouts').glob('game-*'):
        match = re.fullmatch(r'game-(\d+)-p[01]\.pkl\.gz',path.name)
        if match and int(match[1]) >= completed*GAMES_PER_ROUND:
            raise ValueError('Uncommitted rollouts require preserved explicit repair before resume: '+str(path))


def prepare(manifest):
    """Copy source bytes exactly; add provenance outside its learned state/report."""
    source,training = Path(manifest['source_training']),Path(manifest['output'])/'training'
    verify_sources(manifest)
    report = read(source/'report.json')
    config = validate_config(report['config'])
    require(report.get('status')=='complete' and [r['round'] for r in report['rounds']]==list(range(1,101))
        and report.get('critic_warmup',{}).get('status')=='complete'
        and read(source/'checkpoints/latest.json')=={'checkpoint':'round-0100'},
        'Source must be the complete once-warmed100k trajectory')
    state = read(source/'checkpoints/round-0100/state.json')
    require(state['config']==config and state['optimizer_steps']==SOURCE_ACTOR_STEPS
        and state['critic_optimizer_steps']==SOURCE_CRITIC_STEPS,'Exact100k boundary counters/config differ')
    source_digest = sha(source/'report.json')
    fork_path = training/'continuation-fork-140k.json'
    staging = training.with_name('training.prepare')
    require(not staging.exists(),'Partial preparation is preserved; explicit repair required')
    if training.exists():
        fork = read(fork_path)
        require(fork.get('design')==design(manifest) and fork.get('source_report_sha256')==source_digest
            and fork.get('source_training')==str(source) and fork.get('learning_config')==config,
            'Existing continuation fork belongs to another source/design')
        _check_historical_copies(source,training)
        latest = read(training/'checkpoints/latest.json').get('checkpoint')
        require(isinstance(latest,str) and re.fullmatch(r'round-\d{4}',latest) is not None,
            'Invalid existing continuation checkpoint pointer')
        completed = int(latest.removeprefix('round-'))
        require(100<=completed<=140,'Existing round outside finite140k plan')
        snapshot = _report_snapshot(training,completed)
        require(snapshot['config']==config and snapshot['rounds'][:100]==report['rounds'],
            'Existing learning configuration or historical report prefix differs')
        verify_checkpoint(training/'checkpoints'/latest,completed)
        _reject_uncommitted_rollouts(training,completed)
        return training,config,fork
    staging.mkdir(parents=True)
    for n in range(101):
        original = source/'checkpoints'/f'round-{n:04d}'
        verify_checkpoint(original,n)
        require(read(original/'state.json')['config']==config,'Historical learning configuration differs')
        shutil.copytree(original,staging/'checkpoints'/original.name)
    shutil.copyfile(source/'checkpoints/latest.json',staging/'checkpoints/latest.json')
    shutil.copyfile(source/'report.json',staging/'report.json')
    _check_historical_copies(source,staging)
    require(sha(staging/'report.json')==source_digest,'Copied source report bytes differ')
    fork = dict(schema_version=1,created_utc=utc(),source_training=str(source),source_report_sha256=source_digest,
        design=design(manifest),learning_config=config,changed_config_fields=[],
        source_checkpoint_sha256=verify_checkpoint(source/'checkpoints/round-0100',100),
        preservation=dict(all101_checkpoint_files_byte_identical=True,source_report_byte_identical=True,
            model_and_manifest_bytes=True,both_adam_bytes=True,all_rng_states=True,
            all_optimizer_counters=True,no_critic_rewarm=True,no_config_mutation=True),
        topology=dict(before='Raytracer-only ordinary collector',after='Raytracer-only ordinary collector',
            laptop_reserved_games=0,laptop_producer=False,fallback_wait_seconds=0))
    publish(staging/'continuation-fork-140k.json',fork)
    verify_sources(manifest)
    safe._fsync_tree(staging)
    staging.rename(training)
    descriptor = os.open(training.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    publish(training.parent/'fork.json',fork)
    return training,config,fork


def train_command(python, training, config, rounds, runtime_options):
    require(type(rounds) is int and rounds in MILESTONES,'Unregistered finite140k target')
    validate_config(config)
    require(runtime_options==RUNTIME_OPTIONS,'Unvalidated runtime profile')
    command = [python,'-m','examples.train_rustmage','--output',str(training),'--rounds',str(rounds),'--resume']
    for key,value in {**config,**runtime_options}.items():
        if key in ('algorithm','advantage_scale_floor','ppo_clip') or value is None:
            continue
        flag = '--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:command.append(flag)
        else:command.extend((flag,str(value)))
    return command


def validate_training(training, rounds):
    training = Path(training)
    require(type(rounds) is int and 100<=rounds<=140,'Training milestone outside100k to140k')
    latest = read(training/'checkpoints/latest.json')['checkpoint']
    completed = int(latest.removeprefix('round-'))
    require(rounds<=completed<=140,'Checkpoint is behind or outside finite budget')
    report = _report_snapshot(training,completed)
    config = validate_config(report['config'])
    boundary = read(training/'checkpoints/round-0100/state.json')
    require(boundary['optimizer_steps']==SOURCE_ACTOR_STEPS and boundary['critic_optimizer_steps']==SOURCE_CRITIC_STEPS,
        'Original100k boundary counters differ')
    actor,critic = SOURCE_ACTOR_STEPS,SOURCE_CRITIC_STEPS
    optimized,heldout,new_raytracer = 0,0,0
    previous_rng = boundary['trainer_rng_state']
    for row in report['rounds'][:rounds]:
        number,games = row['round'],row['games'];first=(number-1)*1000
        expected=set(range(first,first+1000))
        require(len(games)==1000 and {g['index'] for g in games}==expected,'Missing/duplicate scheduled training IDs')
        for game in games:
            index=game['index']
            require(game['seed']==TRAINING_SEED+1000+index and game['seat']==index%2
                and game['starting_player']==f'P{(index//2)%2}' and game['mode']=='versus_baseline'
                and game['status'] in ('win','loss','draw','cutoff','error'),'Training seed/seat/start/mode/outcome differs')
        split=row['split'];a,b=split['training_indices'],split['validation_indices']
        require(len(a)==len(set(a))==800 and len(b)==len(set(b))==200 and not set(a)&set(b)
            and set(a)|set(b)==expected,'Whole-game800/200 holdout differs')
        optimized+=split['training']['terminal_games'];heldout+=split['validation']['attempted_games']
        if number<=100:continue
        require(not row.get('collection_resources',{}).get('pool')
            and row.get('collection_resources',{}).get('laptop_games',0)==0
            and all(g.get('collection_host','raytracer')=='raytracer' and not g.get('pool_fallback') for g in games),
            'New continuation reserved laptop games or pooled fallback')
        update=row['update'];actor+=update['accepted_steps'];critic+=update['critic_steps']
        require(0<update['critic_steps']<=157 and update['accepted_steps']>=0
            and update['optimizer_steps']==actor and update['critic_optimizer_steps']==critic,
            'New Adam counter continuity or critic cap differs')
        current=read(training/'checkpoints'/f'round-{number:04d}'/'state.json')
        require(current['config']==config and current['optimizer_steps']==actor
            and current['critic_optimizer_steps']==critic,'New checkpoint configuration/counter continuity differs')
        count=update['samples'];batch_count=(count+511)//512
        require(count>0 and update['accepted_steps']%batch_count==0,'Incomplete accepted actor epoch')
        actor_epochs=update['accepted_steps']//batch_count+(1 if update['early_stop'] else 0)
        require(1<=actor_epochs<=2,'Actor epoch budget differs')
        probe=np.random.default_rng();probe.bit_generator.state=previous_rng
        for _ in range(2+actor_epochs):probe.permutation(count)
        require(probe.bit_generator.state==current['trainer_rng_state'],'Exact resumed PCG permutation continuity differs')
        previous_rng=current['trainer_rng_state'];new_raytracer+=1000
    checkpoint=training/'checkpoints'/f'round-{rounds:04d}'
    integrity=verify_checkpoint(checkpoint,rounds)
    state=read(checkpoint/'state.json')
    require(state['config']==config and state['optimizer_steps']==actor and state['critic_optimizer_steps']==critic,
        'Committed target checkpoint counters/config differ')
    return dict(games=rounds*1000,added_games=(rounds-100)*1000,optimization_games=optimized,heldout_games=heldout,
        continuation_raytracer_games=new_raytracer,continuation_laptop_games=0,
        actor_optimizer_steps=actor,critic_optimizer_steps=critic,checkpoint_sha256=integrity)


def validate_evaluation(result, *, checkpoint, seed, games, opponent=None):
    safe.validate_evaluation(result,checkpoint=checkpoint,seed=seed,games=games,opponent=opponent)
    from patchwork_automaton.neural.rustmage_evaluation import baseline_evaluation_config,match_summary
    expected=baseline_evaluation_config(seed_base=seed,games=games,max_own_turns=60,greedy=False)
    mode='versus_neural' if opponent is not None else 'versus_baseline'
    if opponent is None:
        require(result['config']==expected,'Baseline cohort/opponent configuration differs')
        summary=match_summary(result['games'],expected_games=games)
    else:
        collection=dict(expected['collection']);collection.pop('mode');collection.pop('output')
        expected_config=dict(collection=collection,cohort=expected['cohort'],mode='versus_neural',
            policy='sampled temperature 1',focal_seat='index % 2',starting_player='P((index // 2) % 2)',
            decks=["foundations_deck('red')","foundations_deck('red')"])
        require(result['config']==expected_config and result.get('training') is False,
            'Actual frozen neural-duel configuration differs')
        summary=match_summary([dict(row,mode='versus_baseline') for row in result['games']],expected_games=games)
        summary.update(mode=mode,metric='focal_recorded_wins_per_scheduled_attempt')
    for row in result['games']:
        if row['status'] in ('win','loss','draw'):
            winner=None if row['status']=='draw' else row['seat'] if row['status']=='win' else 1-row['seat']
            require(row.get('winner')==winner,'Frozen evaluation winner disagrees with focal outcome')
    require(summary==result['summary'],'Frozen evaluation full denominator/stratum summary differs')
    return result


def _new_log(output,label):
    return Path(output)/(label+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.log')


def evaluate(manifest, checkpoint, output, seed, games, *, opponent=None, workers=None, replicas_per_policy=None):
    checkpoint,output=Path(checkpoint),Path(output)
    before={str(p):sha(p) for cp in [checkpoint]+([Path(opponent)] if opponent is not None else [])
        for p in (cp,cp.with_suffix('.manifest.json'))}
    if not output.exists():
        module='examples.evaluate_rustmage_head_to_head' if opponent is not None else 'examples.evaluate_rustmage_policy'
        command=[manifest['python'],'-m',module,'--checkpoint',str(checkpoint),'--output',str(output),
            '--seed',str(seed),'--games',str(games)]
        if opponent is not None:
            command+=['--opponent-checkpoint',str(opponent)]
            if workers is not None:command+=['--workers',str(workers)]
            if replicas_per_policy is not None:command+=['--replicas-per-policy',str(replicas_per_policy)]
        run_child(command,_new_log(output.parent,output.stem),manifest)
    require({name:sha(name) for name in before}==before,'Frozen evaluation changed source model/manifest bytes')
    return validate_evaluation(read(output),checkpoint=checkpoint,seed=seed,games=games,opponent=opponent)


def preflight(manifest,training):
    verify_sources(manifest)
    checkpoint=Path(training)/'checkpoints/round-0040/policy.keras'
    opponent=Path(training)/'checkpoints/round-0010/policy.keras'
    result=evaluate(manifest,checkpoint,Path(manifest['output'])/'preflight-head-to-head.json',
        manifest['preflight_head_to_head_seed'],8,opponent=opponent,workers=4,replicas_per_policy=1)
    require(result['summary']['terminal_games']==8 and result['summary']['errors']==result['summary']['cutoffs']==0
        and result.get('game_engines')==4 and result.get('replicas_per_policy')==1
        and result.get('inference_device')=='cpu' and result.get('training') is False,'Eight-game smoke runtime/terminal gate failed')
    expected={str(p.resolve()):sha(p) for cp in (checkpoint,opponent) for p in (cp,cp.with_suffix('.manifest.json'))}
    require(result.get('source_files_sha256')==expected,'Smoke model/manifest hashes differ')
    for role in ('focal','opponent'):
        stats=result.get('batching',{}).get(role,{})
        require(stats.get('requests',0)>0 and stats.get('device')=='/CPU:0' and len(stats.get('replicas',[]))==1
            and all(not r.get('visible_gpus') for r in stats['replicas']),'Both smoke roles must use GPU-hidden CPU inference')
    verify_sources(manifest)
    return dict(status='passed',games=8,seed=manifest['preflight_head_to_head_seed'],summary=result['summary'],
        source_files_sha256=expected,both_policy_inference_observed=True,sources_unchanged=True)


def terminal_receipt_context(manifest):
    """Pin this new coordinator, not safe.__file__ from the old trajectory."""
    receipt=Path(manifest['native_terminal_receipt'])
    require(receipt.is_absolute() and not receipt.exists(),'A new absolute terminal receipt path is required')
    require(manifest.get('recovery_coordinator_sha256')==sha(__file__),'New coordinator source pin differs')
    owner=process_identity(os.getpid())
    require(owner and owner['pid']==owner['pgid']==owner['sid'] and owner['uid']==os.getuid(),
        'Coordinator must own a dedicated session')
    guard=native_guard_monitor(manifest);guard.require_fresh()
    native=manifest.get('_native_owner')
    require(native and len(native)==2,'Exact bound native owner is required')
    cleanup=Path(manifest['output'])/'cleanup.json'
    return dict(receipt=str(receipt),coordinator=owner,recovery_epoch=1,
        recovery_coordinator_sha256=manifest['recovery_coordinator_sha256'],started_utc=utc(),
        prior_cleanup_sha256=sha(cleanup) if cleanup.exists() else None,
        native_owner=dict(windows_pid=native[0],creation_utc=native[1]),
        native_manifest_sha256=guard.last_probe['expected_manifest_sha256'])


def run(manifest):
    validate_manifest(manifest)
    output=Path(manifest['output']);output.mkdir(parents=True,exist_ok=True)
    os.chdir(manifest['runtime']);verify_sources(manifest)
    lock=Path(manifest['shared_lock']).open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        lock.close();raise
    owner=process_identity(os.getpid());study_path=output/'study.json'
    study=read(study_path) if study_path.exists() else dict(status='starting',run_id=manifest['run_id'],
        started_utc=utc(),source_round=100,target_round=140,target_games=140000,added_games=40000,
        training_seed=TRAINING_SEED,monitoring_seed=manifest['monitoring_seed'],
        confirmation_seed=manifest['confirmation_seed'],head_to_head_seed=manifest['head_to_head_seed'],
        preregistered_design=design(manifest),milestones={},
        design=dict(continuation='Exact round100 model/both Adam/PCG; no config change or critic rewarm',
            topology='All-Raytracer ordinary collector; no laptop producer/reserve/fallback wait',
            primary='140k-minus-100k on fresh paired active-baseline cohort',
            head_to_head='Actual frozen140k against seven earlier policies',endpoint_fixed=True,
            extra_training_arms=False,elapsed_time_hard_stop=False,additional_training_games=40000,
            additional_physical_executions_without_retry=104008,child_plan_count=19,
            monitoring_games_each=2000,confirmation_games_each=10000,head_to_head_games_each=5000,
            passive_games_each_training_segment=200,evaluation_games_are_additional=True))
    try:
        require(study.get('preregistered_design')==design(manifest),'Existing study has another finite design')
        training,config,fork=prepare(manifest)
        study.update(status='running',training_output=str(training),fork=fork,runtime_options=manifest['runtime_options'])
        publish(study_path,study)
        study['head_to_head_preflight']=preflight(manifest,training);publish(study_path,study)
        for rounds in MILESTONES:
            verify_sources(manifest)
            latest=read(training/'checkpoints/latest.json')['checkpoint'];completed=int(latest.removeprefix('round-'))
            report=read(training/'report.json')
            report_status,reported=report.get('status'),report['rounds'][-1]['round']
            del report  # Never retain a full100–140-round JSON while a learner runs.
            if completed<rounds or (completed==rounds and (report_status!='complete' or reported<rounds)):
                run_child(train_command(manifest['python'],training,config,rounds,manifest['runtime_options']),
                    _new_log(output,f'train-through-{rounds:04d}'),manifest)
                require(read(training/'report.json')['status']=='complete','Trainer did not finish its finite target')
            progress=validate_training(training,rounds)
            study['milestones'][str(rounds)]=dict(completed_utc=utc(),**progress);publish(study_path,study)
            if rounds in MONITORING_ROUNDS:
                result=evaluate(manifest,training/'checkpoints'/f'round-{rounds:04d}/policy.keras',
                    output/f'monitor-{rounds:04d}.json',manifest['monitoring_seed'],2000)
                study['milestones'][str(rounds)]['monitoring_summary']=result['summary'];publish(study_path,study)
        final=validate_training(training,140);study['status']='final_evaluation';publish(study_path,study)
        policies={f'round-{n}':training/'checkpoints'/f'round-{n:04d}/policy.keras' for n in (100,140)}
        confirmations={name:evaluate(manifest,cp,output/f'confirmation-{name}.json',manifest['confirmation_seed'],10000)
            for name,cp in policies.items()}
        from patchwork_automaton.neural.rustmage_evaluation import paired_match_improvement
        paired=paired_match_improvement(confirmations['round-100']['games'],confirmations['round-140']['games'])
        duels={}
        for name in HEAD_TO_HEAD_NAMES:
            opponent=Path(manifest['imitation_checkpoint']) if name=='imitation' else training/'checkpoints'/f"round-{int(name.removeprefix('round-')):04d}/policy.keras"
            result=evaluate(manifest,policies['round-140'],output/f'head-to-head-{name}.json',
                manifest['head_to_head_seed'],5000,opponent=opponent)
            duels[name]=dict(summary=result['summary'],checkpoint_sha256=result['checkpoint_sha256'],
                opponent_checkpoint_sha256=result['opponent_checkpoint_sha256'],seed=manifest['head_to_head_seed'],games=5000)
            study['head_to_head']=duels;publish(study_path,study)
        verify_sources(manifest)
        study.update(status='complete',finished_utc=utc(),final=final,
            confirmation_summaries={k:v['summary'] for k,v in confirmations.items()},
            paired_comparisons={'primary-140k-minus-100k':paired},head_to_head=duels,
            limitations=['One continued training seed; game intervals do not measure training-seed variability.',
                'More data also adds optimizer steps; their causal contributions are not isolated.',
                'A fixed mono-red matchup and earlier related policy opponents limit generalization.',
                'Shared monitoring/duel schedules are not independent extra samples; protocols are distinct.'])
        publish(study_path,study)
    except BaseException as error:
        study.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',
            last_error=dict(type=type(error).__name__,message=str(error),utc=utc()))
        publish(study_path,study);raise
    finally:
        try:
            checked=verify_terminal_owned_sessions(manifest,owner);verify_sources(manifest);os.sync()
        except BaseException as error:
            try:
                publish(output/'cleanup.json',dict(finished_utc=utc(),coordinator_pid=os.getpid(),
                    status=study['status'],shared_lock_released=False,cleanup_error=str(error)))
            finally:
                # No successful cleanup/receipt is claimed. Closing the failing
                # process's descriptor avoids a leak; OS exit would release it.
                lock.close()
            raise
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        publish(output/'cleanup.json',dict(finished_utc=utc(),coordinator_pid=os.getpid(),
            shared_lock_released=True,status=study['status'],owned_children_remaining=0,
            owned_process_records_checked=len(checked)))


def run_with_terminal_receipt(manifest):
    validate_manifest(manifest)
    context=terminal_receipt_context(manifest);failure=None
    try:return run(manifest)
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error));raise
    finally:
        try:publish_native_terminal_receipt(manifest,context,failure)
        except BaseException as error:
            print(json.dumps(dict(event='native-terminal-receipt-not-published',utc=utc(),
                run_id=manifest.get('run_id'),coordinator=context['coordinator'],
                error_type=type(error).__name__,error=str(error),original_run_error=failure),sort_keys=True),
                file=sys.stderr,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    run_with_terminal_receipt(read(args.manifest))
