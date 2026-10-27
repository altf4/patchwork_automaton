"""Adopt27k durable epoch2 outcomes and finish only the six missing frozen duels.

The interrupted round10 duel completed5000 physical games without a durable
result. Its exact scheduled IDs are replayed once, disclosed as duplicate
physical work, and never counted as an extra independent cohort.
"""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import signal
import sys
import json

from examples import run_rustmage_140k_evaluation_recovery as previous

phase,safe = previous.phase,previous.safe
sha,read,publish = previous.sha,previous.read,previous.publish
verify_sources,source_snapshot = previous.verify_sources,previous.source_snapshot
process_identity,verify_terminal_owned_sessions = previous.process_identity,previous.verify_terminal_owned_sessions
publish_native_terminal_receipt = previous.publish_native_terminal_receipt
RUN_ID='raytracer-continuation-140000-evaluation-recovery3-2026-10-27'
PRIOR_RUN_ID=previous.RUN_ID
PARENT_RUN_ID=previous.PARENT_RUN_ID
PREVIOUS_SHA256='d5bd25a325b195384e40bf4f0ac48617ccc2b68c6281f30e82281aa68c7cc4fb'
REPLAY_LOG_SHA256='23e41164329ae66c5e52efce72743d7481dcb1b1fa05062928765a7ad2f0f8c6'
ADOPTED=('monitor-0140','confirmation-round-100','confirmation-round-140','head-to-head-imitation')
NEW_OPPONENTS=('round-10','round-20','round-40','round-60','round-80','round-100')


def require(ok,message):
    if not ok:raise ValueError(message)


def evaluation_specs(manifest):
    root=Path(manifest['source_training'])/'checkpoints'
    policy=lambda n:root/f'round-{n:04d}'/'policy.keras'
    specs={'monitor-0140':(policy(140),320261025,2000,None),
        'confirmation-round-100':(policy(100),520261026,10000,None),
        'confirmation-round-140':(policy(140),520261026,10000,None),
        'head-to-head-imitation':(policy(140),530261026,5000,Path(manifest['imitation_checkpoint']))}
    specs.update({f'head-to-head-{name}':(policy(140),530261026,5000,policy(int(name.removeprefix('round-'))))
        for name in NEW_OPPONENTS})
    return specs


def validate_manifest(manifest):
    expected=dict(run_id=RUN_ID,parent_run_id=PARENT_RUN_ID,prior_recovery_run_id=PRIOR_RUN_ID,recovery_epoch=3,
        source_round=100,target_round=140,target_games=140000,additional_training_games=0,training=False,
        monitoring_seed=320261025,monitoring_games=2000,confirmation_seed=520261026,confirmation_games=10000,
        head_to_head_seed=530261026,head_to_head_games=5000,head_to_head_opponents=list(phase.HEAD_TO_HEAD_NAMES),
        registered_evaluation_attempts=57000,adopted_evaluation_attempts=27000,remaining_evaluation_attempts=30000,
        owned_child_plan=6)
    for key,value in expected.items():
        require(manifest.get(key)==value and type(manifest.get(key))is type(value),'Finite epoch3 evaluation-only design differs:'+key)
    for key in ('python','runtime','output','source_training','original_output','shared_lock','imitation_checkpoint',
        'prior_recovery_output','prior_recovery_manifest','prior_recovery_launch_record','prior_recovery_terminal_receipt',
        'native_liveness','native_terminal_receipt'):
        require(isinstance(manifest.get(key),str)and Path(manifest[key]).is_absolute(),'Absolute recovery path required:'+key)
    output=Path(manifest['output']).resolve()
    for key in ('original_output','prior_recovery_output'):
        source=Path(manifest[key]).resolve()
        require(source!=output and not source.is_relative_to(output)and not output.is_relative_to(source),
            'New epoch3 output must be separate from all preserved outputs')
    require(Path(manifest['source_training']).resolve()==Path(manifest['original_output']).resolve()/'training',
        'Saved140k training source differs')
    require(manifest.get('hard_deadline_utc')is None,'No elapsed-time stop authorized')
    require(manifest.get('recovery_coordinator_sha256')==sha(__file__),'New epoch3 coordinator pin differs')
    require(sha(previous.__file__)==PREVIOUS_SHA256 and sha(phase.__file__)==previous.PHASE_SHA256
        and sha(safe.__file__)==previous.SAFE_SHA256,'Reviewed previous helper bytes changed')
    pins=manifest.get('source_sha256',{})
    require(len(pins)==153 and pins.get('examples/run_rustmage_140k_evaluation_recovery3.py')==sha(__file__)
        and pins.get('examples/run_rustmage_140k_evaluation_recovery.py')==PREVIOUS_SHA256,
        'Frozen153-file runtime with new coordinator required')
    artifacts=manifest.get('source_artifacts_sha256',{})
    training=Path(manifest['source_training']);prior=Path(manifest['prior_recovery_output'])
    needed=[training/'report.json',training/'checkpoints/latest.json',prior/'report.json',prior/'cleanup.json']
    needed += [training/'checkpoints'/f'round-{n:04d}'/name for n in range(141)for name in safe.FILES]
    needed += [Path(manifest[k])for k in ('prior_recovery_manifest','prior_recovery_launch_record','prior_recovery_terminal_receipt')]
    imitation=Path(manifest['imitation_checkpoint']);needed += [imitation,imitation.with_suffix('.manifest.json')]
    require(all(str(p)in artifacts for p in needed),'All705 checkpoints and exact paused-run provenance must be pinned')
    adopted=manifest.get('adopted_results',{})
    require(set(adopted)==set(ADOPTED),'Exactly the four durable27k epoch2 results must be adopted')
    specs=evaluation_specs(manifest)
    for stem,proof in adopted.items():
        path=prior/(stem+'.json');cp,seed,games,opponent=specs[stem]
        require(proof.get('path')==str(path)and artifacts.get(str(path))==proof.get('sha256')
            and proof.get('seed')==seed and proof.get('attempts')==games
            and proof.get('checkpoint_sha256')==artifacts.get(str(cp))
            and (opponent is None or proof.get('opponent_checkpoint_sha256')==artifacts.get(str(opponent))),
            'Exact adoption policy/cohort/path pin differs:'+stem)
    replay=manifest.get('replay_provenance',{})
    for key,value in dict(result_stem='head-to-head-round-10',seed=530261026,index_base=0,attempts=5000,
        completed_physical_attempts=5000,durable_result_present=False,same_ids_replayed=True,
        counts_as_independent_extra_sample=False).items():
        require(replay.get(key)==value and type(replay.get(key))is type(value),'Interrupted same-ID replay provenance differs:'+key)
    require(isinstance(replay.get('log_path'),str)and Path(replay['log_path']).is_absolute()
        and Path(replay['log_path']).parent==prior and replay.get('log_sha256')==REPLAY_LOG_SHA256
        and artifacts.get(replay['log_path'])==REPLAY_LOG_SHA256,'Preserved5000-game interrupted log pin differs')
    native=manifest.get('expected_native_owner')
    require(isinstance(native,dict)and set(native)=={'windows_pid','creation_utc'}
        and type(native['windows_pid'])is int and native['windows_pid']>0 and isinstance(native['creation_utc'],str),
        'Exact epoch3 native owner required')
    require(isinstance(manifest.get('native_manifest_sha256'),str)and len(manifest['native_manifest_sha256'])==64
        and isinstance(manifest.get('source140_rng_sha256'),str)and len(manifest['source140_rng_sha256'])==64,
        'Native manifest and exact140k PCG digest required')
    return manifest


def design(manifest):
    return {key:deepcopy(manifest[key])for key in ('run_id','parent_run_id','prior_recovery_run_id','recovery_epoch',
        'source_round','target_round','target_games','additional_training_games','training','monitoring_seed','monitoring_games',
        'confirmation_seed','confirmation_games','head_to_head_seed','head_to_head_games','head_to_head_opponents',
        'registered_evaluation_attempts','adopted_evaluation_attempts','remaining_evaluation_attempts','owned_child_plan',
        'adopted_results','replay_provenance')}


def verify_prior_stopped(manifest):
    prior=Path(manifest['prior_recovery_output']);report=read(prior/'report.json');cleanup=read(prior/'cleanup.json')
    old=read(manifest['prior_recovery_manifest']);launch=read(manifest['prior_recovery_launch_record'])
    receipt=read(manifest['prior_recovery_terminal_receipt']);owner=launch.get('owner')
    require(isinstance(owner,dict)and all(type(owner.get(k))is int for k in ('pid','pgid','sid','start_ticks','uid'))
        and owner['pid']==owner['pgid']==owner['sid']and owner['uid']==os.getuid(),'Exact prior Linux owner required')
    require(process_identity(owner['pid'])!=owner,'Prior coordinator remains live')
    require(report.get('run_id')==PRIOR_RUN_ID and report.get('recovery_epoch')==2 and report.get('status')=='interrupted'
        and set(report.get('completed_evaluations',{}))==set(ADOPTED),'Paused epoch2 report/status/inventory differs')
    require(cleanup.get('coordinator_pid')==owner['pid']and cleanup.get('status')=='interrupted'
        and cleanup.get('shared_lock_released')is True and cleanup.get('owned_children_remaining')==0
        and cleanup.get('owned_process_records_checked')==5
        and not cleanup.get('cleanup_error'),'Prior exact-owned cleanup unverified')
    runtime=dict(manifest['source_sha256']);runtime.pop('examples/run_rustmage_140k_evaluation_recovery3.py')
    require(old.get('run_id')==PRIOR_RUN_ID and old.get('recovery_epoch')==2
        and old.get('recovery_coordinator_sha256')==PREVIOUS_SHA256 and old.get('source_sha256')==runtime
        and old.get('output')==str(prior),'All152 paused runtime files must remain byte-identical')
    require(receipt.get('run_id')==PRIOR_RUN_ID and receipt.get('recovery_epoch')==2 and receipt.get('status')=='interrupted'
        and receipt.get('coordinator')==owner and receipt.get('recovery_coordinator_sha256')==PREVIOUS_SHA256
        and receipt.get('native_owner')==old['expected_native_owner']
        and receipt.get('native_manifest_sha256')==old['native_manifest_sha256']
        and receipt.get('cleanup_sha256')==sha(prior/'cleanup.json')
        and receipt.get('cleanup',{}).get('cleanup_verified')is True,'Exact paused epoch2 terminal receipt differs')
    checked=verify_terminal_owned_sessions(dict(manifest,output=str(prior)),owner)
    require(len(checked)==5,'All five paused epoch2 child sessions must be verified gone')
    for stem in ADOPTED:
        require(report['completed_evaluations'][stem]==manifest['adopted_results'][stem],
            'Adopted proof differs from paused report:'+stem)
    for name in NEW_OPPONENTS:
        path=prior/f'head-to-head-{name}.json'
        require(not path.exists()and not path.with_suffix('.json.tmp').exists(),
            'A durable/partial remaining result must be preserved and reviewed before replay:'+name)
    require(report.get('source_snapshot')==source_snapshot(manifest),'Paused source model/Adam/PCG snapshot changed')
    return dict(prior_owner=owner,prior_status='interrupted',prior_owned_records_checked=len(checked),
        cleanup_sha256=sha(prior/'cleanup.json'),terminal_receipt_sha256=sha(manifest['prior_recovery_terminal_receipt']))


def validate_adopted_file(manifest,stem,path):
    cp,seed,games,opponent=evaluation_specs(manifest)[stem]
    require(sha(path)==manifest['adopted_results'][stem]['sha256'],'Durable adoption bytes changed:'+stem)
    return phase.validate_evaluation(read(path),checkpoint=cp,seed=seed,games=games,opponent=opponent)


def adopt_results(manifest):
    output=Path(manifest['output']);staging=output/'adoption.prepare';provenance=output/'adoption.json'
    require(not staging.exists(),'Partial adoption staging preserved; explicit repair required')
    proof=dict(schema_version=1,run_id=RUN_ID,recovery_epoch=3,prior_recovery_run_id=PRIOR_RUN_ID,
        source_epoch2_output=manifest['prior_recovery_output'],adopted_results=manifest['adopted_results'],
        adopted_evaluation_attempts=27000,byte_identical=True,replay_provenance=manifest['replay_provenance'])
    if provenance.exists():
        require(read(provenance)==proof,'Existing adoption provenance differs')
        for stem in ADOPTED:validate_adopted_file(manifest,stem,output/(stem+'.json'))
        return proof
    require(not any((output/(stem+'.json')).exists()for stem in ADOPTED),'Unproven preexisting adoption output')
    staging.mkdir()
    for stem in ADOPTED:
        source=Path(manifest['adopted_results'][stem]['path']);validate_adopted_file(manifest,stem,source)
        target=staging/(stem+'.json')
        with source.open('rb')as src,target.open('xb')as dst:
            for block in iter(lambda:src.read(1<<20),b''):dst.write(block)
            dst.flush();os.fsync(dst.fileno())
        validate_adopted_file(manifest,stem,target)
    verify_sources(manifest)
    for stem in ADOPTED:(staging/(stem+'.json')).rename(output/(stem+'.json'))
    staging.rmdir();publish(provenance,proof)
    return proof


def terminal_receipt_context(manifest):
    path=Path(manifest['native_terminal_receipt']);require(not path.exists(),'New epoch3 terminal receipt already exists')
    require(manifest['recovery_coordinator_sha256']==sha(__file__)and manifest['recovery_epoch']==3,'Exact epoch3 coordinator required')
    owner=process_identity(os.getpid());require(owner and owner['pid']==owner['pgid']==owner['sid']and owner['uid']==os.getuid(),
        'Epoch3 coordinator must own a dedicated session')
    guard=safe.native_guard_monitor(manifest);guard.require_fresh();live=read(manifest['native_liveness'])
    require(live.get('run_id')==RUN_ID and live.get('recovery_epoch')==3
        and live.get('recovery_coordinator_sha256')==sha(__file__)
        and live.get('finite_game_budget')==live.get('starting_game_count')==140000
        and live.get('additional_game_budget')==0 and live.get('remaining_evaluation_attempts')==30000,
        'Native six-duel/no-training scope differs')
    native=manifest.get('_native_owner');require(native and len(native)==2,'Exact native binding required')
    cleanup=Path(manifest['output'])/'cleanup.json'
    return dict(receipt=str(path),coordinator=owner,recovery_epoch=3,recovery_coordinator_sha256=sha(__file__),started_utc=safe.utc(),
        prior_cleanup_sha256=sha(cleanup)if cleanup.exists()else None,
        native_owner=dict(windows_pid=native[0],creation_utc=native[1]),native_manifest_sha256=guard.last_probe['expected_manifest_sha256'])


def run(manifest):
    validate_manifest(manifest);verify_sources(manifest);prior=verify_prior_stopped(manifest);snapshot=source_snapshot(manifest)
    output=Path(manifest['output']);output.mkdir(parents=True,exist_ok=True);os.chdir(manifest['runtime'])
    lock=Path(manifest['shared_lock']).open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:lock.close();raise
    owner=process_identity(os.getpid());path=output/'report.json'
    report=dict(status='starting',run_id=RUN_ID,parent_run_id=PARENT_RUN_ID,prior_recovery_run_id=PRIOR_RUN_ID,
        recovery_epoch=3,started_utc=safe.utc(),preregistered_design=design(manifest),source_snapshot=snapshot,
        prior_terminal=prior,completed_evaluations={},training=False,no_training_or_passive_replay=True,
        adopted_evaluation_attempts=27000,new_evaluation_attempts=30000,registered_evaluation_attempts=57000,
        physical_recovery_evaluation_attempts_including_interrupted_replay=62000,
        continuation_100k_to140k_and_evaluations_physical_executions=109008,
        replay_provenance=manifest['replay_provenance'],
        limitations=['One continued training seed; game intervals exclude training-seed variance.',
            'Six new duels complete the existing frozen protocol; no training,smoke,passive or durable-result replay.',
            'The round10 duel replays5000 same-ID physical attempts whose earlier per-game outcomes were not durable.',
            'Logical dataset remains57000 policy/schedule outcomes; replay is not an independent extra sample.',
            'Open monitoring,paired baseline confirmation and actual neural duels are distinct protocols/cohorts.',
            'Original140k SIGKILL cause remains unproven and all original artifacts remain preserved.'])
    try:
        if path.exists():report=read(path)
        require(report.get('preregistered_design')==design(manifest)and report.get('source_snapshot')==snapshot,
            'Existing epoch3 output belongs to another source/design')
        report['adoption']=adopt_results(manifest);report['status']='running';publish(path,report)
        specs=evaluation_specs(manifest);confirmations={}
        for stem in ADOPTED:
            cp,seed,games,opponent=specs[stem]
            result,proof=previous.evaluate(manifest,cp,stem+'.json',seed,games,opponent=opponent)
            proof.update(origin='adopted_epoch2',adopted_from=manifest['adopted_results'][stem]['path'])
            report['completed_evaluations'][stem]=proof
            if stem=='monitor-0140':report['monitoring_summary']=result['summary']
            elif stem.startswith('confirmation-'):
                confirmations[stem]=result['games'];report.setdefault('confirmation_summaries',{})[stem.removeprefix('confirmation-')]=result['summary']
            else:report.setdefault('head_to_head',{})['imitation']=dict(summary=result['summary'],
                checkpoint_sha256=result['checkpoint_sha256'],opponent_checkpoint_sha256=result['opponent_checkpoint_sha256'],seed=seed,games=games)
            del result;publish(path,report)
        from patchwork_automaton.neural.rustmage_evaluation import paired_match_improvement
        report['paired_comparisons']={'primary-140k-minus-100k':paired_match_improvement(
            confirmations['confirmation-round-100'],confirmations['confirmation-round-140'])}
        del confirmations;publish(path,report)
        for name in NEW_OPPONENTS:
            stem='head-to-head-'+name;cp,seed,games,opponent=specs[stem]
            result,proof=previous.evaluate(manifest,cp,stem+'.json',seed,games,opponent=opponent)
            proof['origin']='new_epoch3';report['completed_evaluations'][stem]=proof
            report.setdefault('head_to_head',{})[name]=dict(summary=result['summary'],checkpoint_sha256=result['checkpoint_sha256'],
                opponent_checkpoint_sha256=result['opponent_checkpoint_sha256'],seed=seed,games=games)
            del result;publish(path,report)
        require(len(report['completed_evaluations'])==10,'All ten logical frozen results required')
        verify_sources(manifest);require(source_snapshot(manifest)==snapshot,'Frozen model/Adam/PCG changed')
        report.update(status='complete',finished_utc=safe.utc(),completed_evaluation_attempts=57000,
            completed_new_evaluation_attempts=30000);publish(path,report)
    except BaseException as error:
        report.update(status='interrupted'if isinstance(error,KeyboardInterrupt)else'error',
            last_error=dict(type=type(error).__name__,message=str(error),utc=safe.utc()));publish(path,report);raise
    finally:
        try:
            checked=verify_terminal_owned_sessions(manifest,owner);verify_sources(manifest);os.sync()
            if report['status']=='complete':require(len(checked)==6,'Six new frozen-duel child cleanup records required')
        except BaseException as error:
            try:publish(output/'cleanup.json',dict(finished_utc=safe.utc(),coordinator_pid=os.getpid(),status=report['status'],
                shared_lock_released=False,cleanup_error=str(error)))
            finally:lock.close()
            raise
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        publish(output/'cleanup.json',dict(finished_utc=safe.utc(),coordinator_pid=os.getpid(),status=report['status'],
            shared_lock_released=True,owned_children_remaining=0,owned_process_records_checked=len(checked)))


def run_with_terminal_receipt(manifest):
    validate_manifest(manifest);context=terminal_receipt_context(manifest);failure=None
    try:return run(manifest)
    except BaseException as error:failure=dict(type=type(error).__name__,message=str(error));raise
    finally:
        try:publish_native_terminal_receipt(manifest,context,failure)
        except BaseException as error:
            print(json.dumps(dict(event='native-terminal-receipt-not-published',utc=safe.utc(),run_id=RUN_ID,recovery_epoch=3,
                coordinator=context['coordinator'],error_type=type(error).__name__,error=str(error),original_run_error=failure),sort_keys=True),
                file=sys.stderr,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    run_with_terminal_receipt(read(args.manifest))
