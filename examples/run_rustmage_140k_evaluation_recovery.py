"""Finish only the ten registered frozen evaluations of the saved140k policy.

The failed original output stays immutable. In particular, its large training
report is checksummed as a stream and never decoded or retained in memory.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

from examples import run_rustmage_continuation_140k as phase

safe = phase.safe
utc, publish = safe.utc, safe.publish
process_identity = safe.process_identity
verify_terminal_owned_sessions = safe.verify_terminal_owned_sessions
publish_native_terminal_receipt = safe.publish_native_terminal_receipt
PHASE_SHA256 = '8f75a76f159c9a744fe30d691d508f62af5593f2ec2ab517f9a444e2e710e1b7'
SAFE_SHA256 = phase.SAFE_HELPER_SHA256
PARENT_RUN_ID = 'raytracer-continuation-140000-2026-10-26'
RUN_ID = 'raytracer-continuation-140000-evaluation-recovery2-2026-10-27'
HEAD_TO_HEAD_NAMES = phase.HEAD_TO_HEAD_NAMES
SOURCE_COUNTERS = {100:(32822,15857), 140:(45118,22137)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    path = Path(path)
    require(path.stat().st_size <= 16*1024**2, 'Only small recovery JSON may be decoded: '+str(path))
    return json.loads(path.read_text(), parse_constant=lambda value:(_ for _ in ()).throw(
        ValueError('Nonfinite JSON constant: '+value)))


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_manifest(manifest):
    expected = dict(run_id=RUN_ID,parent_run_id=PARENT_RUN_ID,recovery_epoch=2,
        source_round=100,target_round=140,target_games=140000,additional_training_games=0,
        training=False,monitoring_seed=320261025,monitoring_games=2000,
        confirmation_seed=520261026,confirmation_games=10000,
        head_to_head_seed=530261026,head_to_head_games=5000,
        head_to_head_opponents=list(HEAD_TO_HEAD_NAMES),remaining_evaluation_attempts=57000,
        owned_child_plan=10)
    for key,value in expected.items():
        require(manifest.get(key)==value and type(manifest.get(key)) is type(value),
            'Frozen evaluation-only design differs: '+key)
    for key in ('python','runtime','output','source_training','original_output','shared_lock',
                'original_study_manifest','original_launch_record','original_terminal_receipt',
                'native_liveness','native_terminal_receipt','imitation_checkpoint'):
        require(isinstance(manifest.get(key),str) and Path(manifest[key]).is_absolute(),
            'Absolute recovery path required: '+key)
    output,original = Path(manifest['output']).resolve(),Path(manifest['original_output']).resolve()
    require(output!=original and not output.is_relative_to(original) and not original.is_relative_to(output),
        'Evaluation recovery output must be separate from original output')
    require(Path(manifest['source_training']).resolve()==original/'training', 'Original training source path differs')
    require(manifest.get('hard_deadline_utc') is None, 'No elapsed-time stop is authorized')
    require(manifest.get('recovery_coordinator_sha256')==sha(__file__), 'New recovery coordinator SHA differs')
    require(sha(phase.__file__)==PHASE_SHA256 and sha(safe.__file__)==SAFE_SHA256,
        'Unchanged reviewed evaluation/lifecycle helper differs')
    runtime_pins = manifest.get('source_sha256')
    require(isinstance(runtime_pins,dict) and len(runtime_pins)==152
        and runtime_pins.get('examples/run_rustmage_140k_evaluation_recovery.py')==sha(__file__)
        and runtime_pins.get('examples/run_rustmage_continuation_140k.py')==PHASE_SHA256
        and runtime_pins.get('examples/run_rustmage_continuation.py')==SAFE_SHA256,
        'New152-file immutable runtime pins required')
    pins = manifest.get('source_artifacts_sha256')
    require(isinstance(pins,dict) and bool(pins), 'Immutable original artifacts must be pinned')
    source = Path(manifest['source_training'])
    needed = [source/'report.json',source/'checkpoints/latest.json',original/'study.json',original/'cleanup.json']
    needed += [source/'checkpoints'/f'round-{n:04d}'/name for n in range(141) for name in safe.FILES]
    needed += [Path(manifest[key]) for key in ('original_study_manifest','original_launch_record','original_terminal_receipt')]
    imitation = Path(manifest['imitation_checkpoint'])
    needed += [imitation,imitation.with_suffix('.manifest.json')]
    require(all(str(path) in pins for path in needed), 'All705 checkpoint files and original provenance must be pinned')
    native = manifest.get('expected_native_owner')
    require(isinstance(native,dict) and set(native)=={'windows_pid','creation_utc'}
        and type(native['windows_pid']) is int and native['windows_pid']>0
        and isinstance(native['creation_utc'],str), 'Exact new native owner required')
    require(isinstance(manifest.get('native_manifest_sha256'),str)
        and len(manifest['native_manifest_sha256'])==64, 'New native manifest SHA required')
    require(isinstance(manifest.get('source140_rng_sha256'),str)
        and len(manifest['source140_rng_sha256'])==64, 'Exact source140 PCG64 digest required')
    return manifest


def verify_sources(manifest):
    """Stream every source, including the366MB report; retain only digests."""
    runtime = Path(manifest['runtime']).resolve()
    for name,digest in manifest['source_sha256'].items():
        path = (runtime/name).resolve()
        require(path.is_relative_to(runtime) and sha(path)==digest, 'Frozen recovery runtime changed: '+name)
    for name,digest in manifest['source_artifacts_sha256'].items():
        require(sha(name)==digest, 'Immutable original artifact changed: '+name)


def design(manifest):
    return {key:deepcopy(manifest[key]) for key in ('run_id','parent_run_id','recovery_epoch','source_round',
        'target_round','target_games','additional_training_games','training','monitoring_seed','monitoring_games',
        'confirmation_seed','confirmation_games','head_to_head_seed','head_to_head_games',
        'head_to_head_opponents','remaining_evaluation_attempts','owned_child_plan')}


def verify_original_stopped(manifest):
    original = Path(manifest['original_output'])
    study,cleanup = read(original/'study.json'),read(original/'cleanup.json')
    old_manifest = read(manifest['original_study_manifest'])
    launch = read(manifest['original_launch_record'])
    receipt = read(manifest['original_terminal_receipt'])
    require(study.get('run_id')==manifest['parent_run_id'] and study.get('status')=='error'
        and study.get('last_error',{}).get('message','').startswith('Child failed with exit -9:'),
        'Original failed study identity/status differs')
    owner = launch.get('owner')
    require(isinstance(owner,dict) and all(type(owner.get(key)) is int for key in
        ('pid','pgid','sid','start_ticks','uid')) and owner['pid']==owner['pgid']==owner['sid']
        and owner['uid']==os.getuid(), 'Original exact Linux owner required')
    require(process_identity(owner['pid'])!=owner, 'Original coordinator remains live')
    require(cleanup.get('coordinator_pid')==owner['pid'] and cleanup.get('status')=='error'
        and cleanup.get('shared_lock_released') is True and cleanup.get('owned_children_remaining')==0
        and not cleanup.get('cleanup_error'), 'Original owned cleanup is not verified')
    require(old_manifest.get('run_id')==manifest['parent_run_id']
        and old_manifest.get('recovery_epoch')==1
        and old_manifest.get('recovery_coordinator_sha256')==PHASE_SHA256
        and Path(old_manifest.get('output','')).resolve()==original.resolve()
        and old_manifest.get('target_round')==140 and old_manifest.get('target_games')==140000,
        'Original training deployment provenance differs')
    original_runtime = dict(manifest['source_sha256'])
    original_runtime.pop('examples/run_rustmage_140k_evaluation_recovery.py')
    require(len(original_runtime)==151 and original_runtime==old_manifest.get('source_sha256'),
        'Original151 runtime files must remain byte-identical')
    require(receipt.get('run_id')==manifest['parent_run_id'] and receipt.get('recovery_epoch')==1
        and receipt.get('status')=='error' and receipt.get('coordinator')==owner
        and receipt.get('recovery_coordinator_sha256')==PHASE_SHA256
        and receipt.get('native_manifest_sha256')==old_manifest['native_manifest_sha256']
        and receipt.get('native_owner')==old_manifest['expected_native_owner']
        and receipt.get('cleanup_sha256')==sha(original/'cleanup.json')
        and receipt.get('cleanup',{}).get('cleanup_verified') is True,
        'Original exact-bound terminal receipt differs')
    old_scope = dict(manifest,output=str(original))
    checked = verify_terminal_owned_sessions(old_scope,owner)
    forbidden = [original/'monitor-0140.json',original/'confirmation-round-100.json',original/'confirmation-round-140.json']
    forbidden += [original/f'head-to-head-{name}.json' for name in HEAD_TO_HEAD_NAMES]
    require(not any(path.exists() for path in forbidden), 'Original final evaluation output already exists')
    for pattern in ('monitor-0140-*.process.json','confirmation-*.process.json','head-to-head-*.process.json'):
        require(not any(original.glob(pattern)), 'Original final evaluation was already admitted')
    return dict(original_owner=owner,original_status='error',original_owned_records_checked=len(checked),
        cleanup_sha256=sha(original/'cleanup.json'),terminal_receipt_sha256=sha(manifest['original_terminal_receipt']))


def source_snapshot(manifest):
    source = Path(manifest['source_training'])
    require(read(source/'checkpoints/latest.json')=={'checkpoint':'round-0140'}, 'Only committed140k is authorized')
    snapshots = {}
    for number,counters in SOURCE_COUNTERS.items():
        checkpoint = source/'checkpoints'/f'round-{number:04d}'
        pins = safe.verify_checkpoint(checkpoint,number)
        state = read(checkpoint/'state.json')
        phase.validate_config(state['config'])
        require((state['optimizer_steps'],state['critic_optimizer_steps'])==counters,
            'Source checkpoint Adam counters differ')
        snapshots[str(number)] = dict(round=number,files_sha256=pins,optimizer_steps=counters[0],
            critic_optimizer_steps=counters[1],trainer_rng_state_sha256=canonical(state['trainer_rng_state']),
            learning_config_sha256=canonical(state['config']))
    require(snapshots['100']['learning_config_sha256']==snapshots['140']['learning_config_sha256'],
        'Source100 and140 learning recipes differ')
    require(snapshots['140']['trainer_rng_state_sha256']==manifest['source140_rng_sha256'],
        'Exact committed140k PCG64 state differs')
    return dict(checkpoints=snapshots,training_report_sha256=manifest['source_artifacts_sha256'][str(source/'report.json')],
        original_training_files=705,training_report_decoded=False,models_and_both_adam_and_rng_immutable=True)


def terminal_receipt_context(manifest):
    path = Path(manifest['native_terminal_receipt'])
    require(not path.exists(), 'New epoch2 terminal receipt already exists')
    require(manifest['recovery_coordinator_sha256']==sha(__file__) and manifest['recovery_epoch']==2,
        'Exact new evaluation coordinator/epoch required')
    owner = process_identity(os.getpid())
    require(owner and owner['pid']==owner['pgid']==owner['sid'] and owner['uid']==os.getuid(),
        'Evaluation coordinator must own its dedicated session')
    guard = safe.native_guard_monitor(manifest)
    guard.require_fresh()
    live = read(manifest['native_liveness'])
    require(live.get('run_id')==manifest['run_id'] and live.get('recovery_epoch')==2
        and live.get('recovery_coordinator_sha256')==sha(__file__)
        and live.get('finite_game_budget')==live.get('starting_game_count')==140000
        and live.get('additional_game_budget')==0, 'Native evaluation-only scope differs')
    native = manifest.get('_native_owner')
    require(native and len(native)==2, 'Exact new native owner binding required')
    cleanup = Path(manifest['output'])/'cleanup.json'
    return dict(receipt=str(path),coordinator=owner,recovery_epoch=2,
        recovery_coordinator_sha256=sha(__file__),started_utc=utc(),
        prior_cleanup_sha256=sha(cleanup) if cleanup.exists() else None,
        native_owner=dict(windows_pid=native[0],creation_utc=native[1]),
        native_manifest_sha256=guard.last_probe['expected_manifest_sha256'])


def evaluate(manifest, checkpoint, filename, seed, games, *, opponent=None):
    verify_sources(manifest)
    path = Path(manifest['output'])/filename
    result = phase.evaluate(manifest,checkpoint,path,seed,games,opponent=opponent)
    verify_sources(manifest)
    return result,dict(path=str(path),sha256=sha(path),seed=seed,attempts=games,
        checkpoint_sha256=result['checkpoint_sha256'],
        **({'opponent_checkpoint_sha256':result['opponent_checkpoint_sha256']} if opponent is not None else {}))


def run(manifest):
    validate_manifest(manifest)
    verify_sources(manifest)
    original = verify_original_stopped(manifest)
    snapshot = source_snapshot(manifest)
    output = Path(manifest['output'])
    output.mkdir(parents=True,exist_ok=True)
    os.chdir(manifest['runtime'])
    lock = Path(manifest['shared_lock']).open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        lock.close();raise
    owner = process_identity(os.getpid())
    report_path = output/'report.json'
    report = dict(status='starting',started_utc=utc(),
        run_id=manifest['run_id'],parent_run_id=manifest['parent_run_id'],recovery_epoch=2,
        preregistered_design=design(manifest),source_snapshot=snapshot,original_terminal=original,
        completed_evaluations={},training=False,no_training_or_passive_replay=True,
        limitations=['One continued training seed; game intervals exclude training-seed variation.',
            'The fixed deck/opponent and related old policies limit generalization.',
            'Monitoring uses an open repeated cohort. Shared schedules are not independent samples.',
            'Original SIGKILL cause remains unproven; all training and200 passive endpoint games had completed.'])
    try:
        if report_path.exists():report=read(report_path)
        require(report.get('preregistered_design')==design(manifest) and report.get('source_snapshot')==snapshot,
            'Existing recovery output belongs to another source/design')
        report['status']='running';publish(report_path,report)
        source = Path(manifest['source_training'])
        policies = {number:source/'checkpoints'/f'round-{number:04d}'/'policy.keras' for number in (100,140)}
        monitor,proof = evaluate(manifest,policies[140],'monitor-0140.json',manifest['monitoring_seed'],2000)
        report['monitoring_summary']=monitor['summary'];del monitor
        report['completed_evaluations']['monitor-0140']=proof;publish(report_path,report)
        confirmations = {}
        for number in (100,140):
            name=f'round-{number}'
            result,proof = evaluate(manifest,policies[number],f'confirmation-{name}.json',manifest['confirmation_seed'],10000)
            confirmations[name]=result
            report.setdefault('confirmation_summaries',{})[name]=result['summary']
            report['completed_evaluations'][f'confirmation-{name}']=proof;publish(report_path,report)
        from patchwork_automaton.neural.rustmage_evaluation import paired_match_improvement
        report['paired_comparisons']={'primary-140k-minus-100k':paired_match_improvement(
            confirmations['round-100']['games'],confirmations['round-140']['games'])}
        del result,confirmations
        publish(report_path,report)
        for name in HEAD_TO_HEAD_NAMES:
            opponent = (Path(manifest['imitation_checkpoint']) if name=='imitation' else
                source/'checkpoints'/f"round-{int(name.removeprefix('round-')):04d}"/'policy.keras')
            result,proof = evaluate(manifest,policies[140],f'head-to-head-{name}.json',manifest['head_to_head_seed'],5000,
                opponent=opponent)
            report.setdefault('head_to_head',{})[name]=dict(summary=result['summary'],
                checkpoint_sha256=result['checkpoint_sha256'],opponent_checkpoint_sha256=result['opponent_checkpoint_sha256'],
                seed=manifest['head_to_head_seed'],games=5000)
            del result
            report['completed_evaluations'][f'head-to-head-{name}']=proof;publish(report_path,report)
        require(len(report['completed_evaluations'])==10, 'All ten frozen evaluations are required')
        verify_sources(manifest)
        require(source_snapshot(manifest)==snapshot, 'Model/Adam/RNG checkpoint state changed during frozen evaluation')
        report.update(status='complete',finished_utc=utc(),completed_evaluation_attempts=57000)
        publish(report_path,report)
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',
            last_error=dict(type=type(error).__name__,message=str(error),utc=utc()))
        publish(report_path,report);raise
    finally:
        try:
            checked=verify_terminal_owned_sessions(manifest,owner);verify_sources(manifest);os.sync()
        except BaseException as error:
            try:publish(output/'cleanup.json',dict(finished_utc=utc(),coordinator_pid=os.getpid(),
                status=report['status'],shared_lock_released=False,cleanup_error=str(error)))
            finally:lock.close()
            raise
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        publish(output/'cleanup.json',dict(finished_utc=utc(),coordinator_pid=os.getpid(),
            status=report['status'],shared_lock_released=True,owned_children_remaining=0,
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
                run_id=manifest.get('run_id'),recovery_epoch=2,coordinator=context['coordinator'],
                error_type=type(error).__name__,error=str(error),original_run_error=failure),sort_keys=True),
                file=sys.stderr,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    run_with_terminal_receipt(read(args.manifest))
