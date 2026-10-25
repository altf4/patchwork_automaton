"""Finite, resumable 40k PPO study; one learner, immutable starts and sealed evaluation."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import zipfile

import numpy as np

FILES = ('policy.keras', 'policy.manifest.json', 'optimizer.npz', 'critic-optimizer.npz', 'state.json')
MILESTONES = (4, 10, 20, 40)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def verify_sources(manifest):
    runtime = Path(manifest['runtime'])
    for name, digest in manifest['source_sha256'].items():
        if sha(runtime/name) != digest:
            raise ValueError('Frozen runtime changed: '+name)
    for name, digest in manifest['initialization_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Read-only initialization changed: '+name)


def verify_checkpoint(root, expected_round):
    root=Path(root)
    state=read(root/'state.json')
    if state['round'] != expected_round: raise ValueError('Checkpoint round differs')
    integrity={name:sha(root/name) for name in FILES}
    with zipfile.ZipFile(root/'policy.keras') as archive:
        if archive.testzip() is not None: raise ValueError('Checkpoint ZIP failed CRC')
    for filename, key in [('optimizer.npz','optimizer_steps'),('critic-optimizer.npz','critic_optimizer_steps')]:
        with np.load(root/filename,allow_pickle=False) as arrays:
            if int(arrays['v0']) != state[key] or not all(np.all(np.isfinite(arrays[k])) for k in arrays.files):
                raise ValueError('Checkpoint optimizer mismatch')
    return integrity


def train_command(python, output, config, rounds):
    command=[python,'-m','examples.train_rustmage','--output',str(output),'--rounds',str(rounds),'--resume']
    for key,value in config.items():
        if key in ('algorithm','advantage_scale_floor','ppo_clip') or value is None: continue
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value: command.append(flag)
        else:command.extend((flag,str(value)))
    return command


def prepare(manifest):
    output=Path(manifest['output'])/'training'
    if (output/'report.json').exists():
        existing=read(output/'report.json')
        if (existing['config'].get('seed')!=manifest['training_seed'] or
                existing['config'].get('pooled_collector_config')!=manifest['pool_config']):
            raise ValueError('Existing training belongs to another frozen configuration')
        return output,existing['config']
    if output.exists(): raise ValueError('Incomplete preparation requires explicit repair')
    warm=Path(manifest['warm_checkpoint'])
    state=read(warm/'state.json')
    if state['round'] != 0 or state['optimizer_steps'] != 0 or state['critic_optimizer_steps'] != 157:
        raise ValueError('Expected untouched actor and verified warmed critic')
    config=deepcopy(state['config'])
    config.update(seed=manifest['training_seed'],train_mode='versus_baseline',games_per_round=1000,
                  baseline_eval_games=0,eval_every_games=0,
                  pooled_collector_config=manifest['pool_config'])
    expected=dict(train_batch=512,ppo_epochs=2,validation_fraction=.2,workers=48,inference_processes=4,
                  learning_rate=.0001,value_learning_rate=.0003,value_max_steps=157,
                  target_kl=.02,max_state_kl=.5,entropy_weight=.01,critic_unroll=1,
                  critic_update_scope='all',advantage_baseline='critic',learner_device='gpu',
                  inference_device='cpu',validation_device='cpu')
    if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('Pinned learning recipe differs')
    import shutil
    staging=output.with_name('training.prepare')
    initial=staging/'checkpoints/round-0000'
    shutil.copytree(warm,initial)
    fork=dict(source=str(warm),source_hashes={f:sha(warm/f) for f in FILES},
              actor_rl_updates_at_start=0,actor_matches_imitation=manifest['actor_verification'],
              optimizer_initialization='Exact warmed model and both Adam arrays preserved',
              rng_reset=dict(seed=manifest['training_seed'],purpose='Fresh PPO sampling and shuffle seed'))
    state.update(config=config,trainer_rng_state=np.random.default_rng(manifest['training_seed']).bit_generator.state,
                 pooled_study_fork=fork)
    report=deepcopy(read(manifest['warm_report']))
    for key in ('baseline','before','after','comparison','throughput','seconds_this_invocation','last_error','batching'):
        report.pop(key,None)
    report.update(config=config,status='prepared',rounds=[],invocations=[],evaluations=[],opponent_evaluations=[],
                  training_games_completed=0,training_games_target=40000,collected_games_completed=0,
                  optimization_games_completed=0,validation_games_completed=0,validation_terminal_games_completed=0,
                  pooled_study_fork=fork)
    state['initial_report']=deepcopy(report)
    publish(initial/'state.json',state)
    publish(staging/'checkpoints/latest.json',dict(checkpoint='round-0000'))
    publish(staging/'report.json',report)
    for item in initial.iterdir():
        if item.is_file():
            with item.open('rb') as stream:os.fsync(stream.fileno())
    staging.rename(output)
    publish(output.parent/'initialization.json',fork)
    return output,config


def native_alive(manifest):
    if not manifest.get('native_liveness'): return True
    try:
        live=read(manifest['native_liveness'])
        age=time.time()-datetime.fromisoformat(live['utc'].replace('Z','+00:00')).timestamp()
        expected=os.environ.get('MTGO_NATIVE_MANIFEST_SHA', manifest.get('native_manifest_sha256'))
        if not expected or live['manifest_sha256']!=expected or not -30<=age<60:return False
        owner=(live['windows_pid'],live['creation_utc'])
        if type(owner[0]) is not int or owner[0]<1 or not isinstance(owner[1],str):return False
        return manifest.setdefault('_native_owner',owner)==owner
    except (OSError,ValueError,KeyError):return False


def process_identity(pid, allow_zombie=False):
    try:
        root=Path('/proc')/str(pid)
        text=(root/'stat').read_text(); fields=text[text.rfind(')')+2:].split()
        if fields[0] in ('Z','X') and not allow_zombie:return None
        return dict(pid=int(pid),pgid=int(fields[2]),sid=int(fields[3]),
                    start_ticks=int(fields[19]),uid=root.stat().st_uid)
    except (OSError,ValueError,IndexError):return None


def refresh_owned(owner, known):
    # A surviving exact identity anchors this session even after its leader exits.
    anchors=[v for v in known.values() if process_identity(v['pid'])==v]
    for path in Path('/proc').glob('[0-9]*'):
        value=process_identity(int(path.name))
        if (value and value['sid']==owner['sid'] and value['uid']==owner['uid']
                and value['start_ticks']>=owner['start_ticks']):
            # Subreaper adoption is an independent ownership anchor when a
            # fast-exiting leader dies before its first descendant scan.
            try:
                text=(path/'stat').read_text();parent=int(text[text.rfind(')')+2:].split()[1])
            except (OSError,ValueError,IndexError):continue
            if anchors or parent==os.getpid():known[value['pid']]=value
    return [v for v in known.values() if process_identity(v['pid'])==v]


def drain_owned(owner, known, schedule=((signal.SIGINT,40),(signal.SIGTERM,15),(signal.SIGKILL,10))):
    for number,seconds in schedule:
        live=refresh_owned(owner,known)
        if not live:return
        for value in live:
            if process_identity(value['pid'])==value:
                try:os.kill(value['pid'],number)
                except ProcessLookupError:pass
        until=time.monotonic()+seconds
        while refresh_owned(owner,known) and time.monotonic()<until:time.sleep(.2)
    if refresh_owned(owner,known):raise RuntimeError('Owned child processes survived cleanup')


def run_child(command, log, manifest):
    import ctypes
    # Linux PR_SET_CHILD_SUBREAPER: orphaned pool workers stay attributable to
    # this dedicated coordinator even if the immediate trainer exits quickly.
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'Could not establish child ownership')
    with Path(log).open('ab',buffering=0) as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        owner=process_identity(child.pid,allow_zombie=True)
        known={child.pid:owner} if owner else {}
        try:
            if owner is None:raise RuntimeError('Child ended before identity was captured')
            publish(Path(log).with_suffix('.process.json'),dict(owner=owner,command=command,started_utc=utc()))
            while child.poll() is None:
                refresh_owned(owner,known)
                if not native_alive(manifest):raise RuntimeError('Native owner heartbeat ended; preserving round boundary')
                time.sleep(.25)
            if child.returncode:raise RuntimeError('Child failed with exit '+str(child.returncode)+': '+str(log))
        finally:
            if owner is not None:drain_owned(owner,known)
            child.wait()
            for pid in known:
                if pid!=child.pid:
                    try:os.waitpid(pid,os.WNOHANG)
                    except ChildProcessError:pass
            os.fsync(stream.fileno())
            publish(Path(log).with_suffix('.cleanup.json'),dict(owner=owner,remaining=[],
                child_exit=child.returncode,verified_utc=utc()))


def validate_training(output, rounds):
    d=read(Path(output)/'report.json')
    if d['status']!='complete' or [r['round'] for r in d['rounds']]!=list(range(1,rounds+1)):
        raise ValueError('Training milestone incomplete')
    for r in d['rounds']:
        rows=r['games']
        if len(rows)!=1000 or len({g['index'] for g in rows})!=1000:raise ValueError('Duplicate/missing games')
        if {g['index'] for g in rows}!=set(range((r['round']-1)*1000,r['round']*1000)):
            raise ValueError('Training schedule differs')
        if any(g['seed']!=d['config']['seed']+1000+g['index'] for g in rows):raise ValueError('Training seed differs')
        if len(r['split']['training_indices'])!=800 or len(r['split']['validation_indices'])!=200:
            raise ValueError('Whole-game partition differs')
        if set(r['split']['training_indices'])&set(r['split']['validation_indices']):raise ValueError('Holdout leak')
        if not 0<r['update']['critic_steps']<=157:raise ValueError('Critic step cap exceeded')
    checkpoint=Path(output)/'checkpoints'/f'round-{rounds:04d}'
    verify_checkpoint(checkpoint,rounds)
    if read(checkpoint/'state.json')['critic_optimizer_steps']!=157+sum(r['update']['critic_steps'] for r in d['rounds']):
        raise ValueError('Cumulative critic optimizer counter differs')
    return dict(games=rounds*1000,optimization_games=d['optimization_games_completed'],
                heldout_games=d['validation_games_completed'],laptop_games=sum(
                    g.get('collection_host')=='laptop' for r in d['rounds'] for g in r['games']))


def evaluate(manifest, checkpoint, output, seed, games=5000):
    checkpoint=Path(checkpoint);output=Path(output)
    if not output.exists():
        run_child([manifest['python'],'-m','examples.evaluate_rustmage_policy','--checkpoint',str(checkpoint),
            '--output',str(output),'--seed',str(seed),'--games',str(games)],output.with_suffix('.log'),manifest)
    d=read(output)
    if d['status']!='complete' or d['checkpoint_sha256']!=sha(checkpoint) or d['config']['cohort']['seed_base']!=seed:
        raise ValueError('Frozen evaluation provenance differs')
    if len(d['games'])!=games or d['summary']['attempted_games']!=games:raise ValueError('Incomplete evaluation cohort')
    return d


def run(manifest):
    if manifest['target_games']!=40000 or manifest['milestones']!=list(MILESTONES):raise ValueError('Finite study budget differs')
    output=Path(manifest['output']);output.mkdir(parents=True,exist_ok=True)
    os.chdir(manifest['runtime'])
    verify_sources(manifest)
    lock=Path(manifest['shared_lock']).open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    study=read(output/'study.json') if (output/'study.json').exists() else dict(
        status='starting',run_id=manifest['run_id'],started_utc=utc(),target_games=40000,training_seed=manifest['training_seed'],
        monitoring_seed=manifest['monitoring_seed'],confirmation_seed=manifest['confirmation_seed'],milestones={})
    if any(study.get(k)!=manifest[k] for k in ('run_id','training_seed','monitoring_seed','confirmation_seed')):
        raise ValueError('Existing study belongs to another preregistration')
    exchange=Path(read(manifest['pool_config'])['exchange_root'])
    (exchange/'producer-stop.json').unlink(missing_ok=True)
    try:
        training,config=prepare(manifest)
        study.update(status='running',training_output=str(training));publish(output/'study.json',study)
        evaluate(manifest,manifest['imitation_checkpoint'],output/'monitor-imitation.json',manifest['monitoring_seed'],games=2000)
        for rounds in MILESTONES:
            verify_sources(manifest)
            latest=read(training/'checkpoints/latest.json')['checkpoint']
            completed=int(latest.rsplit('-',1)[1])
            if completed<rounds:
                run_child(train_command(manifest['python'],training,config,rounds),output/f'train-through-{rounds:04d}.log',manifest)
            if completed<=rounds:
                progress=validate_training(training,rounds)
                study['milestones'][str(rounds)]=dict(completed_utc=utc(),**progress)
                publish(output/'study.json',study)
            evaluate(manifest,training/'checkpoints'/f'round-{rounds:04d}'/'policy.keras',
                     output/f'monitor-{rounds:04d}.json',manifest['monitoring_seed'],games=2000)
        policies={'imitation':Path(manifest['imitation_checkpoint']),
                  'round-4':training/'checkpoints/round-0004/policy.keras',
                  'round-40':training/'checkpoints/round-0040/policy.keras'}
        confirmation={name:evaluate(manifest,path,output/f'confirmation-{name}.json',manifest['confirmation_seed'])
                      for name,path in policies.items()}
        from patchwork_automaton.neural.rustmage_evaluation import paired_match_improvement
        comparison={name:paired_match_improvement(confirmation[base]['games'],confirmation[candidate]['games'])
                    for name,base,candidate in [('primary-40k-minus-4k','round-4','round-40'),
                       ('40k-minus-imitation','imitation','round-40'),('4k-minus-imitation','imitation','round-4')]}
        verify_sources(manifest)
        study.update(status='complete',finished_utc=utc(),final=validate_training(training,40),
                     confirmation_summaries={k:v['summary'] for k,v in confirmation.items()},paired_comparisons=comparison,
                     limitations=['One training seed; paired game intervals do not measure training-seed variability.',
                                  'Strength is measured against the fixed training opponent on this deck.'])
        publish(output/'study.json',study)
        try:
            run_child([manifest['python'],'-m','examples.plot_rustmage_pooled_study',
                '--study-root',str(output),'--output',str(output/'learning-curve')],output/'plot.log',manifest)
            study['plots']='learning-curve.png and learning-curve.svg'
        except Exception as error:
            study['plot_error']=str(error)
        publish(output/'study.json',study)
    except BaseException as error:
        study.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',
                     last_error=dict(type=type(error).__name__,message=str(error),utc=utc()))
        publish(output/'study.json',study)
        raise
    finally:
        publish(exchange/'producer-stop.json',dict(status=study['status'],utc=utc()))
        os.sync()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        publish(output/'cleanup.json',dict(finished_utc=utc(),coordinator_pid=os.getpid(),
            shared_lock_released=True,status=study['status']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    run(read(args.manifest))
