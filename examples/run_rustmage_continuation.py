"""Continue one immutable RustMage PPO trajectory from 40k to 100k attempts."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import zipfile

import numpy as np

FILES = ('policy.keras', 'policy.manifest.json', 'optimizer.npz', 'critic-optimizer.npz', 'state.json')
SOURCE_ROUND = 40
TARGET_ROUND = 100
GAMES_PER_ROUND = 1000
TRAINING_SEED = 300261025
MILESTONES = (41, 50, 60, 70, 80, 90, 100)
MONITORING_ROUNDS = (50, 60, 70, 80, 90, 100)
HEAD_TO_HEAD_NAMES = ('imitation', 'round-10', 'round-20', 'round-40', 'round-60', 'round-80')


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
    runtime = Path(manifest['runtime']).resolve()
    for name, digest in manifest['source_sha256'].items():
        path = (runtime/name).resolve()
        if not path.is_relative_to(runtime) or sha(path) != digest:
            raise ValueError('Frozen runtime changed: '+name)
    for name, digest in manifest['source_artifacts_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Immutable continuation source changed: '+name)


def verify_checkpoint(root, expected_round):
    root = Path(root)
    if {p.name for p in root.iterdir()} != set(FILES):
        raise ValueError('Checkpoint artifact set differs')
    state = read(root/'state.json')
    if state['round'] != expected_round:
        raise ValueError('Checkpoint round differs')
    rng = state.get('trainer_rng_state')
    if not isinstance(rng, dict) or rng.get('bit_generator') != 'PCG64':
        raise ValueError('Checkpoint lacks its PCG64 continuation state')
    probe = np.random.PCG64()
    probe.state = rng
    with zipfile.ZipFile(root/'policy.keras') as archive:
        if archive.testzip() is not None:
            raise ValueError('Checkpoint ZIP failed CRC')
    for filename, key in [('optimizer.npz', 'optimizer_steps'),
                          ('critic-optimizer.npz', 'critic_optimizer_steps')]:
        with np.load(root/filename, allow_pickle=False) as arrays:
            if ('v0' not in arrays or int(arrays['v0']) != state[key]
                    or not all(np.all(np.isfinite(arrays[k])) for k in arrays.files)):
                raise ValueError('Checkpoint optimizer counter or finiteness differs')
    return {name:sha(root/name) for name in FILES}


def validate_manifest(manifest):
    expected = dict(source_round=SOURCE_ROUND, target_round=TARGET_ROUND,
        target_games=TARGET_ROUND*GAMES_PER_ROUND, training_seed=TRAINING_SEED,
        milestones=list(MILESTONES), monitoring_rounds=list(MONITORING_ROUNDS),
        monitoring_seed=320261025, monitoring_games=2000,
        confirmation_seed=450261026, confirmation_games=10000,
        head_to_head_seed=460261026, head_to_head_games=5000,
        preflight_head_to_head_seed=470261026, preflight_head_to_head_games=8)
    for key, value in expected.items():
        if manifest.get(key) != value or (type(value) is int and type(manifest.get(key)) is not int):
            raise ValueError('Finite continuation design differs: '+key)
    for key in ('run_id', 'python', 'runtime', 'output', 'source_training', 'shared_lock',
                'native_liveness', 'native_terminal_receipt', 'imitation_checkpoint'):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError('Missing continuation manifest field: '+key)
    for key in ('runtime', 'output', 'source_training', 'shared_lock', 'native_liveness',
                'native_terminal_receipt', 'imitation_checkpoint'):
        if not Path(manifest[key]).is_absolute():
            raise ValueError('Absolute continuation path required: '+key)
    source, output = Path(manifest['source_training']).resolve(), Path(manifest['output']).resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('Continuation output must be separate from immutable source')
    for key in ('source_sha256', 'source_artifacts_sha256'):
        if not isinstance(manifest.get(key), dict) or not manifest[key]:
            raise ValueError('Missing source pins: '+key)
    required_source = [source/'report.json', source/'checkpoints/latest.json']
    required_source += [source/'checkpoints'/f'round-{i:04d}'/name
                        for i in range(SOURCE_ROUND+1) for name in FILES]
    required_source += [Path(manifest['imitation_checkpoint']),
                        Path(manifest['imitation_checkpoint']).with_suffix('.manifest.json')]
    if any(str(path) not in manifest['source_artifacts_sha256'] for path in required_source):
        raise ValueError('Source artifact manifest must pin all checkpoints, report, pointer and imitation')
    runtime_options = manifest.get('runtime_options', {})
    if set(runtime_options) != {'learner_gru_backend', 'update_pipeline',
                                'policy_cache_mib', 'actor_epoch_cache_mib'}:
        raise ValueError('Explicit validated runtime options are required')
    if (runtime_options['learner_gru_backend'] != 'cudnn-memory'
            or runtime_options['update_pipeline'] not in ('gpu', 'auto')
            or any(type(runtime_options[k]) is not int or runtime_options[k] != 8192
                   for k in ('policy_cache_mib', 'actor_epoch_cache_mib'))):
        raise ValueError('Runtime options differ from validated fused/cache profile')
    opponents = manifest.get('head_to_head_opponents')
    if opponents != list(HEAD_TO_HEAD_NAMES):
        raise ValueError('Finite head-to-head opponent list differs')
    return manifest


def migration_config(config):
    """One explicit topology change; every learning and RNG setting stays put."""
    expected = dict(algorithm='ppo', architecture='separate', train_mode='versus_baseline',
        games_per_round=1000, seed=TRAINING_SEED, train_batch=512, ppo_epochs=2,
        validation_fraction=.2, workers=48, inference_processes=4, collector='processes',
        learning_rate=.0001, value_learning_rate=.0003, value_max_steps=157,
        target_kl=.02, max_state_kl=.5, entropy_weight=.01, critic_unroll=1,
        critic_update_scope='all', advantage_baseline='critic', learner_device='gpu',
        inference_device='cpu', validation_device='cpu', critic_warmup_games=1000,
        critic_warmup_max_steps=157, train_horizon=60, eval_every_games=0,
        baseline_eval_games=0, concurrent_updates=True, prefetch_batches=2,
        measurement_workers=4, rollout_cache_mib=512, rollout_storage='pickle')
    if any(config.get(k) != value for k, value in expected.items()):
        raise ValueError('Original 40k learning recipe differs')
    if not isinstance(config.get('pooled_collector_config'), str) or not config['pooled_collector_config']:
        raise ValueError('Expected the original pooled collection topology')
    migrated = deepcopy(config)
    migrated['pooled_collector_config'] = None
    return migrated


def _fsync_tree(root):
    for path in sorted(Path(root).rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    directories = [p for p in Path(root).rglob('*') if p.is_dir()] + [Path(root)]
    for path in sorted(directories, key=lambda p:len(p.parts), reverse=True):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)


def prepare(manifest):
    """Fork historical metadata only; never initialize or reset learned state."""
    source = Path(manifest['source_training'])
    output = Path(manifest['output'])/'training'
    source_report = read(source/'report.json')
    source_config = source_report['config']
    config = migration_config(source_config)
    if (source_report.get('status') != 'complete'
            or [row['round'] for row in source_report['rounds']] != list(range(1, SOURCE_ROUND+1))
            or source_report.get('critic_warmup', {}).get('status') != 'complete'
            or read(source/'checkpoints/latest.json') != {'checkpoint':'round-0040'}):
        raise ValueError('Source report/pointer is not the complete warmed 40k trajectory')
    source_state = read(source/'checkpoints/round-0040/state.json')
    if source_state['config'] != source_config:
        raise ValueError('Source state/config disagrees with report')
    if (source_state.get('optimizer_steps') != 13289
            or source_state.get('critic_optimizer_steps') != 6437):
        raise ValueError('Expected exact original 40k optimizer counters')
    verify_sources(manifest)
    if output.exists():
        migration = read(output/'continuation-migration.json')
        if (migration.get('run_id') != manifest['run_id']
                or migration.get('source_report_sha256') != sha(source/'report.json')
                or migration.get('target_config') != config):
            raise ValueError('Existing continuation belongs to another source/configuration')
        existing = read(output/'report.json')
        latest = read(output/'checkpoints/latest.json')['checkpoint']
        if not isinstance(latest, str) or not latest.startswith('round-') or Path(latest).name != latest:
            raise ValueError('Invalid continuation pointer')
        number = int(latest.removeprefix('round-'))
        if not SOURCE_ROUND <= number <= TARGET_ROUND:
            raise ValueError('Existing continuation round is outside finite plan')
        state = read(output/'checkpoints'/latest/'state.json')
        if existing['config'] != config or state['config'] != config:
            raise ValueError('Existing continuation learning configuration differs')
        verify_checkpoint(output/'checkpoints'/latest, number)
        return output, config, migration

    staging = output.with_name('training.prepare')
    if staging.exists():
        raise ValueError('Partial preparation is preserved and requires explicit repair')
    staging.mkdir(parents=True)
    state_changes = {}
    for number in range(SOURCE_ROUND+1):
        name = f'round-{number:04d}'
        original = source/'checkpoints'/name
        verify_checkpoint(original, number)
        copied = staging/'checkpoints'/name
        shutil.copytree(original, copied)
        state = read(copied/'state.json')
        if state['config'] != source_config:
            raise ValueError('Historical checkpoint learning configurations differ')
        before = deepcopy(state)
        state['config'] = deepcopy(config)
        publish(copied/'state.json', state)
        if {k:v for k,v in before.items() if k != 'config'} != {k:v for k,v in state.items() if k != 'config'}:
            raise ValueError('Migration changed checkpoint state outside its topology config')
        for filename in FILES[:-1]:
            if sha(copied/filename) != sha(original/filename):
                raise ValueError('Migration changed model/manifest/Adam bytes')
        state_changes[name] = dict(source_sha256=sha(original/'state.json'),
                                  migrated_sha256=sha(copied/'state.json'))
    shutil.copyfile(source/'checkpoints/latest.json', staging/'checkpoints/latest.json')
    migration = dict(schema_version=1, run_id=manifest['run_id'], created_utc=utc(),
        source_training=str(source), source_round=SOURCE_ROUND, target_round=TARGET_ROUND,
        source_report_sha256=sha(source/'report.json'), source_config=source_config,
        target_config=config, changed_config_fields=['pooled_collector_config'],
        topology=dict(before='pooled laptop/Raytracer', after='Raytracer-only ordinary process collector',
                      laptop_reserved_games=0, laptop_producer=False, fallback_wait_seconds=0),
        state_changes=state_changes, source_checkpoint_sha256=verify_checkpoint(source/'checkpoints/round-0040',40),
        preservation=dict(model_and_manifest_bytes=True, both_adam_bytes=True,
                          all_rng_states=True, all_optimizer_counters=True, no_critic_rewarm=True))
    report = deepcopy(source_report)
    report['config'] = config
    report['continuation_fork'] = migration
    publish(staging/'report.json', report)
    publish(staging/'continuation-migration.json', migration)
    verify_sources(manifest)
    _fsync_tree(staging)
    staging.rename(output)
    descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    publish(output.parent/'migration.json', migration)
    return output, config, migration


def train_command(python, output, config, rounds, runtime_options):
    if type(rounds) is not int or rounds not in MILESTONES:
        raise ValueError('Unregistered continuation target')
    if config.get('pooled_collector_config') is not None:
        raise ValueError('Continuation cannot reserve laptop games')
    command = [python, '-m', 'examples.train_rustmage', '--output', str(output),
               '--rounds', str(rounds), '--resume']
    for key, value in {**config, **runtime_options}.items():
        if key in ('algorithm', 'advantage_scale_floor', 'ppo_clip') or value is None:
            continue
        flag = '--'+key.replace('_', '-')
        if isinstance(value, bool):
            if value: command.append(flag)
        else: command.extend((flag, str(value)))
    return command


class NativeGuardFailure(RuntimeError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__('Native guard supervision failed: '+json.dumps(diagnostic, sort_keys=True, allow_nan=False))


def native_guard_probe(manifest, *, wall_time=None):
    """One read-only observation; a valid conflicting identity is never retried."""
    path = manifest.get('native_liveness')
    if not path:
        return dict(valid=True, reason='not_configured', hard_failure=False)
    observation = dict(valid=False, path=str(path), hard_failure=False)
    expected = manifest.get('native_manifest_sha256')
    if (not isinstance(expected, str) or len(expected)!=64
            or any(c not in '0123456789abcdefABCDEF' for c in expected)):
        return dict(observation, reason='expected_manifest_hash_invalid', hard_failure=True)
    expected = expected.lower()
    observation['expected_manifest_sha256'] = expected
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        return dict(observation, reason='file_read_error', error_type=type(error).__name__,
                    error_errno=error.errno, error=str(error))
    observation.update(observed_bytes=len(raw), observed_file_sha256=hashlib.sha256(raw).hexdigest())
    if len(raw)>65536:
        return dict(observation, reason='json_too_large')
    try:
        live = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError('Nonfinite JSON constant: '+value)))
    except (ValueError, UnicodeError) as error:
        return dict(observation, reason='json_invalid', error_type=type(error).__name__, error=str(error))
    if not isinstance(live, dict):
        return dict(observation, reason='json_not_object')
    observed_hash = live.get('manifest_sha256')
    if (not isinstance(observed_hash, str) or len(observed_hash)!=64
            or any(c not in '0123456789abcdefABCDEF' for c in observed_hash)):
        return dict(observation, reason='observed_manifest_hash_invalid')
    observation['observed_manifest_sha256'] = observed_hash.lower()
    if observed_hash.lower()!=expected:
        return dict(observation, reason='manifest_hash_mismatch', hard_failure=True)
    owner = live.get('windows_pid'), live.get('creation_utc')
    if type(owner[0]) is not int or owner[0]<1 or not isinstance(owner[1], str):
        return dict(observation, reason='owner_identity_invalid')
    try:
        created = datetime.fromisoformat(owner[1].replace('Z','+00:00'))
        if created.tzinfo is None:
            raise ValueError('Owner creation time must include UTC offset')
    except ValueError as error:
        return dict(observation, reason='owner_creation_time_invalid', error=str(error))
    observation['observed_owner'] = dict(windows_pid=owner[0], creation_utc=owner[1])
    pinned_owner = manifest.get('expected_native_owner')
    if (not isinstance(pinned_owner, dict) or pinned_owner != dict(windows_pid=owner[0], creation_utc=owner[1])):
        return dict(observation, reason='pinned_owner_identity_mismatch', hard_failure=True,
                    expected_owner=pinned_owner)
    previous = manifest.get('_native_owner')
    if previous is not None and tuple(previous)!=owner:
        return dict(observation, reason='owner_identity_mismatch', hard_failure=True,
                    expected_owner=dict(windows_pid=previous[0], creation_utc=previous[1]))
    try:
        stamp = datetime.fromisoformat(live['utc'].replace('Z','+00:00'))
        if stamp.tzinfo is None:
            raise ValueError('Heartbeat time must include UTC offset')
        now = time.time() if wall_time is None else wall_time()
        age = now-stamp.timestamp()
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        return dict(observation, reason='heartbeat_time_invalid', error_type=type(error).__name__, error=str(error))
    if not -30<=age<60:
        return dict(observation, reason='heartbeat_stale_or_future', age_seconds=age)
    manifest.setdefault('_native_owner', owner)
    return dict(observation, valid=True, reason='fresh', age_seconds=age)


def native_alive(manifest):
    """Compatibility API: one fresh observation, without the monitor's grace."""
    return native_guard_probe(manifest)['valid']


class NativeGuardMonitor:
    """Tolerate read/staleness blips for 15 monotonic seconds, never new owners.

    The grace is measured after completed failing observations. OS filesystem
    calls retain their ordinary kernel behavior; this is not an I/O watchdog.
    """
    def __init__(self, manifest, *, grace_seconds=15., monotonic=None, wall_time=None,
                 sleep=None, observer=None):
        if not 0<grace_seconds<=15:
            raise ValueError('Native guard transient grace must be in (0,15] seconds')
        self.manifest, self.grace_seconds = manifest, grace_seconds
        self.monotonic = monotonic or time.monotonic
        self.wall_time = wall_time or time.time
        self.sleep = sleep or time.sleep
        self.observer = observer
        self.failed_since = self.last_valid = self.last_probe = None
        self.failed_reason = None

    def emit(self, event, **fields):
        value = dict(event=event, utc=utc(), **fields)
        if self.observer:
            self.observer(value)
            return
        if not self.manifest.get('native_liveness'):
            return
        path = Path(self.manifest.get('native_liveness_diagnostics') or
                    Path(self.manifest['output'])/'native-liveness-recovery.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())

    def check(self):
        observation = native_guard_probe(self.manifest, wall_time=self.wall_time)
        now = self.monotonic()
        self.last_probe = observation
        if observation['valid']:
            if self.failed_since is not None:
                self.emit('native-guard-recovered', failure_seconds=now-self.failed_since,
                          previous_reason=self.failed_reason, observation=observation)
            self.failed_since = self.failed_reason = None
            self.last_valid = dict(monotonic=now, observation=observation)
            return True
        if self.failed_since is None:
            self.failed_since = now
        elapsed = now-self.failed_since
        if observation['hard_failure'] or elapsed>=self.grace_seconds:
            diagnostic = dict(observation=observation, failure_seconds=elapsed,
                transient_grace_seconds=self.grace_seconds, last_valid=self.last_valid)
            self.emit('native-guard-failed', **diagnostic)
            raise NativeGuardFailure(diagnostic)
        if self.failed_reason!=observation['reason']:
            self.emit('native-guard-transient', observation=observation,
                      failure_seconds=elapsed, transient_grace_seconds=self.grace_seconds)
        self.failed_reason = observation['reason']
        return False

    def require_fresh(self):
        while not self.check():
            remaining = self.grace_seconds-(self.monotonic()-self.failed_since)
            self.sleep(min(.25, max(0.,remaining)))
        self.emit('native-guard-pre-spawn-verified', observation=self.last_probe)


def native_guard_monitor(manifest):
    monitor = manifest.get('_native_guard_monitor')
    if monitor is None:
        monitor = manifest['_native_guard_monitor'] = NativeGuardMonitor(manifest)
    return monitor

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
    guard = native_guard_monitor(manifest)
    guard.require_fresh()  # Never spawn work against an absent or stale native guard.
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
                guard.check()  # Transient failures have one bounded monotonic grace window.
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



def terminal_receipt_context(manifest):
    """Bind one invocation before work; never reuse a preceding terminal file."""
    receipt = Path(manifest['native_terminal_receipt'])
    if not receipt.is_absolute() or receipt.exists():
        raise ValueError('A new absolute native terminal receipt path is required')
    expected = manifest.get('recovery_coordinator_sha256')
    if not isinstance(expected, str) or expected!=sha(__file__):
        raise ValueError('Recovery coordinator source pin differs')
    epoch = manifest.get('recovery_epoch')
    if type(epoch) is not int or epoch!=1:
        raise ValueError('Recovery epoch must be explicitly pinned to integer one')
    if not isinstance(manifest.get('run_id'), str) or not manifest['run_id']:
        raise ValueError('Recovery run identity is missing')
    owner = process_identity(os.getpid())
    if (owner is None or owner['pid']!=owner['pgid'] or owner['pid']!=owner['sid']
            or owner['uid']!=os.getuid()):
        raise ValueError('Current coordinator must own a dedicated current-user session')
    cleanup = Path(manifest['output'])/'cleanup.json'
    context = dict(receipt=str(receipt), coordinator=owner, recovery_epoch=epoch,
        recovery_coordinator_sha256=expected, started_utc=utc(),
        prior_cleanup_sha256=sha(cleanup) if cleanup.exists() else None)
    # Even an already-completed, fully cached study must bind its actual native
    # guardian before a receipt can be considered for that guardian.
    guard = native_guard_monitor(manifest)
    guard.require_fresh()
    native_owner = manifest.get('_native_owner')
    if not native_owner or len(native_owner)!=2:
        raise ValueError('Terminal receipt requires a bound exact native owner')
    context.update(native_owner=dict(windows_pid=native_owner[0],creation_utc=native_owner[1]),
                   native_manifest_sha256=guard.last_probe['expected_manifest_sha256'])
    return context


def verify_terminal_owned_sessions(manifest, coordinator):
    """Read process records and /proc only; never signal a process here."""
    checked = []
    for path in sorted(Path(manifest['output']).glob('*.process.json')):
        record = read(path)
        owner = record.get('owner')
        if (not isinstance(owner, dict) or any(type(owner.get(k)) is not int for k in
                ('pid','pgid','sid','start_ticks','uid')) or owner['uid']!=coordinator['uid']
                or owner['pid']!=owner['pgid'] or owner['pid']!=owner['sid']):
            raise ValueError('Invalid owned process record: '+str(path))
        if owner==coordinator:
            continue  # A launcher may also retain this coordinator's own record.
        live = {value['pid']:value for value in refresh_owned(owner,{owner['pid']:owner})}
        # A leader can exit before descendants are adopted/scanned.  A matching
        # surviving session conservatively blocks receipt even without an anchor.
        for entry in Path('/proc').glob('[0-9]*'):
            value = process_identity(int(entry.name))
            if (value and value['sid']==owner['sid'] and value['uid']==owner['uid']
                    and value['start_ticks']>=owner['start_ticks']):
                live[value['pid']] = value
        if live:
            raise ValueError('Owned child session remains alive: '+json.dumps(list(live.values()),sort_keys=True))
        current = owner['start_ticks']>=coordinator['start_ticks']
        cleanup_path = path.with_name(path.name.removesuffix('.process.json')+'.cleanup.json')
        child_cleanup = read(cleanup_path) if cleanup_path.exists() else None
        if current and (not isinstance(child_cleanup,dict) or child_cleanup.get('owner')!=owner
                or child_cleanup.get('remaining')!=[] or not child_cleanup.get('verified_utc')
                or child_cleanup.get('error') or child_cleanup.get('cleanup_error')):
            raise ValueError('Current owned child cleanup is not verified: '+str(path))
        checked.append(dict(path=str(path),sha256=sha(path),owner=owner,
            current_invocation=current,cleanup_sha256=sha(cleanup_path) if child_cleanup else None))
    return checked


def publish_native_terminal_receipt(manifest, context, run_error):
    if process_identity(os.getpid())!=context['coordinator']:
        raise ValueError('Coordinator identity changed before terminal publication')
    cleanup_path = Path(manifest['output'])/'cleanup.json'
    cleanup_digest = sha(cleanup_path)
    cleanup = read(cleanup_path)
    if (cleanup_digest==context['prior_cleanup_sha256']
            or cleanup.get('coordinator_pid')!=os.getpid()
            or type(cleanup.get('coordinator_pid')) is not int
            or cleanup.get('shared_lock_released') is not True
            or cleanup.get('status') not in ('complete','error','interrupted')
            or not cleanup.get('finished_utc') or cleanup.get('cleanup_error')
            or cleanup.get('error') or (run_error is not None and cleanup['status']=='complete')):
        raise ValueError('New coordinator cleanup and shared-lock release are not verified')
    bound = manifest.get('_native_owner')
    if (not bound or dict(windows_pid=bound[0],creation_utc=bound[1])!=context['native_owner']):
        raise ValueError('Native owner changed before terminal publication')
    checked = verify_terminal_owned_sessions(manifest,context['coordinator'])
    path = Path(context['receipt'])
    if path.exists():
        raise ValueError('Terminal receipt already exists; refusing to replace it')
    receipt = dict(schema_version=1,run_id=manifest['run_id'],recovery_epoch=context['recovery_epoch'],
        native_manifest_sha256=context['native_manifest_sha256'],
        recovery_coordinator_sha256=context['recovery_coordinator_sha256'],
        status=cleanup['status'],coordinator=context['coordinator'],native_owner=context['native_owner'],
        cleanup=dict(cleanup,owned_children_remaining=0,cleanup_verified=True,
                     owned_process_records_checked=len(checked)),cleanup_sha256=cleanup_digest,
        owned_process_records=checked,started_utc=context['started_utc'],
        finished_utc=cleanup['finished_utc'],published_utc=utc(),run_error=run_error)
    publish(path,receipt)
    return receipt


def run_with_terminal_receipt(manifest):
    """Leave run()/ML/cleanup unchanged; receipt failure cannot change its exit."""
    context = terminal_receipt_context(manifest)
    failure = None
    try:
        return run(manifest)
    except BaseException as error:
        failure = dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        try:
            publish_native_terminal_receipt(manifest,context,failure)
        except BaseException as error:
            # An absent receipt intentionally leaves the external native guard
            # awake for operator recovery. Preserve the original return/exception.
            try:
                import sys
                print(json.dumps(dict(event='native-terminal-receipt-not-published',utc=utc(),
                    run_id=manifest.get('run_id'),recovery_epoch=manifest.get('recovery_epoch'),
                    coordinator=context['coordinator'],error_type=type(error).__name__,error=str(error),
                    original_run_error=failure),sort_keys=True,allow_nan=False),file=sys.stderr,flush=True)
            except BaseException:
                pass


def validate_training(output, rounds):
    """Check a committed prefix, including historical pooled provenance."""
    output = Path(output)
    report = read(output/'report.json')
    all_rounds = report['rounds']
    if ([r['round'] for r in all_rounds] != list(range(1, len(all_rounds)+1))
            or len(all_rounds) < rounds or not SOURCE_ROUND <= rounds <= TARGET_ROUND):
        raise ValueError('Training milestone has missing round history')
    config = report['config']
    if config.get('pooled_collector_config') is not None:
        raise ValueError('Continuation learning configuration still reserves laptop games')
    source_state = read(output/'checkpoints/round-0040/state.json')
    actor_steps, critic_steps = source_state['optimizer_steps'], source_state['critic_optimizer_steps']
    optimization_games = heldout_games = continuation_raytracer_games = 0
    for row in all_rounds[:rounds]:
        number, games = row['round'], row['games']
        first = (number-1)*GAMES_PER_ROUND
        expected = list(range(first, first+GAMES_PER_ROUND))
        if len(games) != GAMES_PER_ROUND or sorted(g['index'] for g in games) != expected:
            raise ValueError('Training has duplicate/missing scheduled game IDs')
        for game in games:
            index = game['index']
            if (game['seed'] != TRAINING_SEED+GAMES_PER_ROUND+index
                    or game['seat'] != index%2 or game['starting_player'] != f'P{(index//2)%2}'
                    or game['mode'] != 'versus_baseline'
                    or game['status'] not in ('win', 'loss', 'draw', 'cutoff', 'error')):
                raise ValueError('Training seed, seat, starting player, mode or outcome differs')
            if number > SOURCE_ROUND and (game.get('collection_host', 'raytracer') != 'raytracer'
                                         or game.get('pool_fallback')):
                raise ValueError('Continuation used laptop generation or pooled fallback')
        split = row['split']
        train, held = split['training_indices'], split['validation_indices']
        if (len(train) != 800 or len(held) != 200 or len(set(train)) != 800
                or len(set(held)) != 200 or set(train)&set(held)
                or set(train)|set(held) != set(expected)):
            raise ValueError('Whole-game 800/200 holdout has missing/overlapping IDs')
        optimization_games += split['training']['terminal_games']
        heldout_games += split['validation']['attempted_games']
        if number > SOURCE_ROUND:
            if (row.get('collection_resources', {}).get('laptop_games', 0) != 0
                    or row.get('collection_resources', {}).get('pool')):
                raise ValueError('Continuation retained a pooled collection path')
            if not 0 < row['update']['critic_steps'] <= 157:
                raise ValueError('Continuation critic step cap exceeded')
            if row['update']['accepted_steps'] < 0:
                raise ValueError('Invalid accepted actor step count')
            actor_steps += row['update']['accepted_steps']
            critic_steps += row['update']['critic_steps']
            if (row['update']['optimizer_steps'] != actor_steps
                    or row['update']['critic_optimizer_steps'] != critic_steps):
                raise ValueError('Round optimizer continuity differs')
            continuation_raytracer_games += len(games)
    checkpoint = output/'checkpoints'/f'round-{rounds:04d}'
    integrity = verify_checkpoint(checkpoint, rounds)
    state = read(checkpoint/'state.json')
    if (state['config'] != config or state['optimizer_steps'] != actor_steps
            or state['critic_optimizer_steps'] != critic_steps):
        raise ValueError('Committed optimizer/config continuity differs')
    return dict(games=rounds*GAMES_PER_ROUND, added_games=(rounds-SOURCE_ROUND)*GAMES_PER_ROUND,
        optimization_games=optimization_games, heldout_games=heldout_games,
        continuation_raytracer_games=continuation_raytracer_games, continuation_laptop_games=0,
        actor_optimizer_steps=actor_steps, critic_optimizer_steps=critic_steps,
        checkpoint_sha256=integrity)


def _new_log(output, label):
    return Path(output)/(label+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.log')


def validate_evaluation(result, *, checkpoint, seed, games, opponent=None):
    """No outcome-based replacement and no reuse across different cohorts."""
    checkpoint = Path(checkpoint)
    if (result.get('status') != 'complete' or result.get('checkpoint_sha256') != sha(checkpoint)
            or result.get('config', {}).get('cohort', {}).get('seed_base') != seed
            or len(result.get('games', [])) != games
            or result.get('summary', {}).get('attempted_games') != games):
        raise ValueError('Frozen evaluation cohort/provenance differs')
    mode = 'versus_neural' if opponent is not None else 'versus_baseline'
    if opponent is not None and result.get('opponent_checkpoint_sha256') != sha(opponent):
        raise ValueError('Frozen neural opponent differs')
    cohort = result['config']['cohort']
    collection = result['config'].get('collection', {})
    if (cohort.get('attempted_games') != games or cohort.get('index_base') != 0
            or cohort.get('seed_last') != seed+games-1
            or cohort.get('training_trajectories') is not False
            or collection.get('greedy') is not False or collection.get('max_own_turns') != 60
            or collection.get('max_decisions') != 6000 or collection.get('repeat_limit') != 20):
        raise ValueError('Frozen evaluation sampling, limits or rollout isolation differs')
    expected = [(i, seed+i, i%2, f'P{(i//2)%2}') for i in range(games)]
    actual = [(r['index'], r['seed'], r['seat'], r['starting_player']) for r in result['games']]
    if actual != expected:
        raise ValueError('Evaluation has missing/reordered/replaced scheduled attempts')
    outcomes = {'win':0, 'loss':0, 'draw':0, 'cutoff':0, 'error':0}
    for row in result['games']:
        if row.get('mode') != mode or row.get('trajectories') or row.get('status') not in outcomes:
            raise ValueError('Evaluation mode, rollout isolation or outcome differs')
        outcomes[row['status']] += 1
    for status, key in [('win','wins'), ('loss','losses'), ('draw','draws'),
                        ('cutoff','cutoffs'), ('error','errors')]:
        if result['summary'].get(key) != outcomes[status]:
            raise ValueError('Evaluation summary omits an outcome')
    return result


def evaluate(manifest, checkpoint, output, seed, games, *, opponent=None,
             workers=None, replicas_per_policy=None):
    checkpoint, output = Path(checkpoint), Path(output)
    before = dict(checkpoint=sha(checkpoint))
    if opponent is not None:
        before['opponent'] = sha(opponent)
    if not output.exists():
        module = ('examples.evaluate_rustmage_head_to_head' if opponent is not None
                  else 'examples.evaluate_rustmage_policy')
        command = [manifest['python'], '-m', module, '--checkpoint', str(checkpoint),
                   '--output', str(output), '--seed', str(seed), '--games', str(games)]
        if opponent is not None:
            command += ['--opponent-checkpoint', str(opponent)]
            if workers is not None:
                command += ['--workers', str(workers)]
            if replicas_per_policy is not None:
                command += ['--replicas-per-policy', str(replicas_per_policy)]
        run_child(command, _new_log(output.parent, output.stem), manifest)
    if sha(checkpoint) != before['checkpoint'] or (opponent is not None and sha(opponent) != before['opponent']):
        raise ValueError('Evaluation changed a source policy')
    return validate_evaluation(read(output), checkpoint=checkpoint, seed=seed, games=games, opponent=opponent)


def preflight_head_to_head(manifest, training):
    """Eight extra smoke evaluations, never a training or final sealed cohort."""
    verify_sources(manifest)
    checkpoint = Path(training)/'checkpoints/round-0040/policy.keras'
    opponent = Path(training)/'checkpoints/round-0010/policy.keras'
    before = {str(path.resolve()):sha(path) for policy in (checkpoint, opponent)
              for path in (policy, policy.with_suffix('.manifest.json'))}
    result = evaluate(manifest, checkpoint, Path(manifest['output'])/'preflight-head-to-head.json',
        manifest['preflight_head_to_head_seed'], manifest['preflight_head_to_head_games'],
        opponent=opponent, workers=4, replicas_per_policy=1)
    if (result['summary'].get('terminal_games') != 8
            or result['summary'].get('cutoffs') != 0 or result['summary'].get('errors') != 0
            or result['summary'].get('seat_start_balanced') is not True
            or result.get('game_engines') != 4 or result.get('replicas_per_policy') != 1
            or result.get('inference_device') != 'cpu' or result.get('training') is not False):
        raise ValueError('Head-to-head smoke evaluation did not pass its terminal/balance/runtime gate')
    if (result.get('source_files_sha256') != before
            or {str(path):sha(path) for path in map(Path, before)} != before):
        raise ValueError('Head-to-head smoke evaluation source hashes differ')
    batching = result.get('batching', {})
    for role in ('focal', 'opponent'):
        stats = batching.get(role, {})
        if (stats.get('requests', 0) <= 0 or stats.get('device') != '/CPU:0'
                or len(stats.get('replicas', [])) != 1
                or any(replica.get('visible_gpus') for replica in stats['replicas'])):
            raise ValueError('Both smoke policies must use their own GPU-hidden CPU inference')
    for seat in (0, 1):
        for starter in ('P0', 'P1'):
            if result['summary'].get('seat_start_strata', {}).get(
                    f'seat_{seat}_start_{starter}', {}).get('attempted_games') != 2:
                raise ValueError('Smoke seat/start stratum is incomplete')
    verify_sources(manifest)
    return dict(status='passed', seed=manifest['preflight_head_to_head_seed'], games=8,
                summary=result['summary'], sources_unchanged=True, both_policy_inference_observed=True,
                source_files_sha256=before)


def _head_to_head_checkpoint(manifest, training, name):
    if name == 'imitation':
        return Path(manifest['imitation_checkpoint'])
    if name not in HEAD_TO_HEAD_NAMES:
        raise ValueError('Unregistered neural opponent')
    return Path(training)/'checkpoints'/f"round-{int(name.removeprefix('round-')):04d}"/'policy.keras'


def run(manifest):
    validate_manifest(manifest)
    output = Path(manifest['output'])
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(manifest['runtime'])
    verify_sources(manifest)
    lock = Path(manifest['shared_lock']).open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = process_identity(os.getpid())
    study_path = output/'study.json'
    study = read(study_path) if study_path.exists() else dict(
        status='starting', run_id=manifest['run_id'], started_utc=utc(),
        source_round=SOURCE_ROUND, target_round=TARGET_ROUND, target_games=100000,
        added_games=60000, training_seed=TRAINING_SEED,
        monitoring_seed=manifest['monitoring_seed'], confirmation_seed=manifest['confirmation_seed'],
        head_to_head_seed=manifest['head_to_head_seed'], milestones={},
        design=dict(continuation='Exact original round40 model/both Adam/PCG; one seed',
            topology='Raytracer-only game generation; no laptop producer, reserve or fallback wait',
            primary='100k-minus-40k recorded wins against active baseline on fresh paired cohort',
            head_to_head='Actual frozen 100k games against six older checkpoints',
            endpoint_fixed=True, extra_training_arms=False, elapsed_time_hard_stop=False,
            monitoring_games_each=2000, confirmation_games_each=10000, head_to_head_games_each=5000,
            evaluation_games_are_additional=True))
    try:
        if any(study.get(k) != manifest[k] for k in
               ('run_id', 'training_seed', 'monitoring_seed', 'confirmation_seed', 'head_to_head_seed')):
            raise ValueError('Existing study belongs to another finite preregistration')
        training, config, migration = prepare(manifest)
        study.update(status='running', training_output=str(training), migration=migration,
                     runtime_options=manifest['runtime_options'])
        publish(study_path, study)
        study['head_to_head_preflight'] = preflight_head_to_head(manifest, training)
        publish(study_path, study)
        for rounds in MILESTONES:
            verify_sources(manifest)
            latest = read(training/'checkpoints/latest.json')['checkpoint']
            completed = int(latest.removeprefix('round-'))
            report_status = read(training/'report.json')['status']
            if completed < rounds or (completed == rounds and report_status != 'complete'):
                run_child(train_command(manifest['python'], training, config, rounds, manifest['runtime_options']),
                          _new_log(output, f'train-through-{rounds:04d}'), manifest)
                if read(training/'report.json')['status'] != 'complete':
                    raise ValueError('Trainer did not complete its finite target')
            progress = validate_training(training, rounds)
            study['milestones'][str(rounds)] = dict(completed_utc=utc(), **progress)
            publish(study_path, study)
            if rounds in MONITORING_ROUNDS:
                result = evaluate(manifest, training/'checkpoints'/f'round-{rounds:04d}'/'policy.keras',
                    output/f'monitor-{rounds:04d}.json', manifest['monitoring_seed'], manifest['monitoring_games'])
                study['milestones'][str(rounds)]['monitoring_summary'] = result['summary']
                publish(study_path, study)
        final_progress = validate_training(training, TARGET_ROUND)
        study['status'] = 'final_evaluation'
        publish(study_path, study)
        policies = {name:training/'checkpoints'/f'round-{number:04d}'/'policy.keras'
                    for name, number in [('round-40',40), ('round-100',100)]}
        confirmation = {name:evaluate(manifest, path, output/f'confirmation-{name}.json',
                        manifest['confirmation_seed'], manifest['confirmation_games'])
                        for name,path in policies.items()}
        from patchwork_automaton.neural.rustmage_evaluation import paired_match_improvement
        comparison = paired_match_improvement(confirmation['round-40']['games'], confirmation['round-100']['games'])
        head_to_head = {}
        for name in HEAD_TO_HEAD_NAMES:
            opponent = _head_to_head_checkpoint(manifest, training, name)
            result = evaluate(manifest, policies['round-100'], output/f'head-to-head-{name}.json',
                manifest['head_to_head_seed'], manifest['head_to_head_games'], opponent=opponent)
            head_to_head[name] = dict(summary=result['summary'],
                checkpoint_sha256=result['checkpoint_sha256'],
                opponent_checkpoint_sha256=result['opponent_checkpoint_sha256'],
                seed=manifest['head_to_head_seed'], games=manifest['head_to_head_games'])
            study['head_to_head'] = head_to_head
            publish(study_path, study)
        verify_sources(manifest)
        study.update(status='complete', finished_utc=utc(), final=final_progress,
            confirmation_summaries={k:v['summary'] for k,v in confirmation.items()},
            paired_comparisons={'primary-100k-minus-40k':comparison}, head_to_head=head_to_head,
            limitations=[
                'One continued training seed; game intervals do not measure training-seed variation.',
                'More data also adds rounds/optimizer steps; this does not isolate their individual effects.',
                'Baseline learning/confirmation remains one fixed opponent and deck matchup.',
                'Head-to-head matchups share schedules but have different opponents; do not pool as independent cohorts.',
                'Runtime arithmetic can differ within validated tolerances; exact checkpoint continuation does not promise bitwise future trajectories.',
            ])
        publish(study_path, study)
    except BaseException as error:
        study.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'error',
                     last_error=dict(type=type(error).__name__, message=str(error), utc=utc()))
        publish(study_path, study)
        raise
    finally:
        # Successful cleanup is required before releasing shared ownership.
        # If verification fails, leave a visible failure and no terminal receipt.
        try:
            checked = verify_terminal_owned_sessions(manifest, owner)
            verify_sources(manifest)
            os.sync()
        except BaseException as error:
            publish(output/'cleanup.json', dict(finished_utc=utc(), coordinator_pid=os.getpid(),
                status=study['status'], shared_lock_released=False, cleanup_error=str(error)))
            raise
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        publish(output/'cleanup.json', dict(finished_utc=utc(), coordinator_pid=os.getpid(),
            shared_lock_released=True, status=study['status'], owned_children_remaining=0,
            owned_process_records_checked=len(checked)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    run_with_terminal_receipt(read(args.manifest))

