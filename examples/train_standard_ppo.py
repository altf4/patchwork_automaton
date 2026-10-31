"""One shared Standard PPO agent: balanced CPU games, GPU updates, durable rounds."""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

from patchwork_automaton.baselines.standard_decks import STANDARD_DECKS
from patchwork_automaton.rl.standard_ppo import (schedule, normalization,
    validate_collection, compact_games, cleanup_rollouts)
from patchwork_automaton.rl.standard_selfplay_schedule import (
    assign_opponents, snapshot_round_for_round)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    files = [p for folder in ('src','examples') for p in Path(folder).rglob('*.py')]
    files += list(Path('src/patchwork_automaton/data').glob('*.json'))
    files += [Path('.local/rustmage/target/release/librustmage.so'),
              Path('.local/rustmage/python/rustmage.py'),Path('.local/rustmage/data/observation-schema.txt')]
    if os.environ.get('PATCHWORK_STANDARD_PLAYBACK', 'direct') == 'direct':
        from patchwork_automaton.neural.direct_standard import library_path
        files.append(library_path())
    return {str(p):sha(p) for p in sorted(files)}


CHECKPOINT_FILES = ('policy.keras', 'policy.manifest.json', 'optimizer.npz',
                    'critic-optimizer.npz', 'state.json')


def checkpoint_hashes(directory):
    directory = Path(directory)
    return {name:sha(directory/name) for name in CHECKPOINT_FILES}


def assert_fork_recipe(parent, branch):
    """The fork may change opponents and evaluation points, never PPO learning."""
    old = dict(parent)
    new = dict(branch)
    old.pop('milestones')
    new.pop('milestones')
    # Inference replicas are runtime settings; the self-play A/B benchmark
    # validated a different count from the parent without updating weights.
    if new.get('opponent_mode') == 'frozen_pool':
        old.pop('inference_processes', None)
        new.pop('inference_processes', None)
    for name in ('opponent_mode', 'opponent_script_fraction',
                 'opponent_inference_processes', 'opponent_snapshot_schedule',
                 'pilot_branch'):
        new.pop(name, None)
    if old != new:
        different = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        raise ValueError(f'Fork changed the PPO recipe: {different}')


def resolve_inference_processes(args):
    """Use benchmarked self-play replicas while preserving resumed recipes."""
    if args.inference_processes is not None:
        return args.inference_processes
    if args.resume:
        report = json.loads((args.output / 'report.json').read_text())
        return report['config']['inference_processes']
    return 2 if args.opponent_mode == 'frozen_pool' else 4


def validate_parent_copy(fork, root):
    parent = Path(fork['parent_checkpoint'])
    copied = Path(root) / parent.name
    expected = fork['parent_checkpoint_sha256']
    if checkpoint_hashes(parent) != expected or checkpoint_hashes(copied) != expected:
        raise ValueError('Parent or copied checkpoint changed after fork')
    if 'parent_report' in fork and sha(fork['parent_report']) != fork['parent_report_sha256']:
        raise ValueError('Completed parent report changed after fork')


class Resources:
    def __init__(self, root):
        self.root = Path(root)
        self.stop = threading.Event()
        self.minimum_available_ram = self.minimum_free_disk = 1 << 62
        self.peak_temporary_disk = 0
        self.cpu_total = self.cpu_busy = 0
        self.previous = None
        self.thread = threading.Thread(target=self.watch,daemon=True)

    def watch(self):
        while not self.stop.is_set():
            info = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            self.minimum_available_ram = min(self.minimum_available_ram,int(info['MemAvailable'].split()[0])*1024)
            self.minimum_free_disk = min(self.minimum_free_disk,shutil.disk_usage(self.root).free)
            values = list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
            now = (sum(values[:8]),values[3]+values[4])
            if self.previous:
                total,idle = now[0]-self.previous[0],now[1]-self.previous[1]
                self.cpu_total += total; self.cpu_busy += total-idle
            self.previous = now
            size = 0
            seen = set()
            for p in Path('/proc/self/fd').iterdir():
                try:
                    if str(self.root/'cache') in os.readlink(p):
                        stat = p.stat()
                        identity = (stat.st_dev,stat.st_ino)
                        if identity not in seen:
                            size += stat.st_size
                            seen.add(identity)
                except (FileNotFoundError,OSError):pass
            self.peak_temporary_disk = max(size,self.peak_temporary_disk)
            self.stop.wait(1)

    def __enter__(self):
        self.thread.start();return self

    def __exit__(self,*args):
        self.stop.set();self.thread.join()

    def result(self):
        return dict(minimum_available_ram_bytes=self.minimum_available_ram,
            minimum_free_disk_bytes=self.minimum_free_disk,peak_temporary_disk_bytes=self.peak_temporary_disk,
            cpu_busy_percent=100*self.cpu_busy/max(self.cpu_total,1))


async def run(args):
    os.environ.update(PATCHWORK_ROLLOUT_STORAGE='pickle',PATCHWORK_OBSERVATION_PIPELINE='python')
    import numpy as np
    import keras
    from patchwork_automaton.neural.devices import configure_runtime,device_scope,assert_model_device
    from patchwork_automaton.neural.standard_policy import StandardPolicy
    from patchwork_automaton.neural.batching import example_batch
    from patchwork_automaton.neural.experiment import atomic_json,make_trainer,load_checkpoint,save_checkpoint
    from patchwork_automaton.neural.inference_processes import weights_digest
    from patchwork_automaton.neural.recurrent import configure_gru_backend
    from patchwork_automaton.neural.update_pipeline import resolve_update_pipeline,apply_update_pipeline
    from patchwork_automaton.neural.host_memory import HostMemoryReclaimer
    from patchwork_automaton.neural.validation import GameValidator
    from patchwork_automaton.neural.rollout_records import read_trajectory
    from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector
    from patchwork_automaton.rl.standard_selfplay_processes import StandardSelfplayCollector
    from examples.train_rustmage import place_trainer
    from examples.train_standard_imitation import evaluation_summary

    args.inference_processes = resolve_inference_processes(args)
    schedule(args.games_per_round,seed=args.seed)
    if args.resume and args.fork_checkpoint is not None:
        raise ValueError('Resume and fork initialization are mutually exclusive')
    if args.opponent_mode != 'legacy' and args.fork_checkpoint is None and not args.resume:
        raise ValueError('A controlled opponent branch must start from a checkpoint fork')
    if args.opponent_mode == 'legacy' and args.fork_checkpoint is not None:
        raise ValueError('Fork requires an explicit control or self-play opponent mode')
    if args.pilot_branch and args.opponent_mode != 'frozen_pool':
        raise ValueError('Full-round pilot requires frozen-opponent collection')
    if not args.resume and args.fork_checkpoint is None and args.source is None:
        raise ValueError('A fresh PPO run needs an imitation source')
    if not 0 < args.update_timeout < float('inf'):
        raise ValueError('Update wall-clock budget must be positive and finite')
    if (args.rounds<1 or args.workers%args.runners or args.inference_processes%args.runners
            or args.workers<args.inference_processes or args.inference_processes<args.runners):
        raise ValueError('Worker and replica totals must divide evenly among runners')
    if args.eval_games<4 or args.eval_games%4:raise ValueError('Evaluation games must balance seats and starts')
    if args.opponent_mode == 'frozen_pool' and (
            args.opponent_inference_processes % args.runners or
            args.opponent_inference_processes < args.runners or
            args.opponent_inference_processes > args.workers):
        raise ValueError('Opponent inference replicas must divide among runners')
    if args.fork_checkpoint is not None and (args.games_per_round != 1000 or args.milestones):
        raise ValueError('Checkpoint-200 fork uses 1,000-game rounds and external endpoint evaluation')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    lock = (args.output/'run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (args.output/'cache').mkdir(exist_ok=True)
    tempfile.tempdir = str((args.output/'cache').resolve())
    configure_runtime('gpu')
    config = dict(algorithm='ppo',seed=args.seed,learning_rate=.0001,value_learning_rate=.0003,
        train_batch=512,ppo_epochs=2,entropy_weight=.01,target_kl=.02,max_state_kl=.5,
        value_max_steps=157,critic_update_scope='all',critic_unroll=1,concurrent_updates=True,
        prefetch_batches=2,measurement_workers=4,rollout_cache_mib=512,advantage_scale_floor=1.,ppo_clip=.2,
        games_per_round=args.games_per_round,validation_fraction=.2,policy_scope='shared',
        workers=args.workers,runners=args.runners,inference_processes=args.inference_processes,
        game_timeout=60,max_turns=50,eval_games=args.eval_games,eval_seed=args.eval_seed,
        milestones=args.milestones,policy_cache_mib=args.policy_cache_mib,actor_cache_mib=args.actor_cache_mib,
        numeric_disk_gib=128,policy_disk_gib=64,rollout_storage='pickle')
    if args.opponent_mode != 'legacy':
        config.update(opponent_mode=args.opponent_mode,
            opponent_script_fraction=1. if args.opponent_mode == 'scripted_control' else .2,
            opponent_inference_processes=args.opponent_inference_processes,
            opponent_snapshot_schedule='round_robin_150_200_plus_every_tenth_committed',
            pilot_branch=args.pilot_branch)
    pins = source_hashes()
    report_path = args.output/'report.json'
    root = args.output/'checkpoints'
    if args.resume:
        report = json.loads(report_path.read_text())
        if report['config']!=config or report['source_sha256']!=pins:
            raise ValueError('Resume requires the same source, schedule and learning recipe')
        with device_scope('gpu'):
            model,trainer,state = load_checkpoint(root,policy_type=StandardPolicy,gru_backend='cudnn-memory')
        completed = state['round']
        start_round = report.get('start_round', 0)
        if 'fork' in report:
            validate_parent_copy(report['fork'], root)
            if start_round != 200 or completed < start_round or args.opponent_mode == 'legacy':
                raise ValueError('Invalid checkpoint-200 fork history')
            if args.rounds != report['fork']['target_round'] or args.pilot_branch != report['fork']['pilot_branch']:
                raise ValueError('Fork target or pilot identity changed on resume')
            if completed == start_round:
                assert_fork_recipe(state['config'], config)
            elif state['config'] != config:
                raise ValueError('Branch checkpoint configuration changed')
        elif start_round or args.opponent_mode != 'legacy':
            raise ValueError('Unrecognized branch history')
        if len(report['rounds']) == completed-start_round-1:
            report['rounds'].append(state['round_result'])
        if len(report['rounds']) != completed-start_round:
            raise ValueError('Checkpoint/report history mismatch')
        if completed:
            p=args.output/'rollouts'/f'round-{completed:04d}'
            if p.exists():atomic_json(p/'committed.json',dict(round=completed,checkpoint=str(root/f'round-{completed:04d}')))
    elif args.fork_checkpoint is not None:
        parent = args.fork_checkpoint.resolve(strict=True)
        if parent.name != 'round-0200' or not parent.is_dir():
            raise ValueError('Fork parent must be the completed round-0200 checkpoint')
        parent_root = parent.parent
        latest = json.loads((parent_root/'latest.json').read_text())
        if latest.get('checkpoint') != parent.name:
            raise ValueError('Parent latest checkpoint is not round 200')
        parent_report_path = parent_root.parent/'report.json'
        parent_report = json.loads(parent_report_path.read_text())
        if parent_report.get('status') != 'complete' or parent_report.get('completed_games') != 200_000:
            raise ValueError('Parent 200k report has not completed')
        parent_hashes = checkpoint_hashes(parent)
        parent_state = json.loads((parent/'state.json').read_text())
        if parent_state.get('round') != 200 or len(parent_report.get('rounds', ())) != 200:
            raise ValueError('Parent checkpoint/report is not the 200-round result')
        assert_fork_recipe(parent_state['config'], config)
        target_round = 201 if args.pilot_branch else 250
        if args.rounds != target_round:
            raise ValueError('Fork target must be one pilot or fifty full rounds')
        root.mkdir()
        shutil.copytree(parent, root/parent.name)
        atomic_json(root/'latest.json', {'checkpoint':parent.name})
        fork = dict(parent_checkpoint=str(parent), parent_checkpoint_sha256=parent_hashes,
            parent_report=str(parent_report_path),parent_report_sha256=sha(parent_report_path),
            parent_source_sha256=parent_report['source_sha256'],
            previous_games=200_000, previous_rounds=200,
            target_round=target_round,pilot_branch=args.pilot_branch,
            migration='opponent_schedule_and_inference_only')
        validate_parent_copy(fork,root)
        with device_scope('gpu'):
            model,trainer,state = load_checkpoint(root,policy_type=StandardPolicy,gru_backend='cudnn-memory')
        completed = start_round = 200
        if state['round'] != completed or trainer.rng.bit_generator.state != state['trainer_rng_state']:
            raise AssertionError('Fork did not restore exact round and RNG')
        fork['restored_state'] = dict(
            actor_sha256=weights_digest(model.actor.get_weights()),
            critic_sha256=weights_digest(model.critic.get_weights()),
            actor_adam_sha256=weights_digest([np.asarray(v) for v in trainer.optimizer.variables]),
            critic_adam_sha256=weights_digest([np.asarray(v) for v in trainer.critic_optimizer.variables]),
            actor_adam_steps=int(trainer.optimizer.iterations),
            critic_adam_steps=int(trainer.critic_optimizer.iterations),
            trainer_rng_state=deepcopy(trainer.rng.bit_generator.state))
        report=dict(status='running',config=config,source_sha256=pins,start_round=200,
            rounds=[],evaluations={},started_at=datetime.now(timezone.utc).isoformat(),
            fork=fork,methodology='Checkpoint-200 controlled continuation; learner-only PPO '
                'against scripted control or predeclared frozen-snapshot self-play; '
                'whole-game 80/20 split and unchanged model, optimizer and PPO recipe.')
        atomic_json(report_path,report)
    else:
        with device_scope('gpu'):
            model=StandardPolicy.load(args.source).model
            configure_gru_backend(model,'cudnn-memory')
            before=weights_digest(model.get_weights())
            trainer=make_trainer(model,config)
        if before!=weights_digest(model.get_weights()) or int(trainer.optimizer.iterations) or int(trainer.critic_optimizer.iterations):
            raise AssertionError('PPO initialization changed weights or reused an optimizer')
        state=dict(round=0,config=config)
        source_state=args.source.with_name('state.json')
        report=dict(status='running',config=config,source_sha256=pins,rounds=[],evaluations={},
            started_at=datetime.now(timezone.utc).isoformat(),
            initialization=dict(source=str(args.source),source_sha256=sha(args.source),
                source_state_sha256=sha(source_state),actor_sha256=weights_digest(model.actor.get_weights()),
                critic_sha256=weights_digest(model.critic.get_weights()),fresh_optimizers=True),
            methodology='One Standard actor/critic against fixed scripted baselines; six balanced pairings; '
                'whole-game 80/20 split with rotating balanced pairing/role quotas; terminal +1/-1/0; fixed detached behavior memories; '
                '60-second and turn-50 draws; two optimized PPO epochs; fresh initial optimizers. '
                'Initialization actor is the 24k demonstration checkpoint with a supervised value bootstrap.')
        save_checkpoint(root,model,trainer,state,policy_type=StandardPolicy)
        completed=start_round=0
        atomic_json(report_path,report)
    assert_model_device(model,'gpu')
    place_trainer(trainer,'gpu')
    pipeline=resolve_update_pipeline('gpu','gpu',policy_cache_mib=args.policy_cache_mib,
        actor_epoch_cache_mib=args.actor_cache_mib,compact_inputs=args.compact_input_cache)
    apply_update_pipeline(trainer,pipeline)
    trainer.numeric_disk_limit_bytes=128*1024**3
    trainer.policy_disk_limit_bytes=64*1024**3
    host_memory = HostMemoryReclaimer() if args.release_host_memory else None
    report.update(status='running',requested_rounds=args.rounds,active_round=completed+1,
                  runtime_pipeline=pipeline,
                  runtime_memory=dict(release_host_memory=host_memory is not None,
                      boundaries=['after_update','after_validation'] if host_memory else []))
    atomic_json(report_path,report)
    # Validation replays complete private seat streams with current CPU weights.
    with device_scope('cpu'):
        validation_model=keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(model))
        validation_model(example_batch(model.memory_size,model.feature_spec),training=False)
        validator=GameValidator(validation_model,batch_size=32)
    opponent_model = None
    snapshot_cache = {}
    active_opponent = None
    if args.opponent_mode == 'frozen_pool':
        parent_root = Path(report['fork']['parent_checkpoint']).parent
        parent_policy = root/'round-0200'/'policy.keras'
        with device_scope('cpu'):
            opponent_model = StandardPolicy.load(parent_policy).model
        assert_model_device(opponent_model,'cpu')
        snapshot_cache[200] = dict(path=str(parent_policy), sha256=sha(parent_policy),
            weights=[np.asarray(w).copy() for w in opponent_model.get_weights()],
            weights_sha256=weights_digest(opponent_model.get_weights()))
        report.setdefault('opponent_plan', {str(number):snapshot_round_for_round(number)
            for number in range(201,251)})
        report.setdefault('opponent_checkpoints', {})
        for old in (150,200):
            path = (parent_root if old == 150 else root)/f'round-{old:04d}'/'policy.keras'
            digest = sha(path)
            previous = report['opponent_checkpoints'].setdefault(str(old),
                dict(path=str(path),sha256=digest))
            if previous != dict(path=str(path),sha256=digest):
                raise ValueError('Frozen parent opponent checkpoint changed')
        for old, pinned in report['opponent_checkpoints'].items():
            if sha(pinned['path']) != pinned['sha256']:
                raise ValueError(f'Frozen opponent checkpoint {old} changed')
        atomic_json(report_path,report)
        collectors=[StandardSelfplayCollector(model,opponent_model,
            workers=args.workers//args.runners,
            inference_processes=args.inference_processes//args.runners,
            opponent_inference_processes=args.opponent_inference_processes//args.runners)
            for _ in range(args.runners)]
    else:
        collectors=[ProcessGameCollector(model,workers=args.workers//args.runners,
            inference_processes=args.inference_processes//args.runners,inference_device='cpu')
            for _ in range(args.runners)]

    def persist_progress(number,done,total):
        atomic_json(args.output/'progress.json',dict(round=number,phase='collection',completed=done,total=total))
        print(json.dumps(dict(event='collection',round=number,completed=done,total=total)),flush=True)

    async def collect(planned,number,directory,*,record=True,greedy=False):
        result_root=directory/'results';result_root.mkdir(parents=True,exist_ok=True)
        saved=[json.loads(p.read_text()) for p in result_root.glob('*.json')]
        ids={g['index'] for g in planned}
        if any(g['index'] not in ids for g in saved):raise ValueError('Foreign scheduled result')
        planned_by_id={g['index']:g for g in planned}
        if any(any(row.get(k) != v for k,v in planned_by_id[row['index']].items()) for row in saved):
            raise ValueError('Saved game differs from the pinned opponent schedule')
        seen={g['index'] for g in saved};remaining=[g for g in planned if g['index'] not in seen]
        def progress(row):
            row=dict(row,actions=None)
            atomic_json(result_root/f"game-{row['index']:06d}.json",row)
            saved.append(row)
            if len(saved)%100==0:persist_progress(number,len(saved),len(planned))
        jobs=[dict(seed=g['seed'],index=g['index'],game=g) for g in remaining]
        options=dict(game_runner='standard_ppo',output=directory/'trajectories' if record else None,
            diagnostics=(args.output/'adjudications'/f'round-{number:04d}') if record else directory/'adjudications',
            game_timeout=60,max_turns=50,greedy=greedy)
        if args.opponent_mode == 'frozen_pool':
            if active_opponent is None:
                raise RuntimeError('Frozen opponent must be pinned before collection')
            options.update(opponent_version=active_opponent['round'],
                opponent_sha256=active_opponent['sha256'])
        await asyncio.gather(*(c.run_schedule(jobs[i::len(collectors)],progress=progress,
            **options)
            for i,c in enumerate(collectors)))
        return sorted(saved,key=lambda g:g['index'])

    async def evaluate(number):
        key=str(number)
        point=report['evaluations'].setdefault(key,dict(cells=[],policy_sha256=weights_digest(model.actor.get_weights())))
        if point['policy_sha256']!=weights_digest(model.actor.get_weights()):raise ValueError('Evaluation policy changed')
        for deck in STANDARD_DECKS:
            for opponent in STANDARD_DECKS:
                for greedy in (False,True):
                    if any(c['deck']==deck and c['opponent']==opponent and c['greedy']==greedy for c in point['cells']):continue
                    directory=args.output/'evaluation'/f'round-{number:04d}'/f'{deck}-vs-{opponent}-{int(greedy)}'
                    atomic_json(args.output/'progress.json',dict(round=number,phase='evaluation',
                        deck=deck,opponent=opponent,greedy=greedy,completed_cells=len(point['cells']),total_cells=18))
                    planned=[]
                    pair=sorted((deck,opponent),key=list(STANDARD_DECKS).index)
                    for i in range(args.eval_games):
                        seat=i%2;decks=[opponent,opponent];decks[seat]=deck
                        planned.append(dict(index=i,seed=args.eval_seed+i,pairing=pair,decks=decks,
                            learner_seat=seat,learner_deck=deck,opponent_deck=opponent,
                            starting_player=f'P{(i//2)%2}',partition='evaluation'))
                    rows=await collect(planned,number,directory,record=False,greedy=greedy)
                    if any(g['status'] not in ('win','loss','draw') for g in rows):raise ValueError('Evaluation game failed')
                    point['cells'].append(dict(deck=deck,opponent=opponent,greedy=greedy,
                        summary=evaluation_summary(rows),games=compact_games(rows)))
                    atomic_json(report_path,report)
                    print(json.dumps(dict(event='evaluation',round=number,deck=deck,opponent=opponent,
                        greedy=greedy,summary=point['cells'][-1]['summary'])),flush=True)
        point['complete']=True
        # The unchanged imitation actor is the round-zero control on these same seeds.
        if number:
            initial=report['evaluations']['0']['cells']
            for cell in point['cells']:
                reference=next(c for c in initial if (c['deck'],c['opponent'],c['greedy'])==(cell['deck'],cell['opponent'],cell['greedy']))
                cell['winrate_change_from_imitation_pp']=100*(cell['summary']['win_rate']-reference['summary']['win_rate'])
        atomic_json(report_path,report)

    def frozen_state():
        return (weights_digest(model.get_weights()),weights_digest([np.asarray(v) for v in trainer.optimizer.variables]),
                weights_digest([np.asarray(v) for v in trainer.critic_optimizer.variables]),deepcopy(trainer.rng.bit_generator.state))

    def release_host_memory(stage):
        if host_memory is None:
            return None
        before = frozen_state()
        result = host_memory.release(stage)
        if frozen_state()!=before:
            raise AssertionError('Host allocator release changed model, optimizers or RNG')
        result['model_both_optimizers_and_rng_unchanged'] = True
        print(json.dumps(dict(event='host_memory_release', **result)),flush=True)
        return result

    def verify_behavior(rows):
        checked=0;max_probability_error=max_value_error=0.
        for g in rows[::max(1,len(rows)//24)]:
            for t in g['trajectories']:
                stream=iter(read_trajectory(t['path']))
                for _ in range(2):
                    sample=next(stream,None)
                    if sample is None:break
                    from patchwork_automaton.neural.features import collate
                    inputs=collate([sample.arrays]);inputs.update(memory_in=sample.memory[None,:],advance_memory=np.ones((1,1),np.float32))
                    with device_scope('gpu'):out=model(inputs,training=False)
                    logits=np.asarray(out['logits'])[0].astype(np.float64)
                    logp=logits-logits.max();logp-=np.log(np.exp(logp).sum())
                    np.testing.assert_allclose(logp,sample.log_probs,rtol=5e-4,atol=2e-4)
                    np.testing.assert_allclose(np.asarray(out['value'])[0,0],sample.value,rtol=5e-4,atol=2e-4)
                    max_probability_error=max(max_probability_error,float(np.max(abs(logp-sample.log_probs))))
                    max_value_error=max(max_value_error,abs(float(np.asarray(out['value'])[0,0])-sample.value));checked+=1
        return dict(samples=checked,max_log_probability_error=max_probability_error,max_value_error=max_value_error)

    async def pin_opponent(number):
        nonlocal active_opponent
        if args.opponent_mode != 'frozen_pool':
            return None
        selected = snapshot_round_for_round(number)
        if report['opponent_plan'].get(str(number)) != selected:
            raise ValueError('Opponent selection changed after fork')
        base = parent_root if selected == 150 else root
        path = base/f'round-{selected:04d}'/'policy.keras'
        file_digest = sha(path)
        checkpoint_record = dict(path=str(path), sha256=file_digest)
        previous = report['opponent_checkpoints'].setdefault(str(selected), checkpoint_record)
        if previous != checkpoint_record:
            raise ValueError('Selected opponent checkpoint changed')
        if selected not in snapshot_cache:
            before = frozen_state()
            with device_scope('cpu'):
                loaded = StandardPolicy.load(path).model
                weights = [np.asarray(w).copy() for w in loaded.get_weights()]
            snapshot_cache[selected] = dict(path=str(path),sha256=file_digest,
                weights=weights,weights_sha256=weights_digest(weights))
            del loaded
            gc.collect()
            if frozen_state() != before:
                raise AssertionError('Opponent loading changed learner state')
        cached = snapshot_cache[selected]
        if cached['sha256'] != file_digest:
            raise ValueError('Cached opponent checkpoint changed')
        before = frozen_state()
        with device_scope('cpu'):
            opponent_model.set_weights(cached['weights'])
        if weights_digest(opponent_model.get_weights()) != cached['weights_sha256']:
            raise AssertionError('Opponent weights differ from pinned checkpoint')
        if frozen_state() != before:
            raise AssertionError('Opponent selection changed learner state')
        if any(c.opponent_version != selected or c.opponent_digest != cached['weights_sha256']
                for c in collectors):
            await asyncio.gather(*(c.sync_opponent(opponent_model, selected) for c in collectors))
        if any(c.opponent_digest != cached['weights_sha256'] for c in collectors):
            raise AssertionError('Opponent inference replicas differ from pinned checkpoint')
        active_opponent = dict(round=selected,path=str(path),sha256=file_digest,
                               weights_sha256=cached['weights_sha256'])
        previous_round = report.setdefault('opponent_rounds',{}).setdefault(str(number), active_opponent)
        if previous_round != active_opponent:
            raise ValueError('Opponent assignment differs from saved round')
        atomic_json(report_path,report)
        return active_opponent

    try:
        if args.opponent_mode == 'frozen_pool':
            await asyncio.gather(*(c.start(version=completed, opponent_version=200)
                                   for c in collectors))
        else:
            await asyncio.gather(*(c.start(version=completed) for c in collectors))
        if start_round == 0 and not report['evaluations'].get('0',{}).get('complete'):
            await evaluate(0)
        if completed in args.milestones and not report['evaluations'].get(str(completed),{}).get('complete'):
            await evaluate(completed)
        for number in range(completed+1,args.rounds+1):
            report.update(status='running',active_round=number)
            atomic_json(report_path,report)
            directory=args.output/'rollouts'/f'round-{number:04d}'
            planned=schedule(args.games_per_round,seed=args.seed,round_id=number)
            if args.opponent_mode == 'frozen_pool':
                pinned = await pin_opponent(number)
                planned = assign_opponents(planned,round_id=number,mode='frozen_pool',
                    checkpoint_round=pinned['round'],checkpoint_sha256=pinned['sha256'])
            elif args.opponent_mode == 'scripted_control':
                planned = assign_opponents(planned,round_id=number,mode='scripted')
            __import__('tensorflow').config.experimental.reset_memory_stats('GPU:0')
            with Resources(args.output) as resources:
                start=time.monotonic()
                rows=await collect(planned,number,directory)
                validate_collection(rows,planned,number-1)
                collection_seconds=time.monotonic()-start
                before=frozen_state()
                reproduction=verify_behavior(rows)
                if frozen_state()!=before:raise AssertionError('Behavior verification changed training state')
                training=[g for g in rows if g['partition']=='training']
                validation=[g for g in rows if g['partition']=='validation']
                norm=normalization(training)
                trajectories=[t for g in training for t in g['trajectories']]
                rollout_bytes=sum(p.stat().st_size for p in (directory/'trajectories').glob('*'))
                if shutil.disk_usage(args.output).free < 200*1024**3:
                    raise RuntimeError('Insufficient free disk for declared 128+64 GiB update caches')
                atomic_json(args.output/'progress.json',dict(round=number,phase='gpu_update'))
                def update_progress(event):
                    atomic_json(args.output/'progress.json',dict(round=number,phase='gpu_update',update=event))
                    print(json.dumps(dict(event='update_progress',round=number,update=event)),flush=True)
                trainer.progress_callback=update_progress
                trainer.update_timeout_seconds=args.update_timeout
                report['runtime_limits']=dict(update_timeout_seconds=args.update_timeout,
                    timeout_behavior='stop update without committing; retain games and prior checkpoint')
                atomic_json(report_path,report)
                start=time.monotonic()
                update=await asyncio.to_thread(trainer.update_files,trajectories)
                update_seconds=time.monotonic()-start
                memory_releases = [release_host_memory('after_update')] if host_memory else []
                before=frozen_state()
                with device_scope('cpu'):validation_model.set_weights(model.get_weights())
                def validate():
                    with device_scope('cpu'):return validator.evaluate(validation,norm)
                atomic_json(args.output/'progress.json',dict(round=number,phase='validation'))
                start=time.monotonic();heldout=await asyncio.to_thread(validate);validation_seconds=time.monotonic()-start
                if frozen_state()!=before:raise AssertionError('Validation changed weights, optimizers or RNG')
                if host_memory:memory_releases.append(release_host_memory('after_validation'))
                gpu=__import__('tensorflow').config.experimental.get_memory_info('GPU:0')
            row=dict(round=number,games=compact_games(rows),training_games=len(training),validation_games=len(validation),
                behavior_version=number-1,behavior_weights_sha256=collectors[0].policy_digest,
                behavior_reproduction=reproduction,collection_seconds=collection_seconds,
                update_seconds=update_seconds,validation_seconds=validation_seconds,update=update,validation=heldout,
                validation_read_only=True,rollout_bytes=rollout_bytes,resources=resources.result(),gpu_memory=gpu,
                host_memory_releases=memory_releases,
                inference=[c.stats() for c in collectors])
            if args.opponent_mode != 'legacy':
                row['opponent_mix'] = dict(Counter(g['opponent_kind'] for g in rows))
                if active_opponent is not None:
                    row['frozen_opponent'] = active_opponent
            start=time.monotonic()
            checkpoint=save_checkpoint(root,model,trainer,dict(round=number,config=config,round_result=row),policy_type=StandardPolicy)
            row['checkpoint_seconds']=time.monotonic()-start
            atomic_json(directory/'committed.json',dict(round=number,checkpoint=str(checkpoint)))
            report['rounds'].append(row)
            atomic_json(report_path,report)
            removed=cleanup_rollouts(args.output/'rollouts',number)
            # Only raw diagnostic examples are capped; all seed/outcome JSON remains.
            raw=sorted((args.output/'adjudications').glob('round-*/game-*/*.pkl.gz'))
            for p in raw[8:]:p.unlink()
            await asyncio.gather(*(c.sync_weights(model,number) for c in collectors))
            print(json.dumps(dict(event='round_complete',round=number,collection_seconds=collection_seconds,
                update_seconds=update_seconds,validation_seconds=validation_seconds,rollout_bytes=rollout_bytes,
                removed_rounds=removed,win_rate=sum(g.get('learner_won',False) for g in rows)/len(rows))),flush=True)
            if number in args.milestones:await evaluate(number)
        report.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),
            completed_games=args.rounds*args.games_per_round,
            branch_games=(args.rounds-start_round)*args.games_per_round)
        if source_hashes()!=pins:raise AssertionError('Frozen source changed')
        atomic_json(report_path,report)
        atomic_json(args.output/'progress.json',dict(phase='complete',completed_games=report['completed_games']))
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        atomic_json(report_path,report);raise
    finally:
        await asyncio.gather(*(c.close() for c in collectors))
        lock.close()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,
                   help='Imitation policy for a fresh run; omitted for a checkpoint fork')
    p.add_argument('--fork-checkpoint',type=Path,
                   help='Immutable completed round-0200 directory to copy into a new branch')
    p.add_argument('--opponent-mode',choices=('legacy','scripted_control','frozen_pool'),
                   default='legacy')
    p.add_argument('--opponent-inference-processes',type=int,default=2,
                   help='Total frozen-opponent CPU replicas, divided among runners')
    p.add_argument('--pilot-branch',action='store_true',
                   help='Run one isolated, complete 1,000-game self-play PPO round')
    p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--games-per-round',type=int,default=1000)
    p.add_argument('--workers',type=int,default=24)
    p.add_argument('--runners',type=int,default=2)
    p.add_argument('--inference-processes',type=int,default=None,
                   help='Total policy CPU replicas; default 2 for self-play, 4 otherwise; resume preserves saved count')
    p.add_argument('--seed',type=int,default=402610300000)
    p.add_argument('--eval-seed',type=int,default=802610300000)
    p.add_argument('--eval-games',type=int,default=400)
    p.add_argument('--milestones',type=int,nargs='*',default=list(range(10,101,10)))
    p.add_argument('--policy-cache-mib',type=int,default=2048)
    p.add_argument('--actor-cache-mib',type=int,default=2048)
    p.add_argument('--compact-input-cache',action='store_true',
                   help='Losslessly cache zero-heavy input tensors compactly on GPU')
    p.add_argument('--release-host-memory',action='store_true',
                   help='Collect unreachable cycles and return free glibc pages after updates and validation')
    p.add_argument('--update-timeout',type=float,default=1800,
                   help='PPO update wall-clock budget, checked at batch boundaries (seconds)')
    p.add_argument('--resume',action='store_true')
    return p


if __name__=='__main__':asyncio.run(run(parser().parse_args()))
