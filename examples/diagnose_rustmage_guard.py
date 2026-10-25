"""Replay one saved PPO update, auditing guard proposals without new games.

Run from the project root with the prepared TensorFlow/CUDA environment.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '4')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Completed pilot output directory')
    p.add_argument('--output', type=Path, required=True, help='New independent diagnostic directory')
    p.add_argument('--round', type=int, default=1, help='Only the first update is supported')
    p.add_argument('--device', choices=('cpu', 'gpu'), default='gpu')
    p.add_argument('--validation-device', choices=('cpu', 'gpu'), default='cpu')
    p.add_argument('--max-state-kl', type=float, help='Override only this guard limit; omitted: exact control')
    p.add_argument('--target-kl', type=float, help='Optional mean KL override, recorded explicitly')
    p.add_argument('--save-checkpoint', action='store_true')
    return p


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def exact_checkpoint_root(checkpoint):
    """Use the normal verified loader without touching the source latest.json."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    with tempfile.TemporaryDirectory(prefix='rustmage-guard-checkpoint-') as temporary:
        root = Path(temporary)
        (root/checkpoint.name).symlink_to(checkpoint, target_is_directory=True)
        (root/'latest.json').write_text(json.dumps({'checkpoint':checkpoint.name}))
        yield root


def recorded_partitions(report, source, round_id=1):
    """Preserve recorded whole-game membership and original trajectory order."""
    if round_id != 1:
        raise ValueError('This diagnostic supports only the first on-policy update')
    row = next(r for r in report['rounds'] if r['round'] == round_id)
    training = set(row['split']['training_indices'])
    heldout = set(row['split']['validation_indices'])
    if (training & heldout or len(training)+len(heldout) != len(row['games'])
            or len({g['index'] for g in row['games']}) != len(row['games'])):
        raise ValueError('Invalid saved whole-game partition')
    games = []
    for original in sorted(row['games'], key=lambda g:g['index']):
        game = dict(original, trajectories=[])
        if game['index'] not in training | heldout:
            raise ValueError('Game missing from saved partition')
        for trajectory in original.get('trajectories', ()):
            item = dict(trajectory)
            path = Path(item['path'])
            if not path.is_file():
                # A copied pilot keeps unique rollout basenames even when the
                # report's original project-relative prefix has moved.
                path = Path(source)/'rollouts'/path.name
            if not path.is_file():
                raise FileNotFoundError(path)
            item['path'] = str(path.resolve())
            game['trajectories'].append(item)
        games.append(game)
    return ([g for g in games if g['index'] in training],
            [g for g in games if g['index'] in heldout], row)


def compare_arrays(expected, actual):
    import numpy as np
    if len(expected) != len(actual):
        return dict(same_shapes=False, bitwise_equal=False, expected_arrays=len(expected), actual_arrays=len(actual))
    same = all(np.asarray(a).shape == np.asarray(b).shape for a,b in zip(expected,actual))
    if not same:
        return dict(same_shapes=False, bitwise_equal=False)
    differences = [np.abs(np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))
                   for a,b in zip(expected,actual)]
    return dict(same_shapes=True,
        bitwise_equal=all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(expected,actual)),
        allclose_1e_7=all(np.allclose(np.asarray(a),np.asarray(b),rtol=1e-7,atol=1e-7) for a,b in zip(expected,actual)),
        max_absolute_difference=max((float(d.max()) for d in differences if d.size),default=0.))


def guard_summary(events):
    rejected = [event for event in events if not event['accepted']]
    reasons = Counter(reason for event in rejected for reason in event.get('rejection_reasons', ()))
    max_vetoes = sum('max_state_kl' in event.get('rejection_reasons', ()) for event in rejected)
    return dict(proposals=len(events), accepted_proposals=len(events)-len(rejected),
        rejected_proposals=len(rejected), rejection_reasons=dict(reasons),
        max_state_vetoes=max_vetoes,
        max_state_veto_fraction=max_vetoes/len(rejected) if rejected else 0.,
        max_state_veto_dominates=bool(rejected) and max_vetoes/len(rejected) > .5,
        scope='Guard/optimizer diagnostic on one frozen on-policy rollout; no playing-strength conclusion')


def run(args):
    from patchwork_automaton.neural.devices import configure_runtime, device_scope, assert_model_device
    configure_runtime(args.device, hide_gpu=args.device == 'cpu' and args.validation_device == 'cpu')
    if args.round != 1:
        raise ValueError('Only --round 1 is supported')
    for value in (args.max_state_kl,args.target_kl):
        if value is not None and not 0 < value < float('inf'):
            raise ValueError('KL guard overrides must be finite and positive')
    source = args.source.resolve()
    source_report_path = source/'report.json'
    original = json.loads(source_report_path.read_text())
    train, heldout, original_round = recorded_partitions(original, source, args.round)
    trajectories = [t for game in train for t in game.get('trajectories', ())]
    if not trajectories or not any(g.get('trajectories') for g in heldout):
        raise ValueError('Both recorded partitions need terminal games')
    checkpoint = source/'checkpoints'/'round-0000'
    reference = source/'checkpoints'/'round-0001'
    # Refuse source ancestors/descendants so every diagnostic is independent.
    output = args.output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Diagnostic output must be outside the source experiment')
    output.mkdir(parents=True,exist_ok=False)
    import keras
    import numpy as np
    from examples.train_rustmage import array_digest, neural_call, place_trainer
    from patchwork_automaton.neural.batching import example_batch
    from patchwork_automaton.neural.experiment import atomic_json, load_checkpoint, save_checkpoint
    from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
    from patchwork_automaton.neural.validation import GameValidator, training_normalization
    started = time.monotonic()
    source_files = [source_report_path,*sorted(checkpoint.iterdir()),*sorted(reference.iterdir())]
    source_hashes = {str(p):sha256_file(p) for p in source_files if p.is_file()}
    input_hashes = {t['path']:sha256_file(t['path'])
        for game in train+heldout for t in game.get('trajectories', ())}
    keras.utils.set_random_seed(original['config']['seed'])
    with device_scope(args.device), exact_checkpoint_root(checkpoint) as root:
        model,trainer,state = load_checkpoint(root)
        if state['round'] != 0 or not getattr(model,'separate_value',False):
            raise ValueError('Expected warmed separate-actor/critic checkpoint zero')
        if state.get('critic_warmup',{}).get('status') != 'complete':
            raise ValueError('Source checkpoint zero has no completed critic warm-up')
        placements = assert_model_device(model,args.device)
        traces = place_trainer(trainer,args.device)
    trainer.guard_audit = True
    if args.max_state_kl is not None: trainer.max_state_kl = args.max_state_kl
    if args.target_kl is not None: trainer.target_kl = args.target_kl
    effective_config = dict(state['config'], guard_audit=True,
        max_state_kl=trainer.max_state_kl, target_kl=trainer.target_kl)
    def digests():
        return dict(model=array_digest(model.weights), actor=array_digest(model.actor.weights),
            critic=array_digest(model.critic.weights), actor_optimizer=array_digest(trainer.optimizer.variables),
            critic_optimizer=array_digest(trainer.critic_optimizer.variables),
            trainer_rng=hashlib.sha256(json.dumps(trainer.rng.bit_generator.state,sort_keys=True).encode()).hexdigest())
    report = dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source),source_round=1,source_sha256=source_hashes,input_sha256=input_hashes,
        source_config=state['config'],effective_config=effective_config,
        current_source_sha256={str(p):sha256_file(p) for p in [Path(__file__),Path('examples/train_rustmage.py'),
            *Path('src/patchwork_automaton/neural').glob('*.py')]},
        device=args.device,validation_device=args.validation_device,variable_placements=placements,
        graph_traces=traces,host=platform.platform(),
        runtime={key:os.environ.get(key) for key in ['CUDA_VISIBLE_DEVICES','TF_NUM_INTRAOP_THREADS',
            'TF_NUM_INTEROP_THREADS','TF_ENABLE_ONEDNN_OPTS']},
        split=original_round['split'],training_trajectories=len(trajectories),
        initial_digests=digests(),initial_optimizer_steps=int(trainer.optimizer.iterations),
        initial_critic_optimizer_steps=int(trainer.critic_optimizer.iterations))
    atomic_json(output/'report.json',report)
    validation_model = model
    if args.validation_device != args.device:
        with device_scope(args.validation_device):
            validation_model = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(model))
            validation_model(example_batch(model.memory_size,model.feature_spec),training=False)
            validation_model.set_weights(model.get_weights())
            assert_model_device(validation_model,args.validation_device)
    with device_scope(args.validation_device):
        validator = GameValidator(validation_model,batch_size=state['config'].get('validation_batch',32),
                                  clip_ratio=trainer.clip_ratio)
    normalization = training_normalization(train,trainer.advantage_scale_floor)
    def validate():
        if validation_model is not model:
            neural_call(args.validation_device,validation_model.set_weights,model.get_weights())
        return {name:neural_call(args.validation_device,validator.evaluate,games,normalization)
            for name,games in [('training',train),('validation',heldout)]}
    report['generalization_before'] = validate()
    atomic_json(output/'report.json',report)
    events=[]
    try:
        with (output/'guard-proposals.jsonl').open('x') as stream:
            def observe(event):
                events.append(event)
                stream.write(json.dumps(event)+'\n');stream.flush();os.fsync(stream.fileno())
                print(json.dumps(dict(stage='guard-proposal',**event)),flush=True)
            trainer.guard_observer = observe
            update_started=time.monotonic()
            report['update']=neural_call(args.device,trainer.update_files,trajectories)
            report['update_seconds']=time.monotonic()-update_started
        report['guard_summary']=guard_summary(events)
        report['final_digests']=digests()
        report['generalization_after']=validate()
        with device_scope(args.device),exact_checkpoint_root(reference) as root:
            expected_model,expected_trainer,expected_state=load_checkpoint(root)
        report['comparison_to_original_round1']=dict(
            limits_unchanged=(trainer.max_state_kl == state['config'].get('max_state_kl',.1)
                and trainer.target_kl == state['config'].get('target_kl',.02)),
            actor=compare_arrays(expected_model.actor.weights,model.actor.weights),
            actor_optimizer=compare_arrays(expected_trainer.optimizer.variables,trainer.optimizer.variables),
            critic=compare_arrays(expected_model.critic.weights,model.critic.weights),
            critic_optimizer=compare_arrays(expected_trainer.critic_optimizer.variables,trainer.critic_optimizer.variables),
            rng_equal=expected_trainer.rng.bit_generator.state == trainer.rng.bit_generator.state,
            expected_update={k:original_round['update'][k] for k in ['accepted_steps','rejected_epochs','early_stop']},
            actual_update={k:report['update'][k] for k in ['accepted_steps','rejected_epochs','early_stop']})
        if args.save_checkpoint:
            report['checkpoint']=str(save_checkpoint(output/'checkpoints',model,trainer,
                dict(round=1,config=effective_config,diagnostic_source=str(checkpoint)),policy_type=RustMagePolicy))
        report['source_unchanged']=all(sha256_file(p)==digest for p,digest in source_hashes.items())
        report['inputs_unchanged']=all(sha256_file(p)==digest for p,digest in input_hashes.items())
        if not report['source_unchanged'] or not report['inputs_unchanged']:
            raise RuntimeError('Source checkpoint/report or rollout changed during diagnostic')
        report.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started)
        atomic_json(output/'report.json',report)
        print(json.dumps(dict(stage='diagnostic-complete',report=str(output/'report.json'),
            guard_summary=report['guard_summary'],comparison=report['comparison_to_original_round1'])),flush=True)
        return report
    except BaseException as error:
        report.update(status='error',error=dict(type=type(error).__name__,message=str(error)),
                      guard_summary=guard_summary(events))
        atomic_json(output/'report.json',report)
        raise


if __name__ == '__main__':
    run(parser().parse_args())
