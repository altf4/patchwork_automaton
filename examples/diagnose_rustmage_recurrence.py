"""Audit frozen-memory fitting against recurrent replay of saved RustMage games.

Uses trusted local trajectories only. No game collection, actor updates, or
checkpoint writes. Replayed decisions retain the recorded observations/actions;
this measures network consistency, not the strength of a new rollout policy.
Optional critic fits hold out whole games (both seats), with terminal targets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.update(KERAS_BACKEND='tensorflow', CUDA_VISIBLE_DEVICES='-1',
                  TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='4',
                  TF_ENABLE_ONEDNN_OPTS='0', TF_CPP_MIN_LOG_LEVEL='3')

import keras
import numpy as np
import tensorflow as tf

from patchwork_automaton.neural.batching import example_batch, input_signature
from patchwork_automaton.neural.experiment import atomic_json, make_trainer, read_trajectory
from patchwork_automaton.neural.features import collate
from patchwork_automaton.neural.rustmage_policy import RustMagePolicy


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads(args.report.read_text())
    round_data = next(r for r in report['rounds'] if r['round'] == args.round)
    if args.fit and not 0 < args.heldout_games < len(round_data['games']):
        raise ValueError('Hold out fewer than all games, keeping both seats together')
    root = args.report.parent / 'checkpoints'
    before = root / f'round-{args.round - 1:04d}'
    after = root / f'round-{args.round:04d}'
    sources = [args.report, before / 'policy.keras', after / 'policy.keras',
               before / 'critic-optimizer.npz', root / 'round-0000/policy.keras']
    hashes = {str(p): digest(p) for p in sources}
    samples, targets, game_ids, stages, sequences = [], [], [], [], []
    for gi, game in enumerate(round_data['games']):
        for seat, trajectory in enumerate(game['trajectories']):
            expected = 0. if game['winner'] is None else (1. if seat == game['winner'] else -1.)
            assert expected == trajectory['reward'] == game['terminal_rewards'][seat]
            chunk = list(read_trajectory(trajectory['path']))
            assert len(chunk) == trajectory['samples'] and all(s.advance for s in chunk)
            sequences.append(np.arange(len(samples), len(samples) + len(chunk)))
            samples.extend(chunk)
            targets.extend([expected] * len(chunk))
            game_ids.extend([gi] * len(chunk))
            stages.extend(np.linspace(0, 1, len(chunk)))
    targets = np.asarray(targets, np.float32)
    game_ids, stages = np.asarray(game_ids), np.asarray(stages)
    all_indices = np.arange(len(samples))
    behavior_values = np.array([s.value for s in samples])

    def batch(indices, memory=None, zero=False):
        chunk = [samples[i] for i in indices]
        arrays = collate([s.arrays for s in chunk])
        arrays['memory_in'] = np.stack([s.memory for s in chunk]) if memory is None else memory
        if zero:
            arrays['memory_in'] = np.zeros_like(arrays['memory_in'])
        arrays['advance_memory'] = np.ones((len(chunk), 1), np.float32)
        return arrays

    def stats(predictions, indices=all_indices):
        p, y = np.asarray(predictions, np.float64), targets[indices].astype(np.float64)
        variance = np.var(y)
        covariance = np.mean((p-p.mean())*(y-y.mean()))
        pred_variance = np.var(p)
        result = dict(samples=len(y), mse=float(np.mean((p-y)**2)),
            explained_variance=float(1-np.var(p-y)/variance),
            prediction_mean=float(p.mean()), prediction_std=float(p.std()),
            target_mean=float(y.mean()), prediction_min=float(p.min()), prediction_max=float(p.max()),
            fraction_outside_target_range=float(np.mean(np.abs(p)>1)),
            outcome_correlation=float(covariance/np.sqrt(variance*pred_variance)) if pred_variance else None,
            clipped_mse=float(np.mean((np.clip(p,-1,1)-y)**2)),
            hindsight_linear_scale=float(covariance/pred_variance) if pred_variance else None,
            raw_advantage_variance=float(np.var(y-p)), zero_baseline_variance=float(variance),
            quartile_mse=[float(np.mean((p[(stages[indices]>=q/4)&(stages[indices]<=(q+1)/4)]-
                y[(stages[indices]>=q/4)&(stages[indices]<=(q+1)/4)])**2)) for q in range(4)])
        return result

    model = RustMagePolicy.load(before / 'policy.keras').model
    assert model.separate_value
    signature = input_signature(example_batch(model.memory_size, model.feature_spec))

    @tf.function(input_signature=[signature], autograph=False)
    def predict(inputs):
        output = model(inputs, training=False)
        return dict(value=output['value'][:, 0], logp=tf.nn.log_softmax(output['logits']),
                    memory=output['memory_out'])

    @tf.function(input_signature=[signature], autograph=False)
    def predict_critic(inputs):
        output = model.value_output(inputs, training=False)
        return dict(value=output['value'][:, 0], memory=output['memory_out'])

    def evaluate(indices=all_indices, *, replay=False, zero=False, actor=False):
        values = np.empty(len(samples), np.float64)
        kl = np.empty(len(samples), np.float64)
        differences = np.empty((len(samples), 2), np.float64)
        def consume(ix, memory=None):
            arrays = batch(ix, memory, zero)
            out = {k: np.asarray(v) for k, v in (predict(arrays) if actor else predict_critic(arrays)).items()}
            values[ix] = out['value']
            if actor:
                for j, i in enumerate(ix):
                    old = samples[i].log_probs
                    new = out['logp'][j, :len(old)]
                    kl[i] = max(0., float(np.sum(np.exp(old)*(old-new))))
            if memory is not None:
                old = np.stack([samples[i].memory for i in ix])
                differences[ix, 0] = np.sqrt(np.mean((memory[:, :model.actor_memory]-old[:, :model.actor_memory])**2, axis=1))
                differences[ix, 1] = np.sqrt(np.mean((memory[:, model.actor_memory:]-old[:, model.actor_memory:])**2, axis=1))
            return out['memory']
        if replay:
            wanted = set(indices.tolist())
            selected = [seq for seq in sequences if len(seq) and seq[0] in wanted]
            assert sum(map(len, selected)) == len(indices), 'Replay requires whole trajectories'
            for start in range(0, len(selected), args.batch_size):
                streams = selected[start:start+args.batch_size]
                memory = np.zeros((len(streams), model.memory_size), np.float32)
                for step in range(max(map(len, streams))):
                    active = np.array([j for j,s in enumerate(streams) if step<len(s)])
                    ix = np.array([streams[j][step] for j in active])
                    updated = consume(ix, memory[active])
                    if actor:
                        memory[active] = updated
                    else:
                        memory[active, model.actor_memory:] = updated
        else:
            for start in range(0, len(indices), args.batch_size):
                consume(indices[start:start+args.batch_size])
        summary = stats(values[indices], indices)
        if actor:
            summary.update(mean_action_kl=float(kl[indices].mean()), max_action_kl=float(kl[indices].max()),
                           fraction_state_kl_above_point1=float(np.mean(kl[indices]>.1)))
        if replay:
            summary.update(actor_memory_rms=float(differences[indices, 0].mean()) if actor else None,
                           critic_memory_rms=float(differences[indices, 1].mean()))
        return summary, values[indices]

    result = dict(scope=__doc__, round=args.round, games=len(round_data['games']), samples=len(samples),
                  config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  fitting=dict(learning_rate=report['config']['learning_rate'], shuffle_seed=991,
                               measurements_at_epochs=[0., .25, .5, 1., 2.], ridge_penalty=10.),
                  source_sha256=hashes, tensorflow=tf.__version__, keras=keras.__version__,
                  behavior=stats(behavior_values), replay={}, fits={})
    def save(event):
        atomic_json(args.output / 'report.json', result)
        print(json.dumps(event), flush=True)
    save(dict(stage='loaded', samples=len(samples), behavior=result['behavior']))
    original = model.get_weights()
    for checkpoint, path in [('before', before), ('after', after)]:
        if checkpoint == 'after':
            loaded = RustMagePolicy.load(path / 'policy.keras').model
            model.set_weights(loaded.get_weights())
            del loaded
        for replay in (False, True):
            start = time.monotonic()
            summary, values = evaluate(replay=replay, actor=True)
            if checkpoint == 'before':
                summary['behavior_value_max_abs_error'] = float(np.max(np.abs(values-behavior_values)))
                assert summary['behavior_value_max_abs_error'] < 1e-4
                assert summary['max_action_kl'] < 1e-5
            summary['seconds'] = time.monotonic()-start
            key = f'{checkpoint}_{"replayed" if replay else "frozen"}'
            result['replay'][key] = summary
            save(dict(stage=key, **summary))

    if args.fit:
        split = np.random.default_rng(args.split_seed).permutation(len(round_data['games']))
        heldout_games = split[-args.heldout_games:]
        heldout = all_indices[np.isin(game_ids, heldout_games)]
        train = all_indices[~np.isin(game_ids, heldout_games)]
        result['split'] = dict(seed=args.split_seed, heldout_game_indices=heldout_games.tolist(),
                               train_games=len(split)-len(heldout_games), heldout_games=len(heldout_games),
                               train_samples=len(train), heldout_samples=len(heldout))
        features = np.array([np.r_[1.,s.arrays['entities'][0],s.arrays['entities'][1]] for s in samples])
        ridge = np.linalg.solve(features[train].T@features[train]+np.eye(features.shape[1])*10,
                                features[train].T@targets[train])
        result['ridge'] = {name:stats(features[ix]@ridge,ix) for name,ix in [('train',train),('heldout',heldout)]}
        trainer = make_trainer(model, report['config'])
        fresh = RustMagePolicy.load(root / 'round-0000/policy.keras').model.critic.get_weights()
        for case in ('warm_frozen', 'warm_zero', 'fresh_zero'):
            model.set_weights(original)
            with np.load(before / 'critic-optimizer.npz') as archive:
                for i,v in enumerate(trainer.critic_optimizer.variables):
                    v.assign(archive[f'v{i}'])
            if case == 'fresh_zero':
                model.critic.set_weights(fresh)
                for v in trainer.critic_optimizer.variables:
                    if 'learning_rate' not in v.name:
                        v.assign(np.zeros(v.shape, dtype=np.asarray(v).dtype))
            zero = case.endswith('zero')
            actor_hash = hashlib.sha256(b''.join(np.asarray(v).tobytes() for v in model.actor.variables)).hexdigest()
            rng = np.random.default_rng(991)
            cursor, steps = 0, 0
            order = rng.permutation(train)
            rows = []
            for fraction in (0., .25, .5, 1., 2.):
                target_steps = int(np.ceil(len(train)/args.train_batch)*fraction)
                start = time.monotonic()
                while steps < target_steps:
                    if cursor >= len(train):
                        cursor = 0
                        order = rng.permutation(train)
                    ix = order[cursor:cursor+args.train_batch]
                    trainer.critic_step(batch(ix, zero=zero), targets[ix])
                    cursor += args.train_batch
                    steps += 1
                row = dict(epochs=fraction, steps=steps, seconds=time.monotonic()-start,
                           train=evaluate(train, zero=zero)[0], heldout=evaluate(heldout, zero=zero)[0])
                if not zero:
                    row['heldout_replayed'] = evaluate(heldout, replay=True)[0]
                rows.append(row)
                result['fits'][case] = rows
                save(dict(stage=case, **row))
            assert actor_hash == hashlib.sha256(b''.join(np.asarray(v).tobytes() for v in model.actor.variables)).hexdigest()
        result['actor_unchanged_during_fits'] = True
    assert hashes == {str(p): digest(p) for p in sources}
    result['source_files_unchanged'] = True
    save(dict(stage='complete'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--round', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--train-batch', type=int, default=32)
    parser.add_argument('--heldout-games', type=int, default=25)
    parser.add_argument('--split-seed', type=int, default=881)
    parser.add_argument('--fit', action='store_true')
    args = parser.parse_args()
    if min(args.round, args.batch_size, args.train_batch, args.heldout_games) < 1:
        parser.error('Counts must be positive')
    run(args)
