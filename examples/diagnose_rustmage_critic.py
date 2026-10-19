"""Compare grouped versus shuffled critic minibatches on saved RustMage games.

uv run --no-sync python -m examples.diagnose_rustmage_critic \
    --report path/to/report.json --output .local/critic-diagnostic

This never collects games, updates an actor, or replaces a training checkpoint.
Held-out splits keep both seats and every decision from a game together.
The linear probe is supervised on terminal outcomes; it is not reward shaping.
"""
import argparse
import os
os.environ.update(KERAS_BACKEND='tensorflow', CUDA_VISIBLE_DEVICES='-1',
    TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='4',
    TF_ENABLE_ONEDNN_OPTS='0', TF_CPP_MIN_LOG_LEVEL='2')
import json, time, hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from patchwork_automaton.neural.experiment import read_trajectory, make_trainer, atomic_json
from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
from patchwork_automaton.neural.features import collate
from patchwork_automaton.neural.batching import input_signature, example_batch


def run(args):
    root = args.report.parent
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads(args.report.read_text())
    rounds = report['rounds'][-args.rounds:]
    if len(rounds) != args.rounds: raise ValueError('Not enough saved rounds')
    games = [g for r in rounds for g in r['games']]
    if not 0 < args.heldout_games < len(games): raise ValueError('Invalid held-out game count')
    checkpoint = root/'checkpoints'/f"round-{rounds[0]['round']-1:04d}"
    source_files = [args.report, checkpoint/'policy.keras', checkpoint/'critic-optimizer.npz', root/'checkpoints/round-0000/policy.keras']
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    split = np.random.default_rng(args.split_seed).permutation(len(games))
    test_ids = set(split[-args.heldout_games:].tolist())
    records, labels, groups, stages = [], [], [], []
    train, test = [], []
    for gi, game in enumerate(games):
        assert len(game['trajectories']) == 2
        if game['winner'] is None: assert game['terminal_rewards'] == [0., 0.]
        else:
            assert game['terminal_rewards'][game['winner']] == 1
            assert game['terminal_rewards'][1-game['winner']] == -1
        for ti, trajectory in enumerate(game['trajectories']):
            assert trajectory['reward'] == game['terminal_rewards'][ti]
            samples = list(read_trajectory(trajectory['path']))
            if len(samples) != trajectory['samples']: raise ValueError('Incomplete trajectory file')
            if not all(s.advance for s in samples): raise ValueError('Expected one sample per RustMage decision')
            indices = np.arange(len(records), len(records)+len(samples))
            records.extend(samples)
            labels.extend([trajectory['reward']]*len(samples))
            stages.extend(np.linspace(0, 1, len(samples)))
            if gi in test_ids: test.extend(indices)
            else: train.extend(indices); groups.append(indices)
    labels = np.asarray(labels,np.float32)
    train = np.asarray(train); test = np.asarray(test); stages = np.asarray(stages)
    def batch(indices):
        chunk = [records[i] for i in indices]
        arrays = collate([s.arrays for s in chunk])
        arrays['memory_in'] = np.stack([s.memory for s in chunk])
        arrays['advance_memory'] = np.asarray([[float(s.advance)] for s in chunk],np.float32)
        return arrays
    def stats(pred, indices):
        y = labels[indices]; pred = np.asarray(pred)
        variance = float(np.var(y))
        return dict(mse=float(np.mean((pred-y)**2)), explained_variance=float(1-np.var(pred-y)/variance) if variance else None,
            prediction_mean=float(pred.mean()), prediction_std=float(pred.std()),
            sign_accuracy=float(np.mean((pred>0)==(y>0))), label_mean=float(y.mean()),
            late_mse=float(np.mean((pred[stages[indices]>.8]-y[stages[indices]>.8])**2)))
    model = RustMagePolicy.load(checkpoint/'policy.keras').model
    if not getattr(model, 'separate_value', False): raise ValueError('An independent critic is required')
    trainer = make_trainer(model, report['config'])
    with np.load(checkpoint/'critic-optimizer.npz') as archive:
        for i,v in enumerate(trainer.critic_optimizer.variables): v.assign(archive[f'v{i}'])
    original = [np.asarray(v).copy() for v in model.variables]
    optimizer = [np.asarray(v).copy() for v in trainer.critic_optimizer.variables]
    actor_digest = lambda: hashlib.sha256(b''.join(np.asarray(v).tobytes() for v in model.actor.variables)).hexdigest()
    actor_before = actor_digest()
    predict = tf.function(lambda inputs: model.value_output(inputs,training=False)['value'][:,0],
        input_signature=[input_signature(example_batch(model.memory_size,model.feature_spec))],autograph=False)
    def measure():
        result = {}
        for name, indices in [('train',train),('heldout',test)]:
            predictions = np.concatenate([np.asarray(predict(batch(ix))) for ix in np.array_split(indices, max(1,int(np.ceil(len(indices)/64))))])
            result[name] = stats(predictions,indices)
        return result
    result = dict(scope='Saved numeric rollouts, held out by whole game; terminal targets only. Actor frozen. Multiple rounds use successive behavior policies; this is not a new on-policy performance evaluation.',
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_sha256=source_hashes, rounds=[r['round'] for r in rounds],
        fitting=dict(batch_size=args.batch_size, learning_rate=report['config']['learning_rate'],
            shuffle_seed=991, player_probe_ridge=10, tensorflow=tf.__version__, keras=keras.__version__),
        train_games=len(games)-args.heldout_games,heldout_games=args.heldout_games,train_samples=len(train),heldout_samples=len(test),heldout_game_indices=sorted(test_ids),
        all_advancing=all(s.advance for s in records),cases={})
    # Verify a simple player-fact probe can extract outcome information without rewards shaping.
    features = np.array([np.r_[1., s.arrays['entities'][0], s.arrays['entities'][1]] for s in records])
    ridge = np.linalg.solve(features[train].T@features[train]+np.eye(features.shape[1])*10,features[train].T@labels[train])
    result['player_facts_ridge_probe'] = {name:stats(features[ix]@ridge,ix) for name,ix in [('train',train),('heldout',test)]}
    print(json.dumps({'stage':'loaded',**{k:v for k,v in result.items() if k!='cases'}}),flush=True)
    for case in ['grouped_existing','shuffled_existing','grouped_fresh','shuffled_fresh']:
        for v,a in zip(model.variables,original):v.assign(a)
        for v,a in zip(trainer.critic_optimizer.variables,optimizer):v.assign(a)
        if case.endswith('fresh'):
            fresh=RustMagePolicy.load(root/'checkpoints/round-0000/policy.keras').model
            model.critic.set_weights(fresh.critic.get_weights())
            for v in trainer.critic_optimizer.variables:
                if 'learning_rate' not in v.name:v.assign(np.zeros(v.shape,dtype=np.asarray(v).dtype))
        rows = [dict(epoch=0,**measure())]
        rng = np.random.default_rng(991)
        for epoch in range(1,args.epochs+1):
            order = rng.permutation(train) if case.startswith('shuffled') else np.concatenate([groups[i] for i in rng.permutation(len(groups))])
            pure=0; total=0; norms=[]; start=time.monotonic()
            for offset in range(0,len(order),args.batch_size):
                ix=order[offset:offset+args.batch_size]
                norms.append(float(trainer.critic_step(batch(ix),labels[ix])))
                total+=1;pure+=int(len(set(labels[ix]))==1)
            row=dict(epoch=epoch,seconds=time.monotonic()-start,pure_batches=pure,batches=total,
                mean_gradient_norm=float(np.mean(norms)),**measure())
            rows.append(row)
            print(json.dumps({'case':case,**row}),flush=True)
            result['cases'][case]=rows
            atomic_json(output/'report.json',result)
        assert actor_digest()==actor_before
    result['actor_unchanged']=True
    assert source_hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    result['source_files_unchanged']=True
    atomic_json(output/'report.json',result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--heldout-games', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--split-seed', type=int, default=881)
    args = parser.parse_args()
    if min(args.rounds, args.heldout_games, args.epochs, args.batch_size) < 1:
        parser.error('Counts must be positive')
    run(args)
