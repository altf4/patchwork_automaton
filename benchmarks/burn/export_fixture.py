"""Export trusted local PPO examples and Keras weights for the Burn benchmark.

Run from the repository root with .venv/bin/python benchmarks/burn/export_fixture.py.
Tensor interchange is little-endian float32 plus JSON shape/offset metadata.
Loading/export and graph compilation are excluded from the timed kernels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.local/matplotlib'))
os.environ.setdefault('KERAS_HOME', str(ROOT / '.local/keras'))
from examples import train_rustmage  # CPU settings before importing TF
import numpy as np
import tensorflow as tf
from patchwork_automaton.neural.experiment import read_trajectory
from patchwork_automaton.neural.features import collate
from patchwork_automaton.neural.rustmage_policy import RustMagePolicy


def named_weights(model):
    """Stable semantic names, independent of Keras' auto-generated layer names."""
    out = {}
    def layer(prefix, item):
        for variable in item.weights:
            name = variable.path.split('/')[-1]
            if name in ('seed_generator_state',):
                continue
            if prefix + '/' + name in out:
                raise ValueError('Duplicate weight name')
            out[prefix + '/' + name] = variable
    for prefix, net in [('actor', model.actor), ('critic', model.critic)]:
        for name in ('card_embedding', 'semantic_projection', 'semantic_gru',
                     'entity_projection', 'relation_projection', 'history_projection',
                     'history_memory', 'context_projection', 'memory'):
            layer(prefix + '/' + name, getattr(net, name))
        attention = net.attention[0] if prefix == 'actor' else net.attention
        for name in ('query', 'key', 'value', 'output'):
            layer(prefix + '/attention/' + name, getattr(attention, '_' + name + '_dense'))
        for name in ('attention_norm', 'feedforward_norm'):
            item = getattr(net, name)
            layer(prefix + '/' + name, item[0] if prefix == 'actor' else item)
        ff = net.feedforward[0] if prefix == 'actor' else net.feedforward
        for i, item in enumerate(ff.layers):
            layer(prefix + '/feedforward/' + str(i), item)
        if prefix == 'actor':
            for name in ('candidate_projection', 'query_projection', 'score_hidden', 'score'):
                layer(prefix + '/' + name, getattr(net, name))
        else:
            for i, item in enumerate(net.value_head.layers):
                layer(prefix + '/value_head/' + str(i), item)
    assert {id(v) for v in out.values()} == {id(v) for v in model.trainable_variables}
    return out


class Archive:
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=False)
        self.root = directory
        self.file = (directory / 'arrays.bin').open('wb')
        self.arrays = {}

    def add(self, key, value):
        assert key not in self.arrays
        array = np.asarray(value, dtype='<f4', order='C')
        self.arrays[key] = dict(offset=self.file.tell(), shape=list(array.shape),
                                elements=array.size, dtype='float32')
        self.file.write(array.tobytes())

    def close(self, **metadata):
        self.file.close()
        metadata.update(arrays=self.arrays, binary_sha256=sha(self.root / 'arrays.bin'))
        (self.root / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--round', type=int, default=10)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    row = next(r for r in report['rounds'] if r['round'] == args.round)
    state = json.loads(args.checkpoint.with_name('state.json').read_text())
    assert state['round'] == args.round - 1
    model = RustMagePolicy.load(args.checkpoint).model
    assert model.actor_blocks == 1 and model.actor_width == 32 and model.critic_width == 64
    assert model.actor_memory == 32 and model.critic_memory == 64
    archive = Archive(args.output)
    weights = named_weights(model)
    for name, variable in weights.items():
        archive.add('weights/' + name, variable.numpy())
    ids = set(row['split']['training_indices'])
    games = [g for g in row['games'] if g['index'] in ids]
    selected = [games[i] for i in np.linspace(0, len(games) - 1, 16, dtype=int)]
    pool = []
    for game in selected:
        for trajectory in game['trajectories']:
            pool.extend((sample, trajectory['reward']) for sample in read_trajectory(trajectory['path']))
    normalization = row['generalization']['training']['normalization']
    rng = np.random.default_rng(721)
    order = rng.permutation(len(pool))
    typical = [pool[i] for i in order]
    large = [pool[i] for i in order if len(pool[i][0].arrays['entities']) >= 60]
    assert len(large) >= 32
    fixtures = []
    for category, examples in [('typical', typical), ('large', large)]:
        for batch_size in (1, 4, 12, 32):
            name = f'{category}-{batch_size}'
            samples, rewards = zip(*examples[:batch_size])
            inputs = collate([s.arrays for s in samples])
            inputs.update(memory_in=np.stack([s.memory for s in samples]),
                          advance_memory=np.asarray([[float(s.advance)] for s in samples], np.float32))
            # The prototype's actor loss uses a plain mean over genuine choices.
            assert np.all(inputs['candidate_mask'].sum(axis=1) > 1)
            assert np.all(inputs['advance_memory'] == 1)
            for key, value in inputs.items():
                archive.add(name + '/inputs/' + key, value)
            actions = np.asarray([s.action for s in samples], np.int32)
            old = np.asarray([s.log_probs[s.action] for s in samples], np.float32)
            advantages = np.asarray([(r - s.value - normalization['center']) / normalization['scale']
                                     for s, r in zip(samples, rewards)], np.float32)
            for key, value in [('actions', actions), ('returns', rewards), ('old_logp', old),
                               ('advantages', advantages)]:
                archive.add(name + '/targets/' + key, value)
            outputs = model(inputs, training=False)
            for key, value in outputs.items():
                archive.add(name + '/expected/' + key, value.numpy())
            fixtures.append(dict(name=name, batch_size=batch_size, entities=inputs['entities'].shape[1],
                                 candidates=inputs['candidates'].shape[1], history=inputs['history'].shape[1]))
    archive.close(schema=1, fixtures=fixtures, weights=list(weights), checkpoint=str(args.checkpoint),
                  checkpoint_sha256=sha(args.checkpoint), report=str(args.report), report_sha256=sha(args.report),
                  sample_games=[g['index'] for g in selected], normalization=normalization,
                  tensorflow=tf.__version__, model_config=model.get_config(),
                  scope='Frozen real training examples, stored behavior memories, preceding-round checkpoint. '
                        'This is a numerical kernel benchmark, not a full PPO round or game collection benchmark.')
    print(json.dumps(dict(output=str(args.output), fixtures=fixtures, parameters=sum(v.shape.num_elements() for v in weights.values()))))


if __name__ == '__main__':
    main()
