"""Numerical reference and warmed TensorFlow timings for the Burn experiment."""
import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import time

from export_fixture import Archive, named_weights, sha
import numpy as np
import tensorflow as tf
from patchwork_automaton.neural.batching import input_signature
from patchwork_automaton.neural.ppo import PPOTrainer
from patchwork_automaton.neural.rustmage_policy import RustMagePolicy


class Fixture:
    def __init__(self, root):
        self.manifest = json.loads((root / 'manifest.json').read_text())
        self.data = (root / 'arrays.bin').read_bytes()

    def array(self, key):
        spec = self.manifest['arrays'][key]
        return np.frombuffer(self.data, '<f4', count=spec['elements'], offset=spec['offset']).reshape(spec['shape']).copy()

    def batch(self, name):
        prefix = name + '/inputs/'
        inputs = {key[len(prefix):]: self.array(key) for key in self.manifest['arrays'] if key.startswith(prefix)}
        for key, value in inputs.items():
            if key.endswith('_mask'):
                inputs[key] = value.astype(bool)
            elif key in ('card_ids', 'history_cards'):
                inputs[key] = value.astype(np.int32)
        targets = [self.array(name + '/targets/' + key) for key in ('actions', 'returns', 'old_logp', 'advantages')]
        targets[0] = targets[0].astype(np.int32)
        return {k: tf.convert_to_tensor(v) for k, v in inputs.items()}, [tf.convert_to_tensor(v) for v in targets]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('validate', 'bench'))
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--train-steps', type=int, default=12)
    args = parser.parse_args()
    fixture = Fixture(args.fixture)
    model = RustMagePolicy.load(fixture.manifest['checkpoint']).model
    trainer = PPOTrainer(model, learning_rate=.0003, batch_size=32, entropy_weight=.01, clip_norm=1.)
    weights = named_weights(model)
    batches = {f['name']: fixture.batch(f['name']) for f in fixture.manifest['fixtures']}
    signature = input_signature(batches['typical-32'][0])

    @tf.function(input_signature=[signature], autograph=False)
    def forward(inputs):
        return model(inputs, training=False)

    @tf.function(input_signature=[signature, tf.TensorSpec([None], tf.int32)] + [tf.TensorSpec([None], tf.float32)] * 3, autograph=False)
    def actor_grads(inputs, actions, rewards, old, adv):
        with tf.GradientTape() as tape:
            output = model.actor_output(inputs, training=True)
            terms = trainer._terms_from_output(output, tf.nn.log_softmax(output['logits']), inputs, actions, rewards, old, adv)
            loss = terms[1] - .01 * terms[3]
        grads = [tf.convert_to_tensor(g) for g in tape.gradient(loss, list(model.actor.trainable_variables))]
        return loss, tf.linalg.global_norm(grads), grads

    @tf.function(input_signature=[signature, tf.TensorSpec([None], tf.float32)], autograph=False)
    def critic_grads(inputs, rewards):
        with tf.GradientTape() as tape:
            value = model.value_output(inputs, training=True)['value'][:, 0]
            mask = inputs['advance_memory'][:, 0]
            loss = tf.math.divide_no_nan(tf.reduce_sum(tf.square(value - rewards) * mask), tf.reduce_sum(mask))
        grads = [tf.convert_to_tensor(g) for g in tape.gradient(loss, list(model.critic.trainable_variables))]
        return loss, tf.linalg.global_norm(grads), grads

    def actor_step(name):
        inputs, (actions, rewards, old, adv) = batches[name]
        return float(trainer.step(inputs, actions, rewards, old, adv))

    def critic_step(name):
        inputs, (_, rewards, _, _) = batches[name]
        return float(trainer.critic_step(inputs, rewards))

    variables = list(model.trainable_variables) + list(trainer.optimizer.variables) + list(trainer.critic_optimizer.variables)
    saved = [v.numpy().copy() for v in variables]
    def reset():
        for v, a in zip(variables, saved):
            v.assign(a)

    if args.mode == 'validate':
        output = Archive(args.output)
        for name, (inputs, _) in batches.items():
            for key, value in forward(inputs).items():
                output.add(name + '/' + key, value.numpy())
        losses = {}
        # Two different shapes, all parameter gradients, and three Adam updates.
        for name in ('typical-32', 'large-32'):
            reset()
            inputs, targets = batches[name]
            for kind, variables_, result in [('actor', model.actor.trainable_variables, actor_grads(inputs, *targets)),
                                              ('critic', model.critic.trainable_variables, critic_grads(inputs, targets[1]))]:
                loss, norm, grads = result
                losses[name + '/' + kind] = dict(loss=float(loss), norm=float(norm))
                reverse = {id(v): k for k, v in weights.items()}
                for v, grad in zip(variables_, grads):
                    output.add(name + '/grad/' + reverse[id(v)], grad.numpy().reshape(-1))
            for step in range(3):
                actor_step(name); critic_step(name)
                for k, v in weights.items():
                    output.add(f'{name}/after-{step+1}/' + k, v.numpy().reshape(-1))
            # Export Adam moments to check the optimizer implementation too.
            for kind, optimizer in [('actor', trainer.optimizer), ('critic', trainer.critic_optimizer)]:
                vars_ = model.actor.trainable_variables if kind == 'actor' else model.critic.trainable_variables
                reverse = {id(v): k for k, v in weights.items()}
                for v, m, s in zip(vars_, optimizer._momentums, optimizer._velocities):
                    output.add(name + '/adam-m/' + reverse[id(v)], m.numpy().reshape(-1))
                    output.add(name + '/adam-v/' + reverse[id(v)], s.numpy().reshape(-1))
        output.close(losses=losses, tensorflow=tf.__version__, fixture_sha256=sha(args.fixture / 'manifest.json'))
        print(json.dumps(dict(mode='validate', output=str(args.output), losses=losses)), flush=True)
        return

    rows = []
    jobs = [(f['name'], 'inference') for f in fixture.manifest['fixtures']]
    jobs += [(name, kind) for name in ('typical-32', 'large-32') for kind in ('actor_step', 'critic_step')]
    for name, kind in jobs:
        if kind == 'inference':
            def fn():
                return {k: v.numpy() for k, v in forward(batches[name][0]).items()}
        elif kind == 'actor_step':
            fn = lambda: actor_step(name)
        else:
            fn = lambda: critic_step(name)
        for _ in range(5): fn()
        reset()
        if kind == 'inference':
            start = time.perf_counter()
            for _ in range(5): fn()
            iterations = max(5, min(256, round(.3 / ((time.perf_counter() - start) / 5))))
        else:
            iterations = args.train_steps
        wall, cpu = [], []
        for _ in range(args.trials):
            reset()
            start, cstart = time.perf_counter(), time.process_time()
            for _ in range(iterations): fn()
            wall.append((time.perf_counter() - start) / iterations)
            cpu.append((time.process_time() - cstart) / iterations)
        row = dict(fixture=name, operation=kind, iterations=iterations, wall_seconds=wall, cpu_seconds=cpu,
                   median_ms=statistics.median(wall) * 1000)
        rows.append(row)
        print(json.dumps(row), flush=True)
    output = dict(framework='tensorflow', version=tf.__version__, rows=rows,
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  runtime={k: os.environ.get(k) for k in ('TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'TF_ENABLE_ONEDNN_OPTS')},
                  fixture_sha256=sha(args.fixture / 'manifest.json'),
                  timing='Warmed, preloaded tensors. Inference includes output materialization; training includes forward, backward, finite checks, global norm clipping, fresh Adam updates and norm materialization. Preparation and reset excluded.')
    args.output.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
