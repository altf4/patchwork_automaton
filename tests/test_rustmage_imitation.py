"""Expert labels, recurrent replay, and strict separation from RL/value fitting."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from test_neural import AVAILABLE

if AVAILABLE:
    import keras
    import numpy as np
    from patchwork_automaton.backends.rustmage import action_options
    from patchwork_automaton.core import Decision, Response
    from patchwork_automaton.neural.batching import example_batch
    from patchwork_automaton.neural.demonstrations import Demonstration, DemonstrationWriter, expert_index, read_demonstration
    from patchwork_automaton.neural.experiment import load_checkpoint, save_checkpoint
    from patchwork_automaton.neural.imitation import ImitationTrainer
    from patchwork_automaton.neural.model import SeparateActorCritic
    from patchwork_automaton.neural.rustmage_features import RustMageEncoder, expand_options
    from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
    from patchwork_automaton.neural.validation import split_games
    from patchwork_automaton.rl.rustmage_experiment import play_game

NATIVE = (Path(os.environ.get('RUSTMAGE_ROOT', '.local/rustmage')) / 'target/release/librustmage.so').is_file()


@unittest.skipUnless(AVAILABLE, 'Requires neural dependencies')
class LabelTests(unittest.TestCase):
    def test_domain_response_matches_canonical_deduplicated_action(self):
        raw = dict(actions=[dict(kind='PayCosts', payment=[0, 0, 0, 2, 0, 0])],
            mana_payment=dict(required=[0, 0, 0, 1, 0, 0], available=[1, 0, 0, 2, 0, 0], generic=1))
        decision = Decision(None, 'pay_costs', '', action_options(raw))
        choices = expand_options(decision)
        label = expert_index(decision, Response('domain:mana_payment', (0, 0, 0, 2, 0, 0)), choices)
        self.assertEqual(tuple(choices[label][1]['payment']), (0, 0, 0, 2, 0, 0))
        with self.assertRaises(ValueError):
            expert_index(decision, Response('domain:mana_payment', (99, 0, 0, 0, 0, 0)), choices)


@unittest.skipUnless(AVAILABLE and NATIVE, 'Requires neural dependencies and RustMage')
class ImitationTests(unittest.TestCase):
    def collect(self, directory, seed=721, index=0):
        result = play_game(seed=seed, index=index, mode='demonstrations', output=directory)
        self.assertIn(result['status'], ('win', 'loss', 'draw'), result.get('error'))
        self.assertNotIn('terminal_rewards', result)
        self.assertEqual(len(result['trajectories']), 2)
        self.assertTrue(all('reward' not in t for t in result['trajectories']))
        return result

    def model(self):
        encoder = RustMageEncoder()
        model = SeparateActorCritic(encoder.card_count, encoder.feature_spec,
            actor_width=8, actor_memory=4, actor_blocks=0, critic_width=8, critic_memory=4)
        model(example_batch(model.memory_size, model.feature_spec))
        return model

    def test_labels_history_and_previous_action_match_neural_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.collect(directory)
            records = {f'P{p}': read_demonstration(t) for p, t in enumerate(row['trajectories'])}
            test = self
            class ExpertInference:
                def __init__(self): self.offsets = dict(P0=0, P1=0)
                def predict(self, batch, *, identity, **kwargs):
                    player = identity[-1].value
                    expected = records[player][self.offsets[player]]
                    self.offsets[player] += 1
                    for k, v in expected.arrays.items():
                        np.testing.assert_array_equal(batch[k], v, err_msg=k)
                    logits = np.full((1, len(expected.arrays['candidate_mask'])), -1000., np.float32)
                    logits[0, expected.action] = 0.
                    return dict(logits=logits, value=np.zeros((1, 1), np.float32),
                        memory_out=batch['memory_in'][None, :] + 1.)
            inference = ExpertInference()
            encoder = RustMageEncoder()
            replay = play_game(seed=721, index=0, mode='selfplay',
                model=SimpleNamespace(memory_size=8, feature_spec=encoder.feature_spec), batcher=inference)
            self.assertEqual(replay['status'], row['status'], replay.get('error'))
            self.assertEqual(replay['actions'], row['actions'])
            self.assertEqual(inference.offsets, {p: len(s) for p, s in records.items()})
            for p, samples in records.items():
                self.assertFalse(np.any(samples[0].arrays['selection']))
                for before, after in zip(samples, samples[1:]):
                    np.testing.assert_array_equal(after.arrays['selection'], before.arrays['candidates'][before.action])

    def test_training_reduces_expert_loss_and_leaves_value_and_validation_untouched(self):
        keras.utils.set_random_seed(61)
        with tempfile.TemporaryDirectory() as directory:
            game = self.collect(directory)
            game['partition'] = 'training'
            model = self.model()
            value = [v.copy() for v in model.critic.get_weights()]
            trainer = ImitationTrainer(model, batch_size=2, epochs=2, unroll=4, learning_rate=.003)
            before = trainer.evaluate([game])
            fit = trainer.fit([game])
            after = trainer.evaluate([game])
            self.assertLess(after['cross_entropy'], before['cross_entropy'])
            self.assertGreater(fit['steps_this_update'], 0)
            for a, b in zip(value, model.critic.get_weights()): np.testing.assert_array_equal(a, b)
            weights = [v.copy() for v in model.get_weights()]
            rng = deepcopy(trainer.rng.bit_generator.state)
            steps = int(trainer.optimizer.iterations)
            trainer.evaluate([dict(game, partition='validation')])
            self.assertEqual(trainer.rng.bit_generator.state, rng)
            self.assertEqual(int(trainer.optimizer.iterations), steps)
            for a, b in zip(weights, model.get_weights()): np.testing.assert_array_equal(a, b)
            with self.assertRaisesRegex(ValueError, 'training partition'):
                trainer.fit([dict(game, partition='validation')])
            self.assertIn('Pass', after['per_action'])
            self.assertIsNotNone(after['non_pass_accuracy'])

    def test_imitation_checkpoint_roundtrip_restores_optimizer_and_sampler(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self.model()
            config = dict(algorithm='imitation', learning_rate=.0003, train_batch=2, epochs=1, unroll=2, seed=1)
            trainer = ImitationTrainer(model, batch_size=2, epochs=1, unroll=2, seed=5001)
            trainer.rng.permutation(17)
            save_checkpoint(Path(directory) / 'checkpoints', model, trainer,
                dict(round=0, config=config), policy_type=RustMagePolicy)
            loaded, restored, state = load_checkpoint(Path(directory) / 'checkpoints')
            self.assertIsInstance(restored, ImitationTrainer)
            self.assertEqual(state['optimizer_steps'], 0)
            self.assertNotIn('critic_optimizer_steps', state)
            for a, b in zip(model.get_weights(), loaded.get_weights()): np.testing.assert_array_equal(a, b)
            self.assertEqual(trainer.rng.bit_generator.state, restored.rng.bit_generator.state)
            for a, b in zip(trainer.optimizer.variables, restored.optimizer.variables): np.testing.assert_array_equal(a, b)

    def test_both_seats_are_partitioned_together_and_invalid_records_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            games = [self.collect(directory, seed=721 + i, index=i) for i in range(5)]
            train, validation, split = split_games(games, seed=7, round_id=1)
            self.assertEqual((len(train), len(validation)), (4, 1))
            self.assertFalse({t['path'] for g in train for t in g['trajectories']} &
                             {t['path'] for g in validation for t in g['trajectories']})
            descriptor = games[0]['trajectories'][0]
            sample = read_demonstration(descriptor)[0]
            sample.action = 99999
            bad = DemonstrationWriter(Path(directory) / 'bad.pkl.gz')
            bad.write(sample)
            bad.close()
            with self.assertRaises(ValueError):
                read_demonstration(dict(path=str(bad.path), samples=1))
