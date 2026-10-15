"""Sample structured actions from an untrained or loaded Keras model.

No strategy fallback. Every finite choice and bounded integer digit is selected
by the network. The adapter performs only binding, bounds and wire assembly.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import keras
import numpy as np

from ..core import Response
from ..errors import StaleDecision, UnsupportedInteraction
from .features import Encoder, EncodedDecision, collate, candidate_row
from .model import PolicyNetwork
from .history import History


@dataclass(frozen=True)
class Selection:
    stage: str
    index: int
    candidates: int
    log_probability: float
    entropy: float


@dataclass(frozen=True)
class ActionRecord:
    response: Response
    selections: tuple[Selection, ...]
    value: float

    @property
    def log_probability(self):
        return sum(s.log_probability for s in self.selections)


def digit_options(prefix, remaining, maximum):
    """Digits whose suffix has at least one completion in [0, maximum]."""
    return tuple(d for d in range(10) if (prefix * 10 + d) * 10 ** remaining <= maximum)


def distribution_bounds(entries, minimum, maximum, chosen):
    """Keep at least one feasible completion after each recipient allocation."""
    index = len(chosen)
    remaining = entries[index + 1:]
    used = sum(chosen)
    low = max(entries[index]['min'], minimum - used - sum(e['max'] for e in remaining))
    high = min(entries[index]['max'], maximum - used - sum(e['min'] for e in remaining))
    if low > high: raise ValueError('Infeasible distribution constraints')
    return low, high


class NeuralPolicy:
    def __init__(self, model, encoder=None, *, seed=0, temperature=1.):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be positive and finite')
        self.model = model
        self.encoder = encoder or Encoder()
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature
        self.states = {}
        self.previous = {}
        self.last_choices = {}
        self.history = History(self.encoder.registry)

    def reset(self):
        self.states.clear()
        self.previous.clear()
        self.last_choices.clear()
        self.history.reset()

    def observe(self, observation):
        """Fast synchronous listener for every actor-visible XMage update."""
        self.history.observe(observation)

    def _select(self, arrays, state, *, advance, stage, records):
        batch = collate([arrays])
        batch['memory_in'] = state[None, :]
        batch['advance_memory'] = np.asarray([[float(advance)]], np.float32)
        output = self.model(batch, training=False)
        logits = np.asarray(output['logits'])[0].astype(np.float64) / self.temperature
        allowed = arrays['candidate_mask']
        if not np.isfinite(logits[allowed]).all(): raise ValueError('Model emitted nonfinite scores')
        logits[~allowed] = -np.inf
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        index = int(self.rng.choice(len(probabilities), p=probabilities))
        positive = probabilities > 0
        entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        records.append(Selection(stage, index, int(allowed.sum()), float(np.log(probabilities[index])), entropy))
        return index, np.asarray(output['memory_out'])[0], float(np.asarray(output['value'])[0, 0])

    def _integer(self, arrays, state, low, high, records, *, distribution=False, position=0):
        if type(low) is not int or type(high) is not int or low > high:
            raise ValueError('Integer choices require finite integral bounds')
        maximum = high - low
        prefix = 0
        digits = len(str(maximum))
        for offset in range(digits):
            remaining = digits - offset - 1
            choices = digit_options(prefix, remaining, maximum)
            internal = dict(arrays)
            internal['context'] = arrays['context'].copy()
            internal['context'][-3:] = (0., float(not distribution), float(distribution))
            internal['candidates'] = np.asarray([candidate_row('digit', 'digit',
                {'minimum': low, 'maximum': high}, number=d, position=remaining, prefix=prefix) for d in choices], np.float32)
            # Preserve the selected source/recipient when selecting an argument.
            source = arrays['candidate_refs'][:1]
            internal['candidate_refs'] = np.repeat(source, len(choices), axis=0)
            internal['candidate_mask'] = np.ones(len(choices), bool)
            index, _, _ = self._select(internal, state, advance=False,
                stage=f'amount:{position}:digit:{offset}', records=records)
            prefix = prefix * 10 + choices[index]
        return low + prefix

    def act(self, obs):
        encoded = self.encoder.encode(obs)
        key = (encoded.token.session, encoded.token.game, encoded.token.actor)
        if self.previous.get(key) == encoded.token:
            raise StaleDecision('Policy already answered this decision')
        state = self.states.get(key, np.zeros(self.model.memory_size, np.float32))
        self.observe(obs)
        encoded.arrays.update(self.history.take(obs))
        if key in self.last_choices:
            encoded.arrays['selection'] = self.last_choices[key]
        records = []
        index, updated, value = self._select(encoded.arrays, state, advance=True, stage='option', records=records)
        option = encoded.options[index]
        arrays = dict(encoded.arrays)
        arrays['selection'] = arrays['candidates'][index].copy()
        arrays['candidate_refs'] = arrays['candidate_refs'][index:index + 1]
        response_value = None
        if option.kind == 'integer':
            response_value = self._integer(arrays, updated, option.constraints['minimum'], option.constraints['maximum'], records)
        elif option.kind == 'distribution':
            entries = option.constraints['entries']
            minimum, maximum = option.constraints['minimum'], option.constraints['maximum']
            values = []
            for position in range(len(entries)):
                low, high = distribution_bounds(entries, minimum, maximum, values)
                values.append(self._integer(arrays, updated, low, high, records, distribution=True, position=position))
            if not minimum <= sum(values) <= maximum: raise ValueError('Infeasible empty distribution')
            response_value = tuple(values)
        elif option.kind not in ('boolean', 'uuid', 'string'):
            raise UnsupportedInteraction('Neural response kind is not implemented: ' + option.kind)
        self.states[key] = updated
        self.last_choices[key] = arrays['selection']
        self.previous[key] = encoded.token
        return ActionRecord(Response(option.id, response_value), tuple(records), value)

    def save(self, path):
        path = Path(path)
        self.model.save(path)
        path.with_suffix('.manifest.json').write_text(json.dumps(self.encoder.manifest(), indent=2) + '\n')

    @classmethod
    def load(cls, path, *, encoder=None, **options):
        encoder = encoder or Encoder()
        path = Path(path)
        saved = json.loads(path.with_suffix('.manifest.json').read_text())
        if saved['fingerprint'] != encoder.manifest()['fingerprint']:
            raise ValueError('Checkpoint semantic/feature vocabulary does not match encoder')
        return cls(keras.saving.load_model(path), encoder, **options)


def untrained_policy(*, seed=0, temperature=1., **model_options):
    keras.utils.set_random_seed(seed)
    encoder = Encoder()
    return NeuralPolicy(PolicyNetwork(encoder.card_count, **model_options), encoder,
                        seed=seed, temperature=temperature)
