"""RustMage mono-red actor-critic policy; all non-forced choices are neural."""
import json
from pathlib import Path

import keras
import numpy as np

from ..errors import StaleDecision
from .features import collate
from .policy import NeuralPolicy, ActionRecord, Selection
from .rustmage_features import RustMageEncoder, expand_options
from .training import Sample


class RustMagePolicy(NeuralPolicy):
    def __init__(self, model, encoder=None, *, seed=0, temperature=1., batcher=None,
                 version=0, sink=None, greedy=False):
        super().__init__(model, encoder or RustMageEncoder(), seed=seed, temperature=temperature)
        if model.feature_spec != self.encoder.feature_spec:
            raise ValueError('Model feature schema does not match RustMage')
        self.batcher, self.version, self.sink, self.greedy = batcher, version, sink, greedy

    def act(self, obs):
        if obs.backend != 'rustmage' or obs.decision is None:
            raise ValueError('RustMage decision required')
        token = obs.decision.token
        key = (token.session, token.game, token.actor)
        if self.previous.get(key) == token: raise StaleDecision('Already answered')
        self.observe(obs)
        choices = expand_options(obs.decision, allow_concede=self.encoder.allow_concede)
        # Forced rules choices need no inference. Their visible changes are kept
        # in History and consumed at the next genuine choice, including changes
        # during opponent actions. There is no strategy fallback here.
        if len(choices) == 1:
            self.previous[key] = token
            return ActionRecord(choices[0][0], (), 0.)
        encoded = self.encoder.encode(obs, choices=choices)
        arrays = encoded.arrays
        arrays.update(self.history.take(obs))
        state = self.states.get(key, np.zeros(self.model.memory_size, np.float32))
        if key in self.last_choices: arrays['selection'] = self.last_choices[key]
        if self.batcher:
            # Queue unpadded arrays; the service packs all requests exactly once.
            batch = dict(arrays, memory_in=state, advance_memory=np.ones(1, np.float32))
            output = self.batcher.predict(batch, version=self.version, identity=key, unbatched=True)
        else:
            batch = collate([arrays])
            batch.update(memory_in=state[None, :], advance_memory=np.ones((1, 1), np.float32))
            output = self.model(batch, training=False)
        logits = np.asarray(output['logits'])[0].astype(np.float64) / self.temperature
        if not np.isfinite(logits).all(): raise ValueError('Nonfinite model scores')
        logp = logits - np.max(logits)
        logp -= np.log(np.exp(logp).sum())
        probabilities = np.exp(logp)
        index = int(np.argmax(logits) if self.greedy else self.rng.choice(len(logp), p=probabilities))
        value = float(np.asarray(output['value'])[0, 0])
        if self.sink: self.sink.write(Sample(arrays, state, True, index, value, logp.astype(np.float32)))
        self.states[key] = np.asarray(output['memory_out'])[0].copy()
        self.last_choices[key] = arrays['candidates'][index].copy()
        self.previous[key] = token
        record = Selection('option', index, len(logp), float(logp[index]), float(-np.sum(probabilities*logp)))
        return ActionRecord(encoded.responses[index], (record,), value)

    @classmethod
    def load(cls, path, *, encoder=None, **options):
        path = Path(path)
        saved = json.loads(path.with_suffix('.manifest.json').read_text())
        if saved.get('backend') != 'rustmage': raise ValueError('Not a RustMage checkpoint')
        encoder = encoder or RustMageEncoder(allow_concede=saved['allow_concede'])
        current = encoder.manifest()
        # JSON's sorted-key fingerprint describes the action field mapping, but
        # its insertion order also defines tensor columns. Check that separately
        # for version-1 manifests, whose JSON writer preserves insertion order.
        if (saved['fingerprint'] != current['fingerprint']
                or tuple(saved['actions']) != tuple(current['actions'])):
            raise ValueError('RustMage checkpoint vocabulary mismatch')
        return cls(keras.saving.load_model(path), encoder, **options)
