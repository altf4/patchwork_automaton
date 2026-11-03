"""Numerical Standard inference in game workers, without importing TensorFlow."""
from .compact_dispatch import is_compact,codec,neural,V2

from types import SimpleNamespace
import numpy as np
from ..errors import StaleDecision
from .history import History
from .standard_features import StandardEncoder


class StandardInferencePolicy:
    def __init__(self, model, *, batcher, seed=0, greedy=False, version=0, sink=None):
        self.model, self.batcher, self.greedy, self.version = model, batcher, greedy, version
        if getattr(model,'input_format',None)==V2:
            from .compact_identity import IdentityEncoder
            self.encoder=IdentityEncoder()
        else:self.encoder = StandardEncoder()
        if model.feature_spec != self.encoder.feature_spec:
            raise ValueError('Model feature schema does not match Standard')
        self.history = History(self.encoder.registry)
        self.rng = np.random.default_rng(seed)
        self.states, self.previous, self.last_choices = {}, {}, {}
        self.sink = sink

    def observe(self, observation):
        if is_compact(self.model):return
        self.history.observe(observation)

    def act(self, obs):
        if obs.backend != 'rustmage' or obs.decision is None:
            raise ValueError('RustMage decision required')
        token = obs.decision.token
        key = (token.session, token.game, token.actor)
        if self.previous.get(key) == token: raise StaleDecision('Already answered')
        self.observe(obs)
        choices = self.encoder.expand_options(obs.decision)
        if len(choices) == 1:
            self.previous[key] = token
            return SimpleNamespace(response=choices[0][0])
        encoded = self.encoder.encode(obs, choices=choices)
        arrays = encoded.arrays
        compact=is_compact(self.model)
        if compact:
            arrays=codec(self.model).observation(arrays)
        else:
            arrays.update(self.history.take(obs))
        state = self.states.get(key, np.zeros(self.model.memory_size, np.float32))
        if key in self.last_choices: arrays['selection'] = self.last_choices[key]
        output = self.batcher.predict(dict(arrays, memory_in=state, advance_memory=np.ones(1, np.float32)),
            version=self.version, identity=key, unbatched=True)
        logits = np.asarray(output['logits'])[0].astype(np.float64)
        if not np.isfinite(logits).all(): raise ValueError('Nonfinite model scores')
        logp = logits - np.max(logits)
        logp -= np.log(np.exp(logp).sum())
        index = int(np.argmax(logits) if self.greedy else self.rng.choice(len(logp), p=np.exp(logp)))
        if self.sink is not None:
            from .rollout_records import Sample
            value = float(np.asarray(output['value'])[0, 0])
            if not np.isfinite(value):
                raise ValueError('Nonfinite behavior value')
            self.sink.write(Sample(arrays, state.copy(), True, index, value, logp.astype(np.float32)))
        self.states[key] = np.asarray(output['memory_out'])[0].copy()
        if not compact:self.last_choices[key] = arrays['candidates'][index].copy()
        self.previous[key] = token
        return SimpleNamespace(response=encoded.responses[index])
