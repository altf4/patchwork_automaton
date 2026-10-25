"""Private, text-free expert decisions; no model predictions or RL returns."""
from dataclasses import dataclass
import gzip
from pathlib import Path
import pickle

from ..backends.rustmage import domain_action
from ..baselines.foundations import FoundationsAggro
from .history import History
from .rustmage_features import RustMageEncoder, canonical, expand_options


@dataclass
class Demonstration:
    arrays: dict
    action: int
    action_kind: str
    decision_kind: str


class DemonstrationWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = gzip.open(self.path, 'wb', compresslevel=1)
        self.count = 0

    def write(self, sample):
        pickle.dump(sample, self.file, protocol=5)
        self.count += 1

    def close(self):
        self.file.close()


def read_demonstration(trajectory):
    """Read trusted local records and check the advertised count and labels."""
    records = []
    with gzip.open(trajectory['path'], 'rb') as stream:
        while True:
            try:
                sample = pickle.load(stream)
            except EOFError:
                break
            if not isinstance(sample, Demonstration):
                raise ValueError('Expected a demonstration, not an RL trajectory')
            mask = sample.arrays['candidate_mask']
            if not 0 <= sample.action < len(mask) or not mask[sample.action] or mask.sum() < 2:
                raise ValueError('Invalid expert action')
            records.append(sample)
    if len(records) != trajectory['samples'] or not records:
        raise ValueError('Empty or incomplete demonstration')
    return records


def expert_index(decision, response, choices):
    option = next(o for o in decision.options if o.id == response.option)
    action = option.constraints['action'] if option.kind == 'finite' else domain_action(option, response.value)
    matches = [i for i, (_, a) in enumerate(choices) if canonical(a) == canonical(action)]
    if len(matches) != 1:
        raise ValueError('Expert response must map to exactly one legal neural candidate')
    return matches[0]


class RecordingExpert:
    """Use the same history/previous-action convention as RustMagePolicy."""
    def __init__(self, writer, expert=None):
        self.writer, self.expert = writer, expert or FoundationsAggro()
        self.encoder = RustMageEncoder()
        self.history = History(self.encoder.registry)
        self.last_choice = None

    def observe(self, observation):
        self.history.observe(observation)

    def choose(self, observation):
        self.observe(observation)
        response = self.expert.choose(observation)
        choices = expand_options(observation.decision, allow_concede=False)
        index = expert_index(observation.decision, response, choices)
        # Match inference: forced actions do not consume history or advance memory.
        if len(choices) > 1:
            arrays = self.encoder.encode(observation, choices=choices).arrays
            arrays.update(self.history.take(observation))
            if self.last_choice is not None:
                arrays['selection'] = self.last_choice.copy()
            self.writer.write(Demonstration(arrays, index, choices[index][1]['kind'], observation.decision.kind))
            self.last_choice = arrays['candidates'][index].copy()
        return response
