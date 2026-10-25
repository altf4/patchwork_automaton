"""Preflight record checks use numerical fixtures, without neural imports."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from examples.preflight_rustmage_pool import validate_rollouts


class PoolPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)/'closed.pkl.gz'
        self.path.write_bytes(b'closed-record-fixture')
        self.row = dict(status='win', seat=0, neural_choices=1, collection_host='laptop',
            trajectories=[dict(path=str(self.path), samples=1, player=0, reward=1.)])
        self.sample = SimpleNamespace(arrays=dict(candidate_mask=np.ones(2,bool),
            candidates=np.ones((2,3),np.float32)), memory=np.zeros(3,np.float32),
            log_probs=np.log(np.array([.4,.6],np.float32)), action=1, value=.2, advance=True)

    def tearDown(self):
        self.temporary.cleanup()

    def validate(self):
        return validate_rollouts([self.row], 3, read_trajectory=lambda path:iter([self.sample]))

    def test_counts_exact_frozen_records_and_preserves_host_file_bytes(self):
        result = self.validate()
        self.assertEqual(result['samples'], 1)
        self.assertEqual(result['file_bytes_by_host'], dict(raytracer=0, laptop=self.path.stat().st_size))
        self.assertEqual(len(result['files'][0]['sha256']), 64)

    def test_invalid_probabilities_values_counts_or_opponent_data_fail(self):
        self.sample.log_probs[:] = 0.
        with self.assertRaisesRegex(ValueError, 'behavior'):
            self.validate()
        self.sample.log_probs = np.log(np.array([.4,.6],np.float32))
        self.sample.value = float('nan')
        with self.assertRaisesRegex(ValueError, 'behavior'):
            self.validate()
        self.sample.value = .2
        self.row['trajectories'][0]['player'] = 1
        with self.assertRaisesRegex(ValueError, 'opponent'):
            self.validate()
        self.row['trajectories'][0]['player'] = 0
        self.row['trajectories'][0]['samples'] = 2
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
