"""Defaults and exact data parity for the training rollout reader."""
import os
import unittest
from unittest.mock import patch

import numpy as np

from patchwork_automaton.neural.packed_rollouts import NUMERIC_READ_WORKERS_ENV, NumericRollout
from patchwork_automaton.neural.rollout_records import Sample


class NumericReaderDefaultTests(unittest.TestCase):
    def test_default_override_and_explicit_precedence(self):
        with patch.dict(os.environ):
            os.environ.pop(NUMERIC_READ_WORKERS_ENV, None)
            reader = NumericRollout()
            try:
                self.assertEqual(reader.stats()['workers'], 1)
            finally:
                reader.close()
            os.environ[NUMERIC_READ_WORKERS_ENV] = '4'
            reader = NumericRollout()
            try:
                self.assertEqual(reader.stats()['workers'], 4)
            finally:
                reader.close()
            os.environ[NUMERIC_READ_WORKERS_ENV] = 'invalid'
            reader = NumericRollout(read_workers=2)
            try:
                self.assertEqual(reader.stats()['workers'], 2)
            finally:
                reader.close()

    def test_invalid_environment_override_fails_before_opening_a_file(self):
        for value in ('0', '-1', '', 'invalid'):
            with self.subTest(value=value), patch.dict(os.environ, {NUMERIC_READ_WORKERS_ENV: value}):
                with patch('tempfile.TemporaryFile') as file:
                    with self.assertRaisesRegex(ValueError, NUMERIC_READ_WORKERS_ENV):
                        NumericRollout()
                    file.assert_not_called()

    @unittest.skipUnless(hasattr(os, 'pread'), 'Requires positional file I/O')
    def test_default_reader_matches_explicit_four_workers(self):
        rng = np.random.default_rng(20261031)
        with patch.dict(os.environ):
            os.environ.pop(NUMERIC_READ_WORKERS_ENV, None)
            readers = (NumericRollout(), NumericRollout(read_workers=4))
        try:
            for i in range(20):
                n, count = i % 7 + 1, i % 5 + 1
                sample = Sample(
                    {'entities': rng.normal(size=(n, 3)).astype(np.float32),
                     'relations': rng.normal(size=(n, n, 2)).astype(np.float32),
                     'history': np.zeros((i % 3, 2), np.float32),
                     'candidate_mask': np.ones(count, bool),
                     'selection': np.array([-0., i], np.float32)},
                    np.array([i, -0.], np.float32), True, 0, 0.,
                    np.arange(count, dtype=np.float32))
                for reader in readers:
                    reader.append(sample)
            for reader in readers:
                reader.finish()
            for indices in ([19, 0, 8, 8, 2], [3], list(range(20))):
                a, b = (reader.gather(indices) for reader in readers)
                self.assertEqual(set(a), set(b))
                for key in a:
                    self.assertEqual((a[key].dtype, a[key].shape), (b[key].dtype, b[key].shape))
                    self.assertEqual(a[key].tobytes(), b[key].tobytes(), key)
        finally:
            for reader in readers:
                reader.close()


if __name__ == '__main__':
    unittest.main()
