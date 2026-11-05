"""Run fork and pipe fault tests in the same TF-free environment as workers."""
from pathlib import Path
import subprocess
import sys
import unittest


@unittest.skipUnless(sys.platform.startswith('linux'), 'Requires Linux fork and /proc/self/fd')
class InferenceTimeoutIsolationTests(unittest.TestCase):
    def test_isolated_fork_and_pipe_faults(self):
        cases = Path(__file__).with_name('inference_timeout_cases.py')
        result = subprocess.run([sys.executable, str(cases), '-v'], text=True,
                                capture_output=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
