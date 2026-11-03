"""Completed outcomes survive errors in inference transport cleanup."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from patchwork_automaton.rl.standard_ppo import play_ppo_game


class ResultHandoffTests(unittest.TestCase):
    def test_completed_result_survives_reset_failure_and_blocks_replay(self):
        game = dict(seed=41, index=7, learner_seat=0, opponent_kind='scripted')
        row = dict(game, behavior_version=150, status='draw', draw=True, trajectories=[])
        with tempfile.TemporaryDirectory() as directory:
            with patch('patchwork_automaton.rl.standard_imitation.timed_standard_game', return_value=row) as play, \
                    patch('patchwork_automaton.rl.standard_ppo._reset_inference', side_effect=TimeoutError('Inference reset timed out')):
                with self.assertRaises(TimeoutError):
                    play_ppo_game(seed=41, index=7, game=game, model=None, batcher=None, version=150,
                                  completed_result_directory=directory)
                saved = json.loads((Path(directory) / 'game-7.json').read_text())
                self.assertEqual(saved['row'], row)
                payload = json.dumps(row, sort_keys=True, separators=(',', ':')).encode()
                self.assertEqual(saved['sha256'], hashlib.sha256(payload).hexdigest())
                with self.assertRaises(FileExistsError):
                    play_ppo_game(seed=41, index=7, game=game, model=None, batcher=None, version=150,
                                  completed_result_directory=directory)
                self.assertEqual(play.call_count, 1)


if __name__ == '__main__':
    unittest.main()
