"""Owned dummy-process supervision; never changes Windows power state."""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from patchwork_automaton.rl.hartford_guard import (
    POWERSHELL_KEEP_AWAKE, WindowsAwakeKeeper, load_command, parser, supervise)


@unittest.skipUnless(os.name == 'posix', 'Supervisor requires POSIX process groups')
class HartfordGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.command_path = self.root/'command.json'
        self.command_path.write_text(json.dumps([sys.executable, '-u', '-c', 'print("child success")']))
        self.args = parser().parse_args(['--root', str(self.root/'guard'), '--command', str(self.command_path),
            '--check-seconds', '.02', '--memory-grace-seconds', '.08', '--terminate-grace-seconds', '.12',
            '--keeper-startup-seconds', '.2'])

    def child(self, code):
        self.command_path.write_text(json.dumps([sys.executable, '-u', '-c', code]))

    def fake_keeper(self, code):
        return WindowsAwakeKeeper(command=[sys.executable, '-u', '-c', code])

    def test_normal_child_logs_exit_status_and_never_retries(self):
        result = supervise(self.args)
        self.assertEqual((result['status'], result['exit_code'], result['guard_exit_code']), ('exited', 0, 0))
        self.assertEqual(result['attempts'], 1)
        self.assertEqual(result['child_pid'], result['process_group'])
        self.assertIn('child success', (self.args.root/'runner.log').read_text())
        self.assertTrue((self.args.root/'host-memory-samples.jsonl').exists())
        self.assertEqual(json.loads((self.args.root/'guard-status.json').read_text()), result)
        with self.assertRaisesRegex(FileExistsError, 'automatic retries'):
            supervise(self.args)

    def test_nonzero_child_exit_is_preserved(self):
        self.child('raise SystemExit(7)')
        result = supervise(self.args)
        self.assertEqual((result['exit_code'], result['guard_exit_code']), (7, 7))
        self.assertEqual(result['attempts'], 1)

    def test_low_memory_terminates_only_owned_group_then_kills_term_resistant_child(self):
        marker = self.root/'started'
        self.child('import pathlib,signal,time; '
            'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
            f'pathlib.Path({str(marker)!r}).write_text("once"); '
            'time.sleep(30)')
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        try:
            with patch('patchwork_automaton.rl.hartford_guard.available_memory_bytes',
                       side_effect=lambda: 0 if marker.exists() else 10*1024**3):
                result = supervise(self.args)
            self.assertEqual(result['reason'], 'low_memory')
            self.assertEqual(result['guard_exit_code'], 75)
            self.assertEqual(result['child_exit_code'], -signal.SIGKILL)
            self.assertTrue(result['cleanup']['term_sent'])
            self.assertTrue(result['cleanup']['kill_sent'])
            self.assertEqual(marker.read_text(), 'once')
            self.assertIsNone(unrelated.poll())
            self.assertGreaterEqual(result['last_sample']['low_memory_seconds'], .08)
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_brief_low_memory_spike_resets_grace(self):
        self.child('import time; time.sleep(.2)')
        samples = iter([10*1024**3, 0, 0, 10*1024**3])
        with patch('patchwork_automaton.rl.hartford_guard.available_memory_bytes',
                   side_effect=lambda: next(samples, 10*1024**3)):
            result = supervise(self.args)
        self.assertEqual(result['guard_exit_code'], 0)
        self.assertEqual(result['reason'], 'child_exit')

    def test_ready_keeper_is_closed_by_eof_and_clears_before_guard_completes(self):
        marker = self.root/'released'
        keeper = self.fake_keeper('import pathlib,sys; print("READY",flush=True); '
            f'sys.stdin.read(); pathlib.Path({str(marker)!r}).write_text("released")')
        self.args.keep_windows_awake = True
        with patch('patchwork_automaton.rl.hartford_guard.WindowsAwakeKeeper', return_value=keeper):
            result = supervise(self.args)
        self.assertEqual(result['guard_exit_code'], 0)
        self.assertTrue(result['windows_awake']['ready'])
        self.assertTrue(result['windows_awake']['release']['graceful_release'])
        self.assertEqual(marker.read_text(), 'released')

    def test_keeper_startup_failure_prevents_child_launch(self):
        marker = self.root/'child-started'
        self.child(f'import pathlib; pathlib.Path({str(marker)!r}).write_text("bad")')
        keeper = self.fake_keeper('raise SystemExit(9)')
        self.args.keep_windows_awake = True
        with patch('patchwork_automaton.rl.hartford_guard.WindowsAwakeKeeper', return_value=keeper):
            result = supervise(self.args)
        self.assertEqual(result['guard_exit_code'], 70)
        self.assertEqual(result['attempts'], 0)
        self.assertIsNone(result['child_pid'])
        self.assertFalse(marker.exists())
        self.assertFalse(result['windows_awake']['ready'])

    def test_keeper_handshake_timeout_is_bounded_and_prevents_launch(self):
        keeper = self.fake_keeper('import sys; sys.stdin.read()')
        self.args.keep_windows_awake = True
        started = time.monotonic()
        with patch('patchwork_automaton.rl.hartford_guard.WindowsAwakeKeeper', return_value=keeper):
            result = supervise(self.args)
        self.assertLess(time.monotonic()-started, 2.)
        self.assertEqual(result['attempts'], 0)
        self.assertIn('startup deadline', result['error'])

    def test_unexpected_keeper_exit_stops_child_without_retry(self):
        keeper = self.fake_keeper('import time; print("READY",flush=True); time.sleep(.12)')
        self.child('import time; time.sleep(30)')
        self.args.keep_windows_awake = True
        with patch('patchwork_automaton.rl.hartford_guard.WindowsAwakeKeeper', return_value=keeper):
            result = supervise(self.args)
        self.assertEqual(result['guard_exit_code'], 70)
        self.assertEqual(result['attempts'], 1)
        self.assertIn('exited unexpectedly', result['error'])
        self.assertLess(result['child_exit_code'], 0)

    def test_supervisor_sigterm_stops_child_and_preserves_signal_status(self):
        self.child('import os,signal,time; time.sleep(.06); '
                   'os.kill(os.getppid(),signal.SIGTERM); time.sleep(30)')
        old = signal.getsignal(signal.SIGTERM)
        result = supervise(self.args)
        self.assertEqual(result['reason'], 'signal')
        self.assertEqual(result['signal'], signal.SIGTERM)
        self.assertEqual(result['guard_exit_code'], 143)
        self.assertLess(result['child_exit_code'], 0)
        self.assertEqual(signal.getsignal(signal.SIGTERM), old)

    def test_monitor_exception_stops_child_and_releases_keeper(self):
        keeper = self.fake_keeper('import sys; print("READY",flush=True); sys.stdin.read()')
        self.child('import time; time.sleep(30)')
        self.args.keep_windows_awake = True
        with patch('patchwork_automaton.rl.hartford_guard.WindowsAwakeKeeper', return_value=keeper), \
             patch('patchwork_automaton.rl.hartford_guard.group_memory', side_effect=RuntimeError('diagnostic failed')):
            result = supervise(self.args)
        self.assertEqual(result['guard_exit_code'], 70)
        self.assertIn('diagnostic failed', result['error'])
        self.assertTrue(result['windows_awake']['release']['graceful_release'])
        self.assertLess(result['child_exit_code'], 0)

    def test_keyboard_interrupt_stops_owned_child(self):
        self.child('import time; time.sleep(30)')
        with patch('patchwork_automaton.rl.hartford_guard.group_memory', side_effect=KeyboardInterrupt):
            result = supervise(self.args)
        self.assertEqual(result['guard_exit_code'], 130)
        self.assertEqual(result['reason'], 'signal')
        self.assertLess(result['child_exit_code'], 0)

    def test_group_cleanup_removes_descendant_after_leader_exits(self):
        marker = self.root/'descendant-ready'
        descendant_code = ('import os,pathlib,signal,time; '
            'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
            f'pathlib.Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)')
        self.child('import pathlib,subprocess,sys,time; '
                   f'subprocess.Popen([sys.executable,"-c",{descendant_code!r}]); '
                   f'marker=pathlib.Path({str(marker)!r}); '
                   '\nwhile not marker.exists(): time.sleep(.01)')
        result = supervise(self.args)
        self.assertEqual(result['guard_exit_code'], 0)
        self.assertTrue(result['cleanup']['kill_sent'])
        descendant = int(marker.read_text())
        deadline = time.monotonic()+1
        while time.monotonic() < deadline:
            try:
                state = Path(f'/proc/{descendant}/stat').read_text().rsplit(')', 1)[1].split()[0]
            except FileNotFoundError:
                break
            if state == 'Z':
                break
            time.sleep(.01)
        else:
            self.fail('Owned descendant survived process-group cleanup')

    def test_command_and_numeric_limits_rejected_before_launch(self):
        self.command_path.write_text(json.dumps('shell command'))
        with self.assertRaises(ValueError):
            load_command(self.command_path)
        self.command_path.write_text(json.dumps([sys.executable, '-c', 'pass']))
        invalid = deepcopy(self.args)
        invalid.memory_grace_seconds = float('nan')
        with self.assertRaises(ValueError):
            supervise(invalid)
        self.assertFalse(self.args.root.exists())

    def test_windows_request_is_thread_scoped_and_cleared_in_finally(self):
        self.assertIn('2147483649', POWERSHELL_KEEP_AWAKE)
        self.assertIn('2147483648', POWERSHELL_KEEP_AWAKE)
        self.assertIn('finally', POWERSHELL_KEEP_AWAKE)
        self.assertIn('[Console]::In.ReadToEnd()', POWERSHELL_KEEP_AWAKE)
        self.assertNotIn('powercfg', POWERSHELL_KEEP_AWAKE.lower())
        self.assertNotIn('scheduledtask', POWERSHELL_KEEP_AWAKE.lower())


if __name__ == '__main__':
    unittest.main()
