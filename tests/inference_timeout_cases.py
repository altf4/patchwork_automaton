"""Fork/pipe interruption cases run in a clean worker process without TF."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import signal
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from patchwork_automaton.neural.inference_processes import bridge_requests
from patchwork_automaton.rl import standard_imitation
from patchwork_automaton.rl.inference_transport import (
    InferenceCleanupError, InferenceIsolation, recv_message, send_message,
)
from patchwork_automaton.rl.rustmage_processes import _RemoteInference
from patchwork_automaton.rl.standard_ppo import play_ppo_game


_PERSISTENT_FDS = ()


class _Echo:
    def __init__(self, role):
        self.role, self.calls = role, 0

    def predict(self, batch, *, version, identity, unbatched):
        self.calls += 1
        if batch.get('delay'):
            time.sleep(batch['delay'])
        return dict(role=self.role, calls=self.calls, batch=batch,
                    version=version, identity=identity, unbatched=unbatched)


def _server(server, inherited_clients, role):
    for connection in inherited_clients:
        connection.close()
    bridge_requests(server, _Echo(role), threading.Event())


def _echo_game(game, **options):
    closed = []
    for fd in _PERSISTENT_FDS:
        try:
            os.fstat(fd)
        except OSError:
            closed.append(fd)
    result = dict(game, status='win', winner_seat=0, trajectories=[],
                  child_threads=threading.active_count(), closed_persistent_fds=closed)
    try:
        for role in ('batcher', 'opponent_batcher'):
            if options.get(role) is not None:
                result[role] = options[role].predict(
                    dict(game=game['index'], delay=options.get('inference_delay', 0)),
                    version=options.get('version', 3), identity=(game['seed'], role), unbatched=True)
    except Exception as error:
        result.update(status='error', error=f'{type(error).__name__}: {error}')
    return result


def _interrupted_game(game, **options):
    batcher = options[options.get('break_role', 'batcher')]
    mode = options['break_mode']
    if mode == 'request_body':
        # Leave a request frame incomplete while its sender is still alive.
        os.write(batcher.connection.fileno(), struct.pack('!i', 4096) + b'partial')
    else:
        request = ('inference-v1', batcher._request_id(), 3,
                   (game['seed'], options.get('break_role', 'batcher')), True, {'game': game['index']})
        send_message(batcher.connection, request)
        fd = batcher.connection.fileno()
        # Read only part of the response, exactly where the old persistent-pipe
        # design could be killed and poison the next game's receive.
        header = os.read(fd, 4)
        if len(header) != 4:
            raise AssertionError('Expected a complete response header')
        if mode == 'response_body':
            if not os.read(fd, 7):
                raise AssertionError('Expected response body bytes')
    Path(options['progress_path']).with_name('interruption-reached').write_text(mode)
    time.sleep(30)


def _partial_result(connection, game, options, trace_path, close_connections=()):
    for inherited in close_connections:
        inherited.close()
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    os.write(connection.fileno(), struct.pack('!i', 1000000) + b'partial')
    time.sleep(30)


def _bad_result(connection, game, options, trace_path, close_connections=()):
    for inherited in close_connections:
        inherited.close()
    os.write(connection.fileno(),struct.pack('!i',-2))
    connection.close()


class InferenceTimeoutIsolationTests(unittest.TestCase):
    def setUp(self):
        import sys
        self.assertNotIn('tensorflow', sys.modules)
        self.context = mp.get_context('fork')
        self.clients, self.servers = [], []
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.runtime = patch.dict(os.environ, {'PATCHWORK_STANDARD_PLAYBACK': 'python'})
        self.runtime.start(); self.addCleanup(self.runtime.stop)
        self.worker = patch.object(standard_imitation, '_game_worker_broken', False)
        self.worker.start(); self.addCleanup(self.worker.stop)
        self.addCleanup(self.close_servers)

    def close_servers(self):
        for client in self.clients:
            client.connection.close()
        for process in self.servers:
            if process.is_alive():
                process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill(); process.join(2)
            process.close()

    def client(self, role='learner', timeout=1.):
        server, child = self.context.Pipe()
        process = self.context.Process(target=_server,
            args=(server, [c.connection for c in self.clients]+[child], role))
        process.start(); server.close()
        remote = _RemoteInference(child, timeout=timeout)
        self.clients.append(remote); self.servers.append(process)
        return remote

    def game(self, index=1):
        return dict(seed=100+index, index=index, decks=['first', 'second'], learner_seat=0)

    def play(self, game, *, implementation=_echo_game, timeout=1., **options):
        with patch.object(standard_imitation, 'play_standard_game', implementation):
            return standard_imitation.timed_standard_game(game,
                diagnostics=self.directory, game_timeout=timeout, **options)

    def assert_no_proxy_threads(self):
        self.assertFalse([t for t in threading.enumerate() if t.name=='game-inference-proxy'])

    def test_killed_partial_response_cannot_poison_next_game_on_either_role(self):
        learner, opponent = self.client(), self.client('opponent')
        for index, (role, mode) in enumerate((('batcher','response_header'),
                                             ('opponent_batcher','response_header'),
                                             ('batcher','response_body'),
                                             ('opponent_batcher','response_body')), start=1):
            with self.subTest(role=role, mode=mode):
                first = self.game(index*2)
                row = self.play(first, implementation=_interrupted_game, timeout=.2,
                    batcher=learner, opponent_batcher=opponent, break_role=role, break_mode=mode)
                self.assertEqual((row['status'],row['reason']),('draw','wall_clock_timeout'))
                self.assertEqual(row['timeout_seconds'],.2)
                self.assertTrue((Path(row['diagnostics'])/'interruption-reached').exists())
                self.assert_no_proxy_threads()
                learner.reset(first['seed']); opponent.reset(first['seed'])
                second = self.game(index*2+1)
                success = self.play(second, batcher=learner, opponent_batcher=opponent)
                self.assertEqual(success['status'],'win')
                for key, expected_role in [('batcher','learner'),('opponent_batcher','opponent')]:
                    self.assertEqual(success[key]['batch']['game'],second['index'])
                    self.assertEqual(success[key]['identity'],(second['seed'],key))
                    self.assertEqual(success[key]['role'],expected_role)
                self.assertFalse(learner.broken or opponent.broken)
                self.assert_no_proxy_threads()

    def test_killed_partial_request_never_reaches_reusable_server(self):
        learner = self.client()
        game = self.game()
        row = self.play(game, implementation=_interrupted_game, timeout=.15,
                        batcher=learner, break_mode='request_body')
        self.assertEqual(row['status'],'draw')
        self.assertTrue((Path(row['diagnostics'])/'interruption-reached').exists())
        learner.reset(game['seed'])
        success = self.play(self.game(2), batcher=learner)
        self.assertEqual(success['batcher']['calls'],1)
        self.assertEqual(success['batcher']['batch']['game'],2)
        self.assertFalse(learner.broken)
        self.assert_no_proxy_threads()

    def test_child_has_no_reusable_descriptors_or_prefork_proxy_threads(self):
        global _PERSISTENT_FDS
        learner, opponent = self.client(), self.client('opponent')
        _PERSISTENT_FDS = (learner.connection.fileno(),opponent.connection.fileno())
        try:
            row = self.play(self.game(), batcher=learner, opponent_batcher=opponent)
            self.assertEqual(row['child_threads'],1)
            self.assertEqual(tuple(row['closed_persistent_fds']),_PERSISTENT_FDS)
        finally:
            _PERSISTENT_FDS = ()
        self.assert_no_proxy_threads()

    def test_partial_result_frame_does_not_extend_hard_game_deadline(self):
        started = time.monotonic()
        with patch.object(standard_imitation,'_isolated_game',_partial_result):
            row = self.play(self.game(), timeout=.15)
        self.assertLess(time.monotonic()-started,1.)
        self.assertEqual((row['status'],row['timeout_seconds']),('draw',.15))
        self.assert_no_proxy_threads()

    def test_inference_timeout_poison_prevents_worker_reuse_and_retains_result(self):
        learner = self.client(timeout=.04)
        game = dict(self.game(),opponent_kind='scripted')
        with patch.object(standard_imitation,'play_standard_game',_echo_game):
            with self.assertRaises(InferenceCleanupError):
                play_ppo_game(seed=game['seed'],index=game['index'],game=game,
                    model=object(),batcher=learner,version=3,diagnostics=self.directory,
                    completed_result_directory=self.directory/'handoffs',
                    game_timeout=.5,inference_delay=.25)
        self.assertTrue(learner.broken)
        saved = json.loads((self.directory/'handoffs/game-1.json').read_text())
        self.assertEqual(saved['row']['status'],'error')
        self.assertIn('TimeoutError',saved['row']['error'])
        self.assertTrue((self.directory/'game-000001/completed-result.json').exists())
        with self.assertRaisesRegex(RuntimeError,'unusable'):
            self.play(self.game(2),batcher=learner)
        self.assert_no_proxy_threads()

    def test_completed_success_survives_cleanup_failure_before_publication(self):
        learner = self.client()
        game = dict(self.game(),opponent_kind='scripted')
        original_close = InferenceIsolation.close
        def fail_after_drain(isolation, *args, **kwargs):
            original_close(isolation,*args,**kwargs)
            raise RuntimeError('injected cleanup failure')
        with patch.object(standard_imitation,'play_standard_game',_echo_game), \
                patch.object(InferenceIsolation,'close',fail_after_drain):
            with self.assertRaisesRegex(InferenceCleanupError,'injected cleanup failure'):
                play_ppo_game(seed=game['seed'],index=game['index'],game=game,
                    model=object(),batcher=learner,version=3,diagnostics=self.directory,
                    completed_result_directory=self.directory/'handoffs',game_timeout=1.)
        saved = json.loads((self.directory/'handoffs/game-1.json').read_text())
        self.assertEqual(saved['row']['status'],'win')
        self.assertTrue(learner.broken)
        self.assert_no_proxy_threads()

    def test_repeated_games_close_disposable_file_descriptors(self):
        learner = self.client()
        initial = len(os.listdir('/proc/self/fd'))
        for index in range(8):
            row = self.play(self.game(index),batcher=learner)
            self.assertEqual(row['status'],'win')
            learner.reset(100+index)
        self.assertEqual(len(os.listdir('/proc/self/fd')),initial)
        self.assert_no_proxy_threads()

    def test_cleanup_failure_keeps_primary_error_and_closes_both_roles(self):
        learner, opponent = self.client(), self.client('opponent')
        original = InferenceIsolation.close
        def fail_after_drain(isolation, *args, **kwargs):
            original(isolation,*args,**kwargs)
            raise RuntimeError('injected cleanup failure')
        with patch.object(standard_imitation,'_isolated_game',_bad_result), \
                patch.object(InferenceIsolation,'close',fail_after_drain):
            with self.assertRaisesRegex(ValueError,'negative') as raised:
                self.play(self.game(),batcher=learner,opponent_batcher=opponent)
        self.assertIn('cleanup also failed',str(raised.exception.__notes__))
        self.assertTrue(learner.broken and opponent.broken)
        self.assert_no_proxy_threads()

    def test_failed_start_closes_all_disposable_pipes_before_any_request(self):
        learner = self.client()
        before = len(os.listdir('/proc/self/fd'))
        with patch('multiprocessing.process.BaseProcess.start',side_effect=OSError('start failed')):
            with self.assertRaisesRegex(OSError,'start failed'):
                self.play(self.game(),batcher=learner)
        self.assertEqual(len(os.listdir('/proc/self/fd')),before)
        self.assertFalse(learner.broken)
        self.assertEqual(self.play(self.game(2),batcher=learner)['status'],'win')
        self.assert_no_proxy_threads()

    def test_isolation_setup_error_survives_a_cleanup_exception(self):
        learner = self.client()
        before = len(os.listdir('/proc/self/fd'))
        original_close = InferenceIsolation.close
        def fail_after_close(isolation, *args, **kwargs):
            original_close(isolation,*args,**kwargs)
            raise OSError('injected setup close failure')
        with patch.object(InferenceIsolation,'close',fail_after_close):
            with self.assertRaisesRegex(TypeError,'worker-owned remote') as raised:
                InferenceIsolation(self.context,dict(batcher=learner,opponent_batcher=object()))
        self.assertIn('injected setup close failure',str(raised.exception.__notes__))
        self.assertEqual(len(os.listdir('/proc/self/fd')),before)
        self.assert_no_proxy_threads()

    def test_failed_diagnostic_write_keeps_completed_row_for_handoff(self):
        learner = self.client()
        game = dict(self.game(),opponent_kind='scripted')
        original_close = InferenceIsolation.close
        original_write = Path.write_text
        def fail_after_close(isolation, *args, **kwargs):
            original_close(isolation,*args,**kwargs)
            raise RuntimeError('original cleanup failure')
        def diagnostic_write(path, *args, **kwargs):
            if path.name=='completed-result.json':
                raise OSError('diagnostic disk failure')
            return original_write(path,*args,**kwargs)
        with patch.object(standard_imitation,'play_standard_game',_echo_game), \
                patch.object(InferenceIsolation,'close',fail_after_close), \
                patch.object(Path,'write_text',diagnostic_write):
            with self.assertRaisesRegex(InferenceCleanupError,'original cleanup failure') as raised:
                play_ppo_game(seed=game['seed'],index=game['index'],game=game,
                    model=object(),batcher=learner,version=3,diagnostics=self.directory,
                    completed_result_directory=self.directory/'handoffs',game_timeout=1.)
        self.assertIn('diagnostic disk failure',str(raised.exception.__notes__))
        self.assertEqual(raised.exception.completed_row['status'],'win')
        saved = json.loads((self.directory/'handoffs/game-1.json').read_text())
        self.assertEqual(saved['row'],json.loads(json.dumps(raised.exception.completed_row)))
        self.assertTrue(learner.broken)
        self.assert_no_proxy_threads()

    def test_primary_game_error_survives_both_failed_resets(self):
        class BrokenReset:
            broken=False
            def __init__(self,role):self.role=role;self.calls=0
            def reset(self,seed):
                self.calls+=1
                raise ValueError(self.role+' reset failed')
        learner,opponent = BrokenReset('learner'),BrokenReset('opponent')
        game=dict(self.game(),opponent_kind='frozen_neural',opponent_version=3,
                  opponent_checkpoint_sha256='a'*64)
        with patch.object(standard_imitation,'timed_standard_game',side_effect=RuntimeError('primary failure')):
            with self.assertRaisesRegex(RuntimeError,'primary failure') as raised:
                play_ppo_game(seed=game['seed'],index=game['index'],game=game,
                    model=object(),batcher=learner,version=3,opponent_model=object(),
                    opponent_batcher=opponent,opponent_version=3,opponent_sha256='a'*64)
        self.assertEqual((learner.calls,opponent.calls),(1,1))
        self.assertIn('learner reset failed',str(raised.exception.__notes__))
        self.assertIn('opponent reset failed',str(raised.exception.__notes__))


class BoundedFrameTests(unittest.TestCase):
    def test_stdlib_compatible_small_and_large_frames(self):
        # The payload exceeds a pipe buffer, requiring a concurrent reader.
        context = mp.get_context('fork')
        left,right = context.Pipe()
        payload = {'data':b'x'*(1024*1024)}
        process = context.Process(target=lambda: (right.send(payload),right.close()))
        process.start();right.close()
        try:
            self.assertEqual(recv_message(left,timeout=2),payload)
        finally:
            left.close();process.join(2)
            if process.is_alive():process.kill();process.join(2)
            process.close()

    def test_partial_frame_with_live_sender_has_absolute_timeout(self):
        for body in (b'\x00\x00',struct.pack('!i',4096)+b'partial'):
            with self.subTest(body=body):
                left,right = mp.get_context('fork').Pipe()
                try:
                    os.write(right.fileno(),body)
                    started=time.monotonic()
                    with self.assertRaises(TimeoutError):recv_message(left,timeout=.05)
                    self.assertLess(time.monotonic()-started,.3)
                finally:left.close();right.close()

    def test_backpressured_send_has_absolute_timeout(self):
        left,right = mp.get_context('fork').Pipe()
        try:
            started=time.monotonic()
            with self.assertRaises(TimeoutError):send_message(left,b'x'*(4*1024*1024),timeout=.05)
            self.assertLess(time.monotonic()-started,.3)
        finally:left.close();right.close()

    def test_negative_frame_length_is_rejected_without_unpickling(self):
        left,right = mp.get_context('fork').Pipe()
        try:
            os.write(right.fileno(),struct.pack('!i',-2))
            with self.assertRaisesRegex(ValueError,'negative'):recv_message(left,timeout=.1)
        finally:left.close();right.close()

    def test_extended_eight_byte_length_frame(self):
        left,right = mp.get_context('fork').Pipe()
        try:
            payload=pickle.dumps({'extended':'frame'})
            os.write(right.fileno(),struct.pack('!iQ',-1,len(payload))+payload)
            self.assertEqual(recv_message(left,timeout=.2),{'extended':'frame'})
        finally:left.close();right.close()

    def test_bounded_sender_remains_compatible_with_stdlib_receiver(self):
        left,right = mp.get_context('fork').Pipe()
        try:
            send_message(left,{'ordinary':'frame'},timeout=.2)
            self.assertEqual(right.recv(),{'ordinary':'frame'})
        finally:left.close();right.close()


if __name__=='__main__':
    unittest.main()
