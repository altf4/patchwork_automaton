"""Inference replies must belong to a complete, bounded request exchange."""
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import os
import struct
from threading import Event, Thread
import time
import unittest
from unittest.mock import patch

from patchwork_automaton.neural.inference_processes import bridge_requests
from patchwork_automaton.rl.inference_transport import recv_message, send_message
from patchwork_automaton.rl.rustmage_processes import _RemoteInference, InferenceProtocolError


class EchoBatcher:
    def __init__(self):
        self.calls = []

    def predict(self, batch, *, version, identity, unbatched):
        self.calls.append((batch, version, identity, unbatched))
        if batch == 'reject':
            raise ValueError('rejected by model')
        return dict(batch=batch, version=version, identity=identity)


@unittest.skipUnless(os.name == 'posix', 'Deadline-bounded fork workers require POSIX')
class InferenceRPCTests(unittest.TestCase):
    def pair(self, *, timeout=2.):
        server, client = mp.Pipe()
        self.addCleanup(server.close)
        self.addCleanup(client.close)
        return server, _RemoteInference(client, timeout=timeout)

    def serve(self, server, batcher):
        stopping = Event()
        thread = Thread(target=bridge_requests, args=(server, batcher, stopping), daemon=True)
        thread.start()

        def close():
            stopping.set()
            thread.join(3)
            self.assertFalse(thread.is_alive(), 'Inference bridge did not exit')
        self.addCleanup(close)
        return thread

    def test_predict_reset_and_next_predict_preserve_exact_routing(self):
        server, client = self.pair()
        batcher = EchoBatcher()
        self.serve(server, batcher)
        first = client.predict({'observation': [1, 2, 3]}, version=200,
                               identity=(12345, 'P0'), unbatched=True)
        self.assertEqual(first, dict(batch={'observation': [1, 2, 3]}, version=200,
                                     identity=(12345, 'P0')))
        client.reset(12345)
        second = client.predict('next', version=150, identity=(12346, 'P1'))
        self.assertEqual(second, dict(batch='next', version=150, identity=(12346, 'P1')))
        self.assertEqual(client.requests, 2)
        self.assertFalse(client.broken)
        self.assertEqual(len(batcher.calls), 2)

    def test_model_error_is_identified_and_does_not_desynchronize_next_request(self):
        server, client = self.pair()
        self.serve(server, EchoBatcher())
        with self.assertRaisesRegex(RuntimeError, 'rejected by model'):
            client.predict('reject', version=7, identity=(1, 0))
        self.assertFalse(client.broken)
        client.reset(1)
        self.assertEqual(client.predict('ok', version=7, identity=(2, 0))['batch'], 'ok')

    def test_concurrent_predictions_and_resets_are_serialized_without_cross_consumption(self):
        server, client = self.pair()
        batcher = EchoBatcher()
        self.serve(server, batcher)

        def exchange(index):
            if index % 3 == 0:
                client.reset(index)
                return None
            return client.predict(str(index), version=200, identity=(index, index % 2))
        with ThreadPoolExecutor(max_workers=8) as pool:
            replies = list(pool.map(exchange, range(24)))
        for index, reply in enumerate(replies):
            if index % 3:
                self.assertEqual(reply, dict(batch=str(index), version=200,
                                            identity=(index, index % 2)))
            else:
                self.assertIsNone(reply)
        self.assertEqual(client.requests, 16)
        self.assertEqual(len(batcher.calls), 16)
        self.assertFalse(client.broken)

    def test_invalidation_close_error_cannot_mask_transport_failure(self):
        server, client = self.pair(timeout=.05)
        with patch.object(client.connection, 'close', side_effect=OSError('damaged descriptor')):
            with self.assertRaises(TimeoutError):
                client.predict('input', version=1, identity=(1, 0))
        self.assertTrue(client.broken)
        self.assertIn('damaged descriptor', client.invalidation_error)

    def test_wrong_or_unidentified_responses_poison_the_channel(self):
        def wrong_sequence(r): return (r[0], (r[1][0], r[1][1] + 1), *r[2:4], True, {})
        def wrong_session(r): return (r[0], ('foreign-session', r[1][1]), *r[2:4], True, {})
        mutations = {
            'sequence': wrong_sequence,
            'session': wrong_session,
            'version': lambda r: (*r[:2], r[2] + 1, r[3], True, {}),
            'identity': lambda r: (*r[:3], ('different-game', 0), True, {}),
            'tag': lambda r: ('unknown', *r[1:4], True, {}),
            'unidentified': lambda r: (True, {}),
            'non_boolean_status': lambda r: (*r[:4], 1, {}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                server, client = self.pair()
                errors = []

                def reply():
                    try:
                        request = recv_message(server, timeout=2.)
                        send_message(server, mutate(request), timeout=2.)
                    except BaseException as error:
                        errors.append(error)
                thread = Thread(target=reply, daemon=True)
                thread.start()
                try:
                    with self.assertRaisesRegex(InferenceProtocolError, 'does not match'):
                        client.predict('observation', version=9, identity=(22, 'P1'))
                    self.assertTrue(client.broken)
                    self.assertTrue(client.connection.closed)
                    with self.assertRaisesRegex(InferenceProtocolError, 'unusable'):
                        client.predict('next', version=9, identity=(23, 'P1'))
                finally:
                    thread.join(3)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])

    def test_wrong_reset_acknowledgement_poison_the_channel(self):
        server, client = self.pair()

        def reply():
            request = recv_message(server, timeout=2.)
            send_message(server, (*request[:2], request[2] + 1), timeout=2.)
        thread = Thread(target=reply, daemon=True)
        thread.start()
        try:
            with self.assertRaisesRegex(InferenceProtocolError, 'reset does not match'):
                client.reset(44)
            self.assertTrue(client.broken)
        finally:
            thread.join(3)
        self.assertFalse(thread.is_alive())

    def test_partial_response_from_live_sender_has_a_full_frame_deadline(self):
        server, client = self.pair(timeout=.15)
        release = Event()
        errors = []

        def reply():
            try:
                recv_message(server, timeout=2.)
                os.write(server.fileno(), struct.pack('!i', 4096) + b'partial')
                release.wait(2.)
            except BaseException as error:
                errors.append(error)
        thread = Thread(target=reply, daemon=True)
        thread.start()
        started = time.monotonic()
        try:
            with self.assertRaises(TimeoutError):
                client.predict('observation', version=3, identity=(100, 0))
            self.assertLess(time.monotonic() - started, 1.)
            self.assertTrue(client.broken)
            self.assertTrue(client.connection.closed)
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_duplicate_request_sequence_is_rejected_before_model_execution(self):
        server, client = self.pair()
        batcher = EchoBatcher()
        thread = self.serve(server, batcher)
        request = ('inference-v1', ('session', 1), 200, (42, 0), False, 'input')
        send_message(client.connection, request, timeout=2.)
        self.assertEqual(recv_message(client.connection, timeout=2.)[:4], request[:4])
        send_message(client.connection, request, timeout=2.)
        with self.assertRaises((EOFError, ConnectionResetError)):
            recv_message(client.connection, timeout=2.)
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(batcher.calls), 1)


if __name__ == '__main__':
    unittest.main()
