import asyncio
import json
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace

from patchwork_automaton.protocol import Catalog, ProtocolError, array, login_response, read_frame, string
from patchwork_automaton.client import MTGOClient
from research.session import credentials

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs/discovery/mtgo-message-catalog.json"


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.catalog = Catalog(CATALOG)

    async def test_fragmented_stream_and_adjacent_frames(self):
        reader = asyncio.StreamReader()
        one = self.catalog.frame("HeartbeatReqMessage", struct.pack("<I", 123))
        two = self.catalog.frame("HeartbeatReqMessage", struct.pack("<I", 456))
        task = asyncio.create_task(read_frame(reader))
        reader.feed_data(one[:2])
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        reader.feed_data(one[2:7])
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        reader.feed_data(one[7:] + two)
        self.assertEqual(await task, one)
        self.assertEqual(await read_frame(reader), two)

    async def test_invalid_lengths_rejected_before_payload(self):
        for size in (-1, 4, 17 * 1024 * 1024):
            reader = asyncio.StreamReader()
            reader.feed_data(struct.pack("<i", size))
            with self.assertRaises(ProtocolError):
                await read_frame(reader)

    def test_type_check_rejects_incompatible_schema(self):
        data = bytearray(self.catalog.frame("HeartbeatReqMessage", b"\0" * 4))
        data[6] ^= 1
        with self.assertRaises(ProtocolError):
            self.catalog.parse(data)

    def test_wide_string_counts_utf16_units(self):
        self.assertEqual(string("\U0001f600", True), b"\x02\0\0\0=\xd8\0\xde")

    def test_complete_login_response_and_truncation(self):
        # Distinct synthetic profile values exercise every variable-size field.
        user = struct.pack("<i", 123) + string("tester", True)
        user += string("profile", True) + string("shopper", True) + struct.pack("<i", 7)
        user += b"".join(string(s, True) for s in ("comment", "name", "email"))
        user += struct.pack("<i", 8) + string("clan", True) + b"\0"
        user += b"\0" * 27 + struct.pack("<6i", *range(6))
        user += array(b"x" * 64) + string("away", True)
        user += struct.pack("<4i", 1, 2, 0, 0x1000002)
        user += struct.pack("<i2i", 2, 10, 11) + struct.pack("<2i", 12, 13)
        tail = struct.pack("<iiBBi", 1, 1, 0, 0, 0) + string("connection", True)
        tail += b"\1" + struct.pack("<II", 1700000000, 500) + b"s" * 16 + struct.pack("<i", 1)
        response = login_response(user + tail)
        self.assertEqual(response.response_code, 1)
        self.assertEqual(response.client_flags, 0x1000002)
        self.assertEqual(response.user_id, 123)
        self.assertEqual(response.server_time, tail[-29:-20])
        self.assertNotIn("connection", repr(response))
        with self.assertRaises(ProtocolError):
            login_response((user + tail)[:-1])

    async def test_nested_ack_and_heartbeat_echo(self):
        with tempfile.TemporaryDirectory() as directory:
            session = MTGOClient(catalog=self.catalog)
            sent = []
            async def write(frame):
                sent.append(frame)
            session._write = write
            heartbeat = self.catalog.frame("HeartbeatReqMessage", struct.pack("<I", 99))
            envelope = self.catalog.frame("AckMessageReqMessage", struct.pack("<I", 42) + array(heartbeat))
            await session._dispatch(*await session._unwrap(envelope))
            self.assertEqual(self.catalog.parse(sent[0]), (17, struct.pack("<I", 42)))
            self.assertEqual(self.catalog.parse(sent[1]), (23, struct.pack("<I", 99)))
            self.assertEqual(session.heartbeats_answered, 1)

    def test_credentials_preserve_delimiters_in_password(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "creds"
            path.write_text("username: test\npassword: secret=with:delimiters\n")
            self.assertEqual(credentials(path), ("test", "secret=with:delimiters"))


if __name__ == "__main__":
    unittest.main()
