"""Binary primitives recovered from client 3.4.159.4704."""
import asyncio
import hashlib
import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

MAX_FRAME = 16 * 1024 * 1024


class ProtocolError(ValueError):
    pass


class Catalog:
    def __init__(self, path: Path | None = None):
        path = path or Path(__file__).with_name('data') / 'messages.json'
        entries = json.loads(path.read_text())["messages"]
        self.by_name = {e["name"]: e for e in entries}
        self.by_opcode = {e["opcode"]: e for e in entries}

    def frame(self, name: str, payload: bytes = b"") -> bytes:
        if len(payload) + 8 > MAX_FRAME:
            raise ProtocolError('Outgoing frame exceeds limit')
        entry = self.by_name[name]
        return struct.pack("<iHH", 8 + len(payload), entry["opcode"], entry["type_check"]) + payload

    def parse(self, data: bytes):
        if len(data) < 8:
            raise ProtocolError("Short frame header")
        length, opcode, check = struct.unpack_from("<iHH", data)
        if length != len(data) or length > MAX_FRAME:
            raise ProtocolError("Invalid frame length")
        entry = self.by_opcode.get(opcode)
        if entry and entry["type_check"] != check:
            raise ProtocolError(f"Type check mismatch for opcode {opcode}")
        return opcode, data[8:]


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    header = await reader.readexactly(4)
    length = struct.unpack("<i", header)[0]
    if not 8 <= length <= MAX_FRAME:
        raise ProtocolError("Invalid incoming frame length")
    return header + await reader.readexactly(length - 4)


def string(value: str, wide: bool = False, limit: int | None = None) -> bytes:
    data = value.encode("utf-16-le" if wide else "latin-1")
    count = len(data) // 2 if wide else len(data)
    if limit is not None and count > limit:
        raise ProtocolError("String exceeds protocol limit")
    return struct.pack("<i", count) + data


def array(data: bytes) -> bytes:
    return struct.pack("<i", len(data)) + data


def current_time() -> bytes:
    milliseconds = time.time_ns() // 1_000_000
    return struct.pack("<BII", 1, milliseconds // 1000, milliseconds % 1000)


def login_payload(username: str, password: str, device: str, session_id: str,
                  challenge: str = "") -> bytes:
    # This digest is a credential. Never include payloads in logs.
    digest = hashlib.md5(password.encode("utf-8"), usedforsecurity=False).hexdigest().upper()
    return b"".join([
        string(username, True, 30), string(digest, False, 32),
        string(device, False, 64), string(challenge, False, 6), b"\x00",
        string("7f5223f1b14593e4d02021d496770682", True, 32),
        struct.pack("<i", 2 | 0x800000 | 0x1000000), string(session_id, True),
    ])


class Cursor:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, length: int) -> bytes:
        if length < 0 or self.pos + length > len(self.data):
            raise ProtocolError("Truncated or invalid payload")
        result = self.data[self.pos:self.pos + length]
        self.pos += length
        return result

    def integer(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def string(self, wide=True) -> str:
        count = self.integer()
        return self.take(count * (2 if wide else 1)).decode("utf-16-le" if wide else "latin-1")

    def blob(self, element_size=1) -> bytes:
        return self.take(self.integer() * element_size)

    def end(self):
        if self.pos != len(self.data):
            raise ProtocolError(f"Unexpected trailing payload at {self.pos}/{len(self.data)}")


@dataclass(repr=False)
class LoginResponse:
    response_code: int
    authorization_type: int
    client_flags: int
    user_id: int = field(repr=False)
    server_time: bytes = field(repr=False)
    connection_id: str = field(repr=False)
    sentinel: bytes = field(repr=False)


def login_response(payload: bytes) -> LoginResponse:
    """Consume UserInfo_t without retaining personal profile fields."""
    c = Cursor(payload)
    user_id = c.integer()
    c.string()  # ScreenName
    c.string(); c.string()  # Profile / shopper type
    c.integer()  # Avatar
    for _ in range(3):
        c.string()  # Comment, unofficial name/email
    c.integer(); c.string(); c.take(1)  # Clan
    c.take(27)  # Three WOTCDateTime fields (9 bytes each)
    c.take(24)  # Role and five privacy fields
    userdata = c.blob()
    if len(userdata) != 64:
        raise ProtocolError("Unexpected fixed UserData length")
    c.string()  # Away text
    c.take(12)  # Icon, permission, challenge type
    flags = c.integer()
    c.blob(4)  # Blocked users
    c.take(8)  # Rating decay counters
    c.take(10)  # Two eligibility ints, two booleans
    authorization = c.integer()
    connection = c.string()
    server_time = c.take(9)
    sentinel = c.take(16)
    response = c.integer()
    c.end()
    return LoginResponse(response, authorization, flags, user_id, server_time, connection, sentinel)
