"""Solitaire discovery codecs. Game decisions are not automated yet."""
from dataclasses import dataclass
import struct
import uuid

from .protocol import Cursor, ProtocolError, string


@dataclass(frozen=True)
class DeckItem:
    catalog_id: int
    annotation: int
    permission: int
    quantity: int
    sideboard: bool

    def encode(self):
        return struct.pack("<iIii?", self.catalog_id, self.annotation, self.permission,
                           self.quantity, self.sideboard)


def decks_response(payload):
    c = Cursor(payload)
    decks = []
    for _ in range(2):  # new and updated decks
        count = c.integer()
        if not 0 <= count <= 10000:
            raise ProtocolError("Invalid deck count")
        for _ in range(count):
            deck_id, name, description, digest = c.integer(), c.string(), c.string(), c.string(False)
            count_cards = c.integer()
            if not 0 <= count_cards <= 100000:
                raise ProtocolError("Invalid deck item count")
            cards = [DeckItem(*struct.unpack("<iIii?", c.take(17))) for _ in range(count_cards)]
            decks.append(dict(deck_id=deck_id, name=name, description=description,
                              cards=[vars(card) for card in cards]))
    c.blob(4); c.blob(4)  # removed and unchanged IDs
    c.integer()  # last binder
    response = c.integer()
    c.end()
    return response, decks


def solitaire_payload(token: uuid.UUID, cards, game_structure, match_type="SEL", channel=6):
    if not game_structure or len(game_structure) > 64:
        raise ProtocolError("Invalid format identifier")
    return b"".join([
        token.bytes_le, string(match_type, True), string(game_structure, True),
        string("SOLO", True), string("", True), b"\0\0",
        struct.pack("<ii?iii", channel, 60, False, 0, 0, 0),
        string("bgname", True), struct.pack("<i", len(cards)),
        b"".join(DeckItem(**card).encode() for card in cards),
        struct.pack("<ii", 0, 0),  # not preconstructed, no game-server options
    ])


def game_status(payload):
    c = Cursor(payload)
    state = c.blob()
    clocks = [c.integer() for _ in range(8)]
    waiting = c.integer()
    game_id, state_size, undiffed_size, elements = struct.unpack("<4I", c.take(16))
    priority, checksum, previous_checksum = struct.unpack("<3i", c.take(12))
    replaying, flags, timestamp = struct.unpack("<BBI", c.take(6))
    c.end()
    return dict(game_id=game_id, priority_player=priority, waiting_for_player=waiting,
                game_state_timestamp=timestamp, state_bytes=len(state), state_elements=elements,
                state_size=state_size, undiffed_size=undiffed_size, flags=flags,
                full_state=not bool(flags & 1), head=bool(flags & 2), tail=bool(flags & 4))


def count(c):
    n = c.integer()
    if not 0 <= n <= 100000:
        raise ProtocolError("Invalid event array count")
    return n


def event_sync(c):
    """Normal match events only; deliberately excludes tournament structures."""
    token = uuid.UUID(bytes_le=c.take(16))
    event_id = c.integer()
    c.integer(); c.take(16); c.integer()
    structure = c.string()
    player_count = count(c)
    for _ in range(player_count):
        c.take(8); c.string(); c.take(5)
    c.integer()
    for _ in range(count(c)):
        c.integer(); c.string(); c.integer()
    state = c.integer()
    c.string(); c.take(8); c.take(18); c.blob(2); c.string(); c.string()
    games, player_format, minimum, maximum = [], None, None, None
    if c.take(1) == b"\1":
        c.integer()  # Host game-server ID
        c.take(16); c.take(8)  # MatchCreateInfo token, ID, parent type
        c.string(); c.string()
        player_format = c.string()
        # Response MatchCreateInfo uses int32 AllowStartHandicap, whereas the
        # creation request uses a boolean. They are different wire layouts.
        c.integer(); c.string(); c.take(5); c.take(12); c.take(1)
        c.take(18); c.take(4); c.take(1); c.blob(8)
        minimum, maximum = c.integer(), c.integer()
        for _ in range(count(c)):
            game_id, game_state = c.integer(), c.integer()
            c.blob(12)  # Fixed-size player rankings
            games.append(dict(game_id=game_id, game_state=game_state))
        c.integer(); c.string(); c.take(9)
        for _ in range(count(c)):
            c.take(8); c.string()
        for _ in range(count(c)):
            c.integer(); c.string(); c.take(5)
    if c.take(3) != b"\0\0\0":
        raise ProtocolError("Tournament event decoding is not implemented")
    return dict(token=token, event_id=event_id, state=state, structure=structure,
                player_format=player_format, minimum_players=minimum,
                maximum_players=maximum, players=player_count, games=games)
