"""Game-state reconstruction and typed prompt/card attribute decoding."""
import struct

from .protocol import Cursor, ProtocolError, MAX_FRAME

from .properties import PROPERTY_NAMES, freeze
from .interactions import DamageAssignment, DamageTarget, ManaEntry, Ordering, PromptCode


def zstring(c, width=0):
    start = c.pos
    data = bytearray()
    while True:
        value = c.take(1)
        if value == b"\0":
            break
        data.extend(value)
    if width:
        c.take(width - (c.pos - start))
    return data.decode("latin-1")


def attributes(c, strings=(), depth=0):
    if depth > 32:
        raise ProtocolError("Attribute nesting limit exceeded")
    result = {}
    while c.pos < len(c.data):
        raw = struct.unpack("<I", c.take(4))[0]
        if raw == 0:
            break
        kind = raw & 0x78000000
        normalized = raw - 0x08000000 if kind == 0x28000000 else raw
        name = PROPERTY_NAMES.get(str(normalized), f"attribute_{normalized}")
        if kind == 0x40000000:
            length = struct.unpack("<H", c.take(2))[0]
            if length == 65535:
                index = struct.unpack("<I", c.take(4))[0]
                value = strings[index] if index < len(strings) else f"#str:{index}"
            else:
                value = c.take(length).decode("latin-1")
        elif kind == 0x20000000:
            value = struct.unpack("<b", c.take(1))[0]
        elif kind == 0x28000000:
            value = c.integer()
        elif kind == 0x08000000:
            value = attributes(c, strings, depth+1)
        elif kind == 0x10000000:
            value = True
        else:
            raise ProtocolError(f"Unsupported attribute type {kind}")
        result[name] = value
    return result


def interaction(data, legacy=False):
    c = Cursor(data)
    turn, phase, timestamp = struct.unpack("<iiI", c.take(12))
    prompt = dict(turn=turn, phase=phase+1, timestamp=timestamp, buttons=[], mana=[], selectors=[])
    if legacy:
        flags, default = struct.unpack("<II", c.take(8))
        minimum, maximum, initial, player, special_player, special_type = c.take(6)
        labels = [zstring(c,32) for _ in range(4)]
        for i, label in enumerate(labels):
            code = 0x08000000 << i
            if flags & code:
                prompt['buttons'].append(dict(code=code,label=label,enabled=True))
        for code,label in ((1,'OK'),(2,'Cancel'),(4,'Yes'),(8,'No')):
            if flags & code:
                prompt['buttons'].append(dict(code=code,label=label,enabled=True))
        if flags & PromptCode.DAMAGE_OK and not flags & PromptCode.OK:
            prompt['buttons'].append(dict(code=int(PromptCode.DAMAGE_OK), label='OK', enabled=True))
        prompt.update(flags=flags, default=default, minimum=minimum, maximum=maximum, initial=initial, player=player, text=zstring(c))
        if flags & PromptCode.NUMBER:
            prompt['selectors'].append(dict(id=0, kind='number', code=int(PromptCode.NUMBER),
                minimum=minimum, maximum=maximum, initial=maximum if flags & 0x800000 else initial))
        if flags & PromptCode.PLAYER:
            prompt['selectors'].append(dict(id=len(prompt['selectors']), kind='player', code=int(PromptCode.PLAYER)))
        if flags & PromptCode.WISH:
            prompt['selectors'].append(dict(id=len(prompt['selectors']), kind='wish', code=int(PromptCode.WISH)))
        if flags == PromptCode.PILE:
            prompt['selectors'].append(dict(id=len(prompt['selectors']), kind='pile', code=int(PromptCode.PILE),
                                           options=[(1, 'Pile 1'), (2, 'Pile 2')]))
        if flags & 0x3f0:
            prompt['mana'].append(dict(player=player, flags=flags & 0x3f0))
    else:
        player,special_player,special_type = c.take(3)
        prompt.update(player=player,text=zstring(c))
        n = c.take(1)[0]
        for selector_id in range(n):
            size = c.integer()
            kind = c.take(1)[0]
            ui = Cursor(c.take(size-5))
            if kind == 1:
                code = ui.integer()
                hotkey,enabled = ui.take(2)
                prompt['buttons'].append(dict(code=code,label=zstring(ui),enabled=bool(enabled)))
            elif kind == 2:
                player = ui.take(1)[0]
                flags = struct.unpack('<I',ui.take(4))[0]
                prompt['mana'].append(dict(player=player,flags=flags))
            elif kind == 3:
                minimum,maximum,initial = ui.take(3)
                prompt['selectors'].append(dict(id=selector_id, kind='number', code=int(PromptCode.NUMBER),
                    minimum=minimum, maximum=maximum, initial=initial))
            elif kind == 4:
                prompt['selectors'].append(dict(id=selector_id, kind='player', code=int(PromptCode.PLAYER)))
            elif kind == 5:
                source, invert, number = ui.integer(), bool(ui.take(1)[0]), ui.take(1)[0]
                prompt['selectors'].append(dict(id=selector_id, kind='list', code=int(PromptCode.NAME),
                    source=source, invert=invert, except_values=[ui.integer() for _ in range(number)]))
            elif kind == 6:
                flags = struct.unpack('<I', ui.take(4))[0]
                prompt['selectors'].append(dict(id=selector_id, kind='color', code=0,
                    options=[(code, label) for code, label in ((16,'Black'),(32,'Blue'),(64,'Green'),
                             (128,'Red'),(256,'White'),(512,'Colorless')) if flags & code]))
            elif kind == 7:
                options = []
                for _ in range(ui.take(1)[0]):
                    texture, enabled = ui.integer(), bool(ui.take(1)[0])
                    if enabled:
                        options.append((texture, f'Card texture {texture}'))
                prompt['selectors'].append(dict(id=selector_id, kind='card', code=int(PromptCode.NAME), options=options))
            elif kind == 8:
                prompt['selectors'].append(dict(id=selector_id, kind='wish', code=int(PromptCode.WISH)))
            else:
                prompt['selectors'].append(dict(type=kind,bytes=ui.data.hex()))
                continue
            ui.end()
    # The official UI decoder stops after its descriptors; current ordinary
    # priority prompts have one unused trailing zero byte.
    if c.pos<len(c.data) and not any(c.data[c.pos:]):
        c.take(len(c.data)-c.pos)
    c.end()
    return prompt


def checksum(data):
    value = 826366246
    for byte in data:
        value = ((value << 1) + byte) & 0xffffffff
    return value


def apply_diff(previous, delta, expected_size):
    c = Cursor(delta)
    position, output = 0, bytearray()
    def copy(length, offset):
        nonlocal position
        position += offset
        if position < 0 or position+length > len(previous):
            raise ProtocolError("Diff source range invalid")
        output.extend(previous[position:position+length])
        position += length
    while c.pos < len(delta):
        tag = c.take(1)[0]
        if tag == 0:
            length,offset = struct.unpack('<Hh',c.take(4))
            copy(length,offset)
        elif tag & 128:
            length = tag & 127
            if length:
                copy(length,struct.unpack('<b',c.take(1))[0])
            else:
                length = c.take(1)[0]
                if length:
                    copy(length,struct.unpack('<h',c.take(2))[0])
                else:
                    length = int.from_bytes(c.take(3),'little')
                    output.extend(c.take(length))
        else:
            output.extend(c.take(tag))
        if len(output)>expected_size:
            raise ProtocolError("Diff output exceeded expected size")
    if len(output)!=expected_size:
        raise ProtocolError("Diff output size mismatch")
    return bytes(output)


class State:
    def __init__(self, strings=()):
        self.previous = {}
        self.assembly = bytearray()
        self.strings = strings
        self.prompt = None
        self.cards = {}
        self.players = []
        self.raw = b''
        self.unknown_elements = ()
        self.status = {}
        self.mana = ()
        self.damage = ()
        self.ordering = ()

    def update(self,payload):
        c = Cursor(payload)
        fragment = c.blob()
        clocks = [c.integer() for _ in range(8)]
        waiting = c.integer()
        game_id,size,undiffed,elements = struct.unpack('<4I',c.take(16))
        priority,expected,prior_expected = struct.unpack('<3i',c.take(12))
        replay,flags,timestamp = struct.unpack('<BBI',c.take(6))
        c.end()
        assembly_size = len(fragment) + (0 if flags & 2 else len(self.assembly))
        if max(size, undiffed, assembly_size) > MAX_FRAME or elements > MAX_FRAME // 8:
            raise ProtocolError('Game state exceeds limit')
        if flags & 2:
            self.assembly = bytearray()
        self.assembly.extend(fragment)
        if not flags & 4:
            return False
        data = bytes(self.assembly)
        if flags & 1:
            previous = self.previous.get(priority)
            if previous is None or checksum(previous)!=(prior_expected & 0xffffffff):
                raise ProtocolError("Previous state checksum mismatch")
            data = apply_diff(previous,data,undiffed)
        if checksum(data)!=(expected & 0xffffffff):
            raise ProtocolError("State checksum mismatch")
        if len(data) != undiffed:
            raise ProtocolError('Uncompressed state size mismatch')
        self.previous[priority] = data
        self.raw = data
        c = Cursor(data)
        cards,players,prompt,unknown,mana,damage,ordering = {},[],None,[],[],[],[]
        for _ in range(elements):
            length,kind = struct.unpack('<ii',c.take(8))
            raw = c.take(length-8)
            if kind in (3,9):
                prompt = interaction(raw,legacy=kind==3)
            elif kind == 5:
                obj = Cursor(raw)
                from_zone = obj.integer()
                props = attributes(obj,self.strings)
                obj.end()
                object_id = props.get('THINGNUMBER')
                if object_id is not None:
                    cards[object_id] = dict(id=object_id,from_zone=from_zone,properties=props)
            elif kind == 2:
                obj=Cursor(raw)
                life=list(struct.unpack('<16h',obj.take(32)))
                hand=list(struct.unpack('<16h',obj.take(32)))
                library=list(struct.unpack('<16h',obj.take(32)))
                graveyard=list(struct.unpack('<16h',obj.take(32)))
                players.append(dict(life=life,hand=hand,library=library,graveyard=graveyard))
                if obj.pos < len(raw):
                    unknown.append((kind, raw[obj.pos:]))
            elif kind == 4:
                obj = Cursor(raw)
                player = obj.integer()
                while obj.pos < len(raw):
                    mana_id, amount, color = struct.unpack('<iii', obj.take(12))
                    mana.append(ManaEntry(player, mana_id, amount, color, freeze(attributes(obj, self.strings))))
            elif kind == 7:
                obj = Cursor(raw)
                source, assignment_flags, total, number = struct.unpack('<4i', obj.take(16))
                if not 0 <= number <= len(raw) // 16:
                    raise ProtocolError('Invalid damage assignment count')
                targets = tuple(DamageTarget(*struct.unpack('<4i', obj.take(16))) for _ in range(number))
                obj.end()
                damage.append(DamageAssignment(source, total, targets, assignment_flags))
            elif kind == 8:
                obj = Cursor(raw)
                source, number = obj.integer(), obj.integer()
                if not 0 <= number <= len(raw) // 4:
                    raise ProtocolError('Invalid ordering count')
                targets = tuple(obj.integer() for _ in range(number))
                obj.end()
                ordering.append(Ordering(source, targets))
            else:
                unknown.append((kind, raw))
        c.end()
        self.prompt,self.cards,self.players = prompt,cards,players
        self.unknown_elements = tuple(unknown)
        self.mana, self.damage, self.ordering = tuple(mana), tuple(damage), tuple(ordering)
        self.status = dict(game_id=game_id, clocks=clocks, waiting_for_player=waiting,
                           priority_player=priority, timestamp=timestamp, checksum=expected & 0xffffffff)
        return True
