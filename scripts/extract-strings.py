"""Extract an installed client's string table for local use (requires dnfile)."""
import argparse
import json
from pathlib import Path
import struct

import dnfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('assembly', type=Path, help='Installed MTGO GamePlay assembly containing strconst_v3.dat')
parser.add_argument('output', type=Path)
args = parser.parse_args()
pe = dnfile.dnPE(str(args.assembly))
for resource in pe.net.resources:
    for entry in getattr(resource.data, 'entries', ()):
        if entry.name == 'strconst/strconst_v3.dat':
            raw, pos = entry.value, 4
            count = struct.unpack_from('<i', raw)[0]
            strings = []
            for _ in range(count):
                size = struct.unpack_from('<i', raw, pos)[0]
                pos += 4
                if size < 1 or pos + size > len(raw) or raw[pos + size - 1] != 0:
                    raise ValueError('Invalid string table')
                strings.append(raw[pos:pos + size - 1].decode('utf-8'))
                pos += size
            args.output.write_text(json.dumps(strings))
            print(f'Extracted {len(strings)} strings')
            raise SystemExit(0)
raise SystemExit('String table resource not found')
