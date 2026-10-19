#!/usr/bin/env python3
"""Copy complete NVFP4 expert blobs, including their 16-byte scale tails.

Usage: python3 tools/extract_pool_nvfp4_fixture.py PACK_DIR OUTPUT_DIR
Does not write to the model pack. Records exact source offsets and hashes.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def main():
    pack, dest = map(Path, sys.argv[1:])
    rows = {}
    for line in (pack / 'native_experts.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            layer, gu, down, offset, size = map(int, line.split())
            rows[layer] = (gu, down, offset, size)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    with (pack / 'experts.bin').open('rb') as source:
        for layer, expert in [(0, 0), (0, 17), (7, 63), (23, 127), (47, 255), (47, 511)]:
            gu, down, base, size = rows[layer]
            if (gu, down, size) != (40, 40, 2764816):
                raise ValueError('expected H=2560 FF=640 NVFP4/NVFP4 with scale tail')
            offset = base + expert * size
            source.seek(offset)
            blob = source.read(size)
            if len(blob) != size:
                raise ValueError('short source read')
            scales = struct.unpack('<4f', blob[-16:])
            if not all(math.isfinite(s) and s > 0 for s in scales[:3]):
                raise ValueError('invalid source scale tail')
            out = dest / f'layer{layer:02d}-expert{expert:03d}.bin'
            with out.open('xb') as target:
                target.write(blob)
            if out.read_bytes() != blob:
                raise ValueError('fixture readback differs')
            manifest.append(dict(file=out.name, source=str(pack / 'experts.bin'),
                                 layer=layer, expert=expert, offset=offset, bytes=size,
                                 scales=scales, sha256=hashlib.sha256(blob).hexdigest()))
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
