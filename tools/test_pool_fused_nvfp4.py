#!/usr/bin/env python3
"""Run real-blob parity and require once-per-pool *executed dispatch* traces.
Usage: python3 tools/test_pool_fused_nvfp4.py EXE FIXTURE_DIR LOG_DIR
"""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    exe, fixture, logs = map(Path, sys.argv[1:])
    logs.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((fixture / 'manifest.json').read_text())
    blobs = [str(fixture / row['file']) for row in manifest]
    cases = [
        ('default', {'STRATA_POOL_SPIN_US': '0', 'STRATA_POOL_FUSED_TRACE': '1'}, 'parts=8,4 lag=1'),
        ('lag0-parts7-3', {'STRATA_POOL_SPIN_US': '0', 'STRATA_POOL_FUSED_TRACE': '1',
                         'STRATA_POOL_FUSED_LAG': '0', 'STRATA_POOL_FUSED_PARTS': '7,3'}, 'parts=7,3 lag=0'),
        ('lag-clamped-parts64', {'STRATA_POOL_SPIN_US': '0', 'STRATA_POOL_FUSED_TRACE': '1',
                               'STRATA_POOL_FUSED_LAG': '2147483647', 'STRATA_POOL_FUSED_PARTS': '64,64'}, 'parts=64,64 lag=96'),
        ('trace-off', {'STRATA_POOL_FUSED_TRACE': '0'}, None),
    ]
    for name, settings, expected in cases:
        env = os.environ.copy()
        for key in ('STRATA_POOL_FUSED', 'STRATA_POOL_FUSED_TRACE', 'STRATA_POOL_FUSED_LAG',
                    'STRATA_POOL_FUSED_PARTS', 'STRATA_POOL_SPIN_US'):
            env.pop(key, None)
        env.update(settings)
        log = logs / (name + '.log')
        with log.open('w') as output:
            result = subprocess.run([str(exe.resolve()), *blobs], stdout=output, stderr=subprocess.STDOUT,
                                    env=env, timeout=600, check=False)
        text = log.read_text()
        if result.returncode:
            raise RuntimeError(f'{name}: exit={result.returncode}\n{text[-4000:]}')
        traces = [line for line in text.splitlines() if line.startswith('[STRATA_POOL_FUSED]')]
        if expected is None:
            if traces:
                raise AssertionError('trace disabled but logging occurred')
        elif len(traces) != 6 or any('gu_type=40 d_type=40' not in t or expected not in t for t in traces):
            raise AssertionError(f'{name}: missing/wrong executed fused traces: {traces}')
        summary = [line for line in text.splitlines() if line.startswith('PASS memcmp_windows=')]
        if len(summary) != 1 or 'memcmp_windows=2048 layer_calls=4096 fused_calls=1248' not in summary[0]:
            raise AssertionError(f'{name}: wrong coverage: {summary}')
        print(f'{name}: traces={len(traces)} {summary[0]}', flush=True)
    print('PASS four scheduler configurations; 8192 memcmp windows / 16384 layer calls', flush=True)


if __name__ == '__main__':
    main()
