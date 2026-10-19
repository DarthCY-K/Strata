#!/usr/bin/env python3
"""Run native CUDA parity under the shared external GPU lock (not numerical simulation)."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument("binary", type=Path)
p.add_argument("--devices", default="0,1", help="physical CUDA device indices; one visible device per process")
p.add_argument("--output", type=Path, default=Path("evidence/green"))
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
variants = [("default", None, None, 0), ("off", "0", None, 0),
            ("on", "1", None, 1), ("invalid", "true", None, 0),
            ("trace-on", "1", "1", 1), ("trace-off", "1", "0", 1)]
results = []
for device in a.devices.split(","):
    for label, mapped, trace, expected in variants:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=device, CUDA_DEVICE_MAX_CONNECTIONS="1", CUDA_MODULE_LOADING="LAZY")
        for name, value in [("STRATA_REMOTE_REDUCE_MAPPED", mapped), ("STRATA_REMOTE_REDUCE_TRACE", trace)]:
            env.pop(name, None)
            if value is not None:
                env[name] = value
        r = subprocess.run([str(a.binary.resolve()), "0", str(expected)], env=env,
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (a.output / f"gpu{device}-{label}.log").write_text(r.stdout)
        match = re.search(r"cases=(\d+) graph_checks=(\d+) graph_skips=(\d+) hash=([0-9a-f]+) device_delta_bytes=(\d+)", r.stdout)
        trace_lines = [line for line in r.stdout.splitlines() if line.startswith("strata: remote reduce ")]
        ok = r.returncode == 0 and match is not None
        if trace == "1":
            ok = ok and len(trace_lines) == 1 and "enabled=1 zero_copy=1 path=mapped-host" in trace_lines[0]
        else:
            ok = ok and not trace_lines
        record = dict(physical_device=int(device), variant=label, returncode=r.returncode, passed=bool(ok))
        if match:
            record.update(cases=int(match[1]), graph_checks=int(match[2]), graph_skips=int(match[3]),
                          hash=match[4], device_delta_bytes=int(match[5]))
        results.append(record)
        (a.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"gpu={device} variant={label} passed={ok}\n{r.stdout}", flush=True)
        if not ok:
            raise SystemExit(1)
assert len({r["hash"] for r in results}) == 1, "different output bytes across modes/devices"
print(json.dumps(dict(processes=len(results), cases=sum(r["cases"] for r in results),
                      graph_checks=sum(r["graph_checks"] for r in results),
                      graph_skips=sum(r["graph_skips"] for r in results), all_passed=True)))
