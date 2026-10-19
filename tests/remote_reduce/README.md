# Opt-in mapped helper reduction

`STRATA_REMOTE_REDUCE_MAPPED=1` sends the existing reduction kernel's output to
`RemoteExperts::z_out_` **only when `zero_copy_` is true**. Unset, `0`, or any value
other than exactly `1` keeps the original device-sum + D2H copy. Both settings
are cached on first use: set them before starting the process / capturing graphs.
The kernel and arithmetic are unchanged. The existing `finish()` stream wait
precedes host consumption. Device-sum allocation remains for fallback.

`STRATA_REMOTE_REDUCE_TRACE=1` prints the first selected path once per attached
helper. It is off by default; leave it off for performance A/B. The trace reports
the first call, not later test-only changes to a helper's `zero_copy_` flag.

## Standalone linked CUDA test

The executable links the actual `remote_expert_opt.cu`, `remote_experts.cpp`,
`expert_cache.cpp`, and `vmm.cpp`. Two friend declarations allow constructing small
fixtures without opening a model/cache. No reduction/kernel/finish implementation
is copied or mocked. Function-section garbage collection removes unrelated model
entry points. Existing ggml headers are used, without downloading/building ggml.

Each process performs 480 checks: token widths 1..8 x empty/sparse/full routing x
zero-copy false/true x eager/graph x five repetitions. Every repetition changes
input values, weights, and (for sparse routing) holes; graphs are replayed with
new uploads, not stale inputs. Tests use production `prepare`, `reduce`, and
`finish`, bitwise compare against an ordered C++ `std::fma` oracle on exact dyadic
inputs, verify output bounds, and inspect captured kernel destination + D2H node
counts. A device-sum sentinel catches a missing mapped path even without graph
support. Empty metadata explicitly exercises reduction-to-zero; this does not
claim production's all-empty-group fast exit launches a reduction.

The Python runner only orchestrates native CUDA processes and verifies logs.
It covers unset / 0 / 1 / invalid `true`, plus trace on/off, requiring identical
output hashes. Each process sees just the selected physical GPU as CUDA device 0.
Unsupported graph capture is reported, not counted as a passing graph test.
Device allocations are under 1 MiB; observed incremental usage after context
initialization was 4 MiB on RTX 5060 Ti. **CUDA context overhead is additional**;
it can OOM despite this small fixture. Do not stop production from this test.

## Build on the approved isolated remote tree

```sh
cd /root/strata/Strata-nvopt-helper
/root/strata/Strata-145/.venv/bin/cmake \
  -S tests/remote_reduce -B build-parity -G Ninja \
  -DCMAKE_MAKE_PROGRAM=/root/strata/Strata-145/.venv/bin/ninja \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DGGML_ROOT=/root/strata/Strata-145/third_party/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release
nice -n 10 /root/strata/Strata-145/.venv/bin/cmake --build build-parity -j4
```

Parent controls the maintenance window. Only after it provides headroom:

```sh
cd /root/strata/Strata-nvopt-helper
flock /root/strata/nvopt123-gpu-parity.lock \
  python3 tests/remote_reduce/run.py build-parity/remote_reduce_parity \
  --devices 0,1 --output evidence/green-both
```

Expected with graph support: 12 processes, 5760 numerical/path checks,
576 captured-graph structure checks, 0 graph skips. Run does not load a model or
request the production API. Compile outside the GPU lock.

## Recorded evidence and remaining gate

Before modifying production `reduce`, the baseline executable was run on physical
GPU1 (RTX 5060 Ti). Default-off passed 480 checks / 48 graphs. Mapped-on failed
with `PATH FAIL: mapped enabled but production reduce still writes device sum
(missing new path)`. Thus RED is a real missing-path assertion, not a compile error.
The binary remains at `evidence/remote_reduce_baseline` in the isolated remote tree.

After patching, physical GPU1 passed all six runner variants: 2880 checks,
288 graph inspections, zero graph skips. All outputs hashed `cbba175a7b8e5fb3`;
trace-on emitted exactly one `enabled=1 zero_copy=1 path=mapped-host` line and
trace-off emitted none. Logs and machine-readable counts are at
`evidence/green-gpu1/`, with baseline logs in `evidence/red-*-gpu1.log`.

Physical GPU0 (RTX 5090 D v2) could not initialize a CUDA context while production
was alive: `cudaSetDevice(dev): out of memory`, with 501 MiB reported free.
Restricting visible devices / queue connections did not resolve it. Per parent
instruction, further GPU attempts stopped; production was never stopped/restarted
or requested. Both-GPU rerun and end-to-end performance remain parent-owned gates.
Two test-only review fixes (safe error-string lifetime and partial graph-skip
counting) were compiled after the GPU-stop instruction, but not runtime-retested.
Production code is unchanged since the passing GPU1 run. Baseline and candidate
builds both emit existing NVCC warning 611-D from `expert_source.hpp`.

To reproduce baseline RED on GPU0 in the parent's window, run the retained binary:

```sh
flock /root/strata/nvopt123-gpu-parity.lock env \
  CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_MODULE_LOADING=LAZY \
  STRATA_REMOTE_REDUCE_MAPPED=1 \
  /root/strata/Strata-nvopt-helper/evidence/remote_reduce_baseline 0 1
# Expected exit 1 with the missing-new-path message, not CUDA initialization OOM.
```
