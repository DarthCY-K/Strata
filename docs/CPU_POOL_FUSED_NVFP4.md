# Candidate #3: NVFP4 CPU fused pool (PR #2751)

## Scope and provenance

- Base: `d7ab32b50133681843e60f1bb8a4a1e35bbb495c`.
- Upstream: https://github.com/Niko1221/Strata/pull/2751
- Inspected API patch: https://api.github.com/repos/Niko1221/Strata/pulls/2751
  (`Accept: application/vnd.github.patch`). Patch header pins
  `ae51f925726e7d70166110452c3075b95aa14311`.
- Download SHA256: `99422d0d7f5e03951cd0fa55505e9734b2a022db2f7ddcf7b91e3c1ded0ba081`.
- Only the pool and end-of-run statistics were ported. Upstream's synthetic
  `pool_layer_bench` was NOT imported: it does not initialize the NVFP4 scale tail.
- No weight/quantization precision changes, no native arithmetic changes.
- No production operations, model API calls, GPU benchmarks, or performance claims.

## Runtime contract

`STRATA_POOL_FUSED=1` before pool construction opts in. Unset, `0`, and other
values leave the existing phased scheduler enabled. One-worker and Q2-native
layers retain the phased path. `tasks` still controls the old phased scheduler;
fused mode uses its own per-expert row partitions.

- `STRATA_POOL_FUSED_PARTS=8,4` default; each integer accepted in 1..64.
- `STRATA_POOL_FUSED_LAG=1` default; clamped to 0..96 to avoid integer overflow
  and unbounded task-construction loops.
- All settings are per instance, unlike upstream's process-static caches and
  static lambda capturing the first pool's `this`.
- Readiness flags reset for every active expert before publishing the next batch,
  also protecting batch-counter wraparound from stale ready flags.
- `STRATA_POOL_FUSED_TRACE=1` prints once per pool, **after actual mode-7 work
  completes**, for example:

```
[STRATA_POOL_FUSED] completed native dispatch gu_type=40 d_type=40 workers=15 jobs=1 parts=8,4 lag=1
```

Trace unset/0 means no trace I/O, formatting, locking or getenv in dispatch;
only a false branch remains. Performance arms should leave trace unset/0.
`multi_fused` counts completed nonempty fused layer calls, not internal batches.

The original scheduler's `run_phase` waits for all workers to park. Every gate/up
part precedes that expert's down parts in the claim order. The final acq_rel
`fgu_left` decrement acquires the other parts' writes before quantization; the
release/acquire ready flag publishes the quantized intermediate to down workers.
No multi-caller use of a single pool is supported or claimed.

## Real fixture

Read-only source: `/root/strata/Strata-data/models/qwen-nvfp4-gptq/pack`.
Six complete blobs at `(layer, expert)`:
`(0,0), (0,17), (7,63), (23,127), (47,255), (47,511)`.
Each is 2,764,816 bytes, types 40/40, H=2560, FF=640, including the original
16-byte `{s_gate,s_up,s_down,0}` tail. The extractor records offsets, scales and
SHA256, reads back copied bytes, and never writes the source pack. Fixture data
is not committed.

## TDD and actual results

Baseline was compiled with the new test but without any feature changes. The
fixture first verified real outputs, then failed (exit 1) on an eligible call:

```
TRACE ASSERTION FAILED: fused path absent/wrong workers=2 jobs=1 off=-1 on=-1 want_delta=1
```

After the port, each of four Release configurations passed:

```
PASS memcmp_windows=2048 layer_calls=4096 fused_calls=1248 workers=1,2,8,15 tokens=1..8 jobs=0,1,2,7,15,96,97 host=0,1 fresh_activations=every_window
```

Configurations: defaults + forced sleep (`SPIN_US=0`), lag=0/parts=7,3,
lag=2147483647 clamped to 96/parts=64,64, and trace=0/default spin.
The first three emitted exactly six once-per-pool type-40 traces; trace=0
emitted zero. Total: **8,192 memcmp windows / 16,384 OFF+ON layer calls**.
The test also checks finite/fully written outputs and untouched inactive slots;
odd repeats use mixed token counts and every window receives fresh activations.
Job counts 96/97 cross the pool's internal batch boundary. One worker is checked
as an intentional fallback, never counted as fused.

A separately compiled ASan+UBSan build passed another **2,048 windows**, with
`detect_leaks=1:halt_on_error=1` and UBSan `halt_on_error=1:print_stacktrace=1`.
No sanitizer findings. **TSAN was not run**.

Existing regressions: `pool_tasks_test` and 20-second `pool_stress` passed.
Existing `pool_tasks_test 40 40 fixtures/layer00-expert000.bin` passed 168 bitwise
comparisons with FUSED=0 and another 168 with FUSED=1.
`pool_test` could not run: its unrelated `pack/full/experts.bin` fixture is absent
(`cannot read expert 0 of layer 0`); this is retained in `evidence/regression.log`,
not silently counted as passed. Full model/GPU/SYCL builds were not run.

## Reproduce (root shell on desktop-ext)

All paths below are isolated from production. Builds use `nice 10`, `-j4`.

```bash
cd /root/strata/Strata-nvopt-cpu2751
CMAKE=/root/strata/Strata-145/.venv/bin/cmake
NINJA=/root/strata/Strata-145/.venv/bin/ninja
GGML=/root/strata/Strata-145/third_party/llama.cpp
$CMAKE -S . -B build-cpu -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_GGML_DIR="$GGML"
nice -n 10 "$CMAKE" --build build-cpu --target \
  pool_fused_nvfp4_test pool_tasks_test pool_stress -j4
# Run extraction only for a fresh destination (existing blob files are refused).
python3 tools/extract_pool_nvfp4_fixture.py \
  /root/strata/Strata-data/models/qwen-nvfp4-gptq/pack fixtures
nice -n 10 python3 tools/test_pool_fused_nvfp4.py \
  build-cpu/pool_fused_nvfp4_test fixtures evidence/parity
```

Sanitizer build uses a separate `build-sanitize`, `RelWithDebInfo`, both
`CMAKE_C_FLAGS` and `CMAKE_CXX_FLAGS` set to
`-fsanitize=address,undefined -fno-omit-frame-pointer`, and
`CMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined`, with the same CPU-only
options above. Run:

```bash
STRATA_POOL_SPIN_US=0 STRATA_POOL_FUSED_TRACE=1 \
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  nice -n 10 build-sanitize/pool_fused_nvfp4_test fixtures/*.bin
```

Toolchain observed: GCC 14.2.0, ggml 0.24.0 commit f9d9c553. The real test ran on
DarthCY-Desktop with the existing native CPU build (not AVX2-only or other OSes).

## Evidence and handoff

- Local worktree: `C:/Users/45271/AppData/Local/hermes/cache/scratch/nvopt-cpu2751`.
- Branch: `cand-nvfp4-cpu2751` (parent may cherry-pick; no push).
- Remote tree: `/root/strata/Strata-nvopt-cpu2751`.
- Remote logs: `evidence/{red,green,build-green,regression,regression-supported,sanitize}.log`,
  `evidence/parity/*.log`, `evidence/source-sha256.json`.
- Real blob provenance: `fixtures/manifest.json`.
- Local evidence copy: `C:/Users/45271/AppData/Local/hermes/cache/scratch/cpu2751-evidence`.
- Eight source-file normalized SHA256 values matched remote tested sources to
  the local candidate before commit. Only this candidate's files were edited.
- During source transfer a generated tar initially used epoch member mtimes,
  leaving objects stale; corrected mtimes, explicitly rebuilt `pool.cpp` and
  the test, then ran all GREEN tests. No stale build result is used as evidence.

Self-review: no new credential handling or network operations in production
code; no arithmetic changes; bounded task list; release/acquire publication;
zero/single/batch-boundary handling; default disabled; `git diff --check` clean.
Independent review and integrated GPU-engine performance remain parent tasks.
