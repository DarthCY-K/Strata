# PcieAuto: separate CPU pool timing from helper/peer wait

## Change

Add cumulative `ExpertDispatch::ms_pool_cpu`, measured from the existing c3 timestamp to immediately after `run_split_multi[_native]`. Keep remote/peer synchronization, the c4 timestamp, and `ms_run` unchanged. Feed all PcieAuto `window()` and `mark()` calls with the new counter. This fixes the CPU-rate measurement rather than hard-coding a PCIe fraction; it does not change model weights, quantization or compute kernels.

## Hardware validation (2026-10-19)

5090D v2 + 5060Ti, CJK draft vocabulary and helper cache, same production configuration except engine executable. Interleaved old_a / fixed_a / old_b / fixed_b, separate engine processes. CUDA 13.1 sm_120 build succeeded. Candidate:

`/root/strata/releases/strata-nvfp4-pa-20261019-1729`

SHA256: `3190ee9649225204d9795dc083fecc537383b55e4fe73566470cfc1a4970981f`

| Metric | Old | Fixed |
|---|---:|---:|
| Final CPU rate, runs A/B | 21.2 / 21.3 GB/s | 54.4 / 54.0 GB/s |
| Final automatic PCIe fraction | 0.40 / 0.40 | 0.20 / 0.20 |
| Decode pooled across 24 requests/variant | 164.079 tok/s | 162.654 tok/s |
| Mean of per-arm ms/step | 18.5495 | 18.3210 |
| Mean 9.6K prefill | 3891.7 tok/s | 3883.3 tok/s |
| Mean 53K prefill | 7305.95 tok/s | 7329.75 tok/s |
| Needle checks | 4/4 | 4/4 |

The old engine reproduces wait-contaminated CPU estimates in both runs; the candidate excludes that wait and chooses 0.20 without a fixed-fraction argument. Throughput does NOT show a clear speedup (slightly lower pooled tok/s despite lower mean ms/step). Small samples, changing draft acceptance and routing prevent a universal performance or quality claim. No single-GPU or peer-mode runtime test, long-context soak, or comprehensive quality test was performed in this window.

Raw evidence on target: `/root/strata/nvfp4-ab4-pafix-20261019/` (`results.jsonl`, `summary.json`, `progress.log`, `completion.json`); engine logs: `/root/strata/Strata-nvfp4/strata-ab4.log`.

The test restored the original production executable (SHA256 `054565d770025bbef425b88b6bf417c05256e173ae5d174021f18e12658a2bf1`), not the fixed candidate. Follow-up readback confirmed this via `/proc/<pid>/exe` and health HTTP 200. Promotion requires a separately authorized restart. No fixed `--pcie-frac` was added to production.
