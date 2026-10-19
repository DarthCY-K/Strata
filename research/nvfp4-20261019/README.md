# NVFP4 on a 5090 D v2 (24 GB) + RTX 5060 Ti (16 GB) helper — findings (2026-10-19)

Branch `nvfp4` = upstream fork **sergqwer/strata-nvfp4 `v0.1.44.1-nvfp4.2`** (257727d5) + this folder.
Nothing in the engine source is changed; this folder holds the configs, scripts, the CJK draft vocabulary and the
measurements behind them. Machine: DESKTOP-1AQNBHK (Linux side), Ryzen 9 9950X3D, 2×48 GB DDR5-6000, RTX 5090 D v2
24 GB @ Gen5 x8 + RTX 5060 Ti 16 GB @ Gen5 x8 (the board splits the CPU slot x8/x8 when both cards are in), Linux 6.12,
CUDA 13.1, engine built from source for sm_120 (`-DSTRATA_GGML_DIR` = our 0.1.45 tree's `third_party/llama.cpp`).

Model: `Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata` (rev `379ff0b8a39f`, 120 GiB, `sha256sum -c SHA256SUMS` clean).

## Result: decode +33 % with two changes, prefill and needles unchanged

6 arms interleaved (2 rounds each), fresh process per arm, temp 0, `reasoning_effort: medium`, 4 tasks × 3 reps
(3 Chinese + 1 English), pooled decode = Σ generated ÷ Σ generation time from the engine's own log.

| arm | decode (pooled) | zh_prose / code_zh / math_zh / code_en | draft acceptance | prefill 9K / 53K | needles |
|---|---|---|---|---|---|
| as shipped (single 5090) | 121.7 | 98 / 111 / 141 / 136 | 0.672 | 3868 / 7348 | 4/4 |
| **+ CJK draft vocabulary** | 138.1 (**+13.5 %**) | 123 / 135 / 159 / 133 | 0.769 | 3864 / 7396 | 4/4 |
| **+ CJK vocab + 5060 Ti helper cache** | **162.0 (+33.2 %)** | 124 / 166 / 187 / 166 | 0.764 | 3900 / 7302 | 4/4 |

Round-to-round spread < 3 %. Raw data: `results/nvfp4-ab2-20261019_*`.

### 1. The shipped draft vocabulary has no Chinese

`mtp/draft_vocab.bin` in the model repo = this fork's `data/draft_vocab.bin`: 58,963 ids, **27 of 55,328 Han tokens**
(the author added the whole Cyrillic script for Ukrainian, not CJK). The MTP head can never propose a Chinese token, so
Chinese answers accept 49–71 % of drafts against 79 % for English. Fix, with the fork's own tool:

```sh
python tools/draft_vocab.py --gguf <model>/qwen-nvfp4-gptq-dense.gguf \
    --base <model>/mtp/draft_vocab.bin --add cjk --out <model>/mtp-cjk/draft_vocab.bin
# then hard-link dense.bin, dense.txt, experts.bin from mtp/ into mtp-cjk/ and point --mtp at mtp-cjk
```

`draft_vocab_cjk.bin` here is that output (124,737 ids, sha256 `11c70123…10d5`). Cost: the draft head grows 153 → 324 MiB
of VRAM, the expert cache loses 65 of 5,612 slots (−1 %). English is unchanged (0.79 → 0.79).

### 2. "The NVFP4 model runs on one GPU" does not cover the helper cache

The README's limit is about the low-RAM mode and the growing K/V. `--expert-cache-device1 auto --remote-expert-opt`
with `CUDA_VISIBLE_DEVICES=0,1` starts and works: CUDA1 takes 5,726 more experts (14.75 GiB), the hit rate goes
84 % → 90 %, each request waits 250–270 ms on the helper in total. `STRATA_ADAPT_FETCH` switches itself off with a helper.
Gains are largest on code/maths, ~0 on Chinese prose (the helper's wait eats it there). No P2P is needed (this pair has
none: `nvidia-smi topo -p2p r` = NS).

### Why the 24 GB card is much slower than the README's 177 tok/s

Decode is bound by host RAM bandwidth (the CPU pool and the PCIe share read the same DDR5). An NVFP4 expert is 4.5 bpw,
so a 24 GB card caches 5,612 experts (23 %) where the author's 32 GB card caches ~8,500 (93 % hits). Fewer hits × bigger
experts ≈ 2.4× the RAM bytes per token of our IQ3_S. The fork's own 24 GB emulation gives 84–95 tok/s; we measure 122.

## Other measurements

| | fork NVFP4 (single) | official 0.1.45 engine, same files | SC117 IQ3_S (0.1.45, dual helper) |
|---|---|---|---|
| decode pooled | 116–122 | 103.5 | 209 |
| prefill 9K / 53K | 3861 / 7367 | 2136 / 3306 | 3592 / 4694 |
| 12-question exact-answer set | **12/12** | — | 11/12 |

- The official 0.1.45 engine runs these files but is −12 % decode / −45…−55 % prefill (EXPDEC/B6 do not cover type-40
  experts, no `w4a4x2` prefill; matches upstream issue #3023). NVFP4 belongs on this fork's engine.
- Merging this fork into our 0.1.45 branch (`sm120`) was tried and dropped: 37 conflicting files, and
  `expert_source.{hpp,cpp}` were rewritten on both sides (`blob_parts` / `pinned_on` / `ComplementSlice` vs the fork's
  versions) — a semantic merge, not a textual one. If anything is ported to `sm120`, take only the `w4a4x2` prompt
  units (`src/prefill/mmq_nvfp4_w4a4.cu`, `mmq_nvfp4_a44.cu`, their CMake target, `STRATA_PREFILL_NVFP4`).
- Prefill wins (+56 % vs our dual-card IQ3_S) come from Blackwell FP4 MMA in the `w4a8` prompt path.
- PCIe x8 vs x16: single-card NVFP4 sends only 0.7–3.7 % of routed experts over PCIe in decode; the x8 split costs
  decode little (~6 % on IQ3_S when the second card is pulled) and prefill a lot (~40 %).

## Gotchas found on the way

- **The uncensored switch needs engine flags.** `"uncensored": true` in the config does nothing by itself; add
  `--control-vector-scaled <fork>/data/uncensor/qwen-nvfp4-gptq-uncensored.gguf:1.0 --control-vector-layer-range 8 33
  --cvec-mode project --cvec-dir per-layer` (the start log then prints `control vector mode = project … 26 steered`).
  A request can still set `"uncensored": false`.
- The engine turns `STRATA_ROUTE_TAIL_SKIP` off for NVFP4 packs (the author measured KL 0.035 with it on).
- aria2c pre-allocates: a file looks complete by `stat` long before it is. Watch the `.aria2` control file.
- The desktop's GitHub access is flaky (`curl 16` / `443 timeout`); the source was shipped as a tarball instead.

## Files

| path | what |
|---|---|
| `configs/strata-nvfp4-gptq.single.json` | the deployed single-5090 config (paths and key replaced by placeholders) |
| `configs/strata-nvfp4-gptq.cjk-helper.json` | the recommended config: CJK draft vocabulary + 5060 Ti helper |
| `draft_vocab_cjk.bin` | the CJK-extended draft subset (int32 ids) |
| `scripts/dl_nvfp4.sh` | download from hf-mirror, one aria2c per file, then `sha256sum -c` |
| `scripts/build_nvfp4b.sh` | engine build (sm_120) |
| `scripts/deploy_nvfp4.sh` | stop the miner, set power limits, write config, start, read back |
| `scripts/mk_cjk_vocab.sh` | build `mtp-cjk/` |
| `scripts/ab2_nvfp4.py` | the 6-arm A/B above |
| `scripts/qset.py` | the 12-question exact-answer set |
| `scripts/accept_nvfp4.py`, `scripts/nvfp4_official_baseline.sh` | acceptance run; official-engine comparison |
| `results/` | summaries, progress logs, quality runs |
