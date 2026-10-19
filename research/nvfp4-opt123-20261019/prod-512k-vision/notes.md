# 512K context + vision on the 5060Ti (2026-10-19)

- Context: `--max-context 524288 --rope-scaling yarn --rope-scale 2`; needle hit at 507,266 / 507,378 tokens; cold prefill 6037 / 6137 tok/s.
- KV: `--kv-resident 32768` unchanged; engine log reports K/V in 6.18 GiB pinned RAM; VmRSS 78.6 GiB; available 15.5 GiB; swap 193 MiB.
- Vision: `releases/strata-vision-nvfp4-20261019-1937` (built from the same tree, CUDA sm_120), mmproj `Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf`, config section `vision.cuda_device = 1`.
- Helper cache reduced from `auto` (5726 experts) to `4900` to free ~2.1 GiB on the 5060Ti; measured vision usage 1268 MiB, free 1171 MiB. Short-request cache hit rate 86.8 / 85.8% (was ~88-90%).
- Acceptance: `/health` images true + 524288; solid-colour images 4/4 single, 4/4 with a 20K text prefix; red/blue recheck; tool roundtrip; no kernel Xid.
- `--vram-elastic` (elastic vision) is single-GPU only and conflicts with the helper cache, so vision is resident.
- Rollback: `/root/strata/rollback-512k-vision.py` (262K, no vision, ATTN_DAILY kept); `/root/strata/rollback-to-pcieauto-only.py` drops ATTN_DAILY too.
