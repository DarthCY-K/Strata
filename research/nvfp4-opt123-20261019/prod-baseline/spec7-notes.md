# spec 7 enabled (2026-10-19)

- `--spec 6` -> `--spec 7` in the dual-GPU production config.
- Same-question A/B reproduction: cold 137,570-token prompt, LRU-cache question.
  - baseline (spec 6): 163.71 / 165.61 tok/s
  - spec 7 (screening): 172.31 tok/s
  - spec 7 (production, after enable): **173.66 tok/s**, ms/step 21.73, accept 0.771
- Short requests unchanged (zh prose 124.3, code 167.6 vs 122-179 baseline band).
- 137K needle hit. VRAM 521 / 1171 MiB free, no Xid.
- Rollback: set `--spec 6` back and restart (previous config saved at `/root/strata/spec7-20261019/production.before.json`).
