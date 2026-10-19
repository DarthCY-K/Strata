# ATTN_DAILY production rollout (2026-10-19)

- Engine: `/root/strata/releases/strata-nvfp4-attn-20261019-1924`, sha256 `b58b980839c1ff890e3cdd118ed29251ce0e154fe3fc653b55e449ca29435b72`
- Source tree: `/root/strata/Strata-nvfp4-pa` (PcieAuto fix) + upstream `731528be` cu/hpp + once-only trace; no parity target.
- Production env gain: `STRATA_ATTN_DAILY=1` only. `MERGE_V2` follows daily; `PRE75` unset resolves to `daily_attn()`.
- Rollback: `sudo -n python3 /root/strata/rollback-attn2.py`.
- Acceptance: trace `chunk=1 merge_v2=1 kv_mode=1 nq=8 cap=2051`; text/tool roundtrip; short decode 128.41/175.92/186.33/172.93 tok/s; 137592-token cold prefill + needle; no kernel Xid; free VRAM 607/309 MiB; final config has no trace variable.
- First attempt self-rolled back because the trace was checked before the first decode request; trace is emitted on the first real decode only.
- CRLF source tree: upstream diff must be converted to CRLF and applied with `patch --binary`.
