# Runtime audit: <slug>
| Item | PASS / FAIL | Evidence (line numbers, measured numbers, hashes) |
|---|---|---|
| argv contract, output parent created, no hardcoded paths | | |
| Dependencies in the allowed image; weights only from HF/timm, pinned revision | | |
| Declared hardware used (A10G default / CPU if stated); no cuda-availability branches | | |
| Fixed work plan: no clock, hardware, environment, import-fallback branches; no time/timeout arguments | | |
| Seeds (random, numpy, torch, cuda, loaders, boosters, samplers); deterministic flags; fixed threads/workers | | |
| determinism_check (two runs byte-identical) and half_rows_test | | |
| Output schema/grammar validated after reload (keep_default_na=False) | | |
| Failure handling: loud; no fallback/placeholder writes; nothing written inside except | | |
| Runtime: per-phase profile × fixed counts = <min>; limit <min>; headroom <%> | | |
VERDICT: READY / NOT READY
