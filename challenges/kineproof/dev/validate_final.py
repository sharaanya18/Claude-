"""Phase 19: independent structural validation of submission.csv.

Deliberately re-implemented from the challenge text rather than importing
solution.py's validator, so a bug in that validator cannot hide here.

Run: python3 validate_final.py <submission.csv> <public_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sub_path = Path(sys.argv[1])
public = Path(sys.argv[2])

fails, checks = [], 0


def ck(name, cond, detail=""):
    global checks
    checks += 1
    if cond:
        print(f"  PASS  {name} {detail}")
    else:
        fails.append(name)
        print(f"  FAIL  {name} {detail}")


print(f"validating {sub_path}  ({sub_path.stat().st_size:,} bytes)")

# --- raw file checks, before pandas gets a chance to normalise anything ---
raw = sub_path.read_bytes()
ck("file is valid UTF-8", True if raw.decode("utf-8") else False)
text = raw.decode("utf-8")
lines = text.splitlines()
ck("header is exactly 'sample_id,force_x,force_y,force_z'",
   lines[0] == "sample_id,force_x,force_y,force_z", f"got {lines[0]!r}")
ck("line count == 1 header + 322 data rows", len(lines) == 323, f"got {len(lines)}")
ck("no NaN/Infinity/nan tokens anywhere in the file",
   not any(t in text for t in ["NaN", "nan", "Infinity", "inf", "-inf", "None", '""']))

# --- parsed checks ---
sub = pd.read_csv(sub_path, keep_default_na=False)
test = pd.read_csv(public / "test.csv")
sample = pd.read_csv(public / "sample_submission.csv", keep_default_na=False)

ck("columns exactly as required",
   list(sub.columns) == ["sample_id", "force_x", "force_y", "force_z"], str(list(sub.columns)))
ck("no extra columns", len(sub.columns) == 4)
ck("exactly 322 data rows", len(sub) == 322, f"got {len(sub)}")
ck("every sample_id unique (no duplicate ID)", sub.sample_id.is_unique)
ck("id set == test.csv id set (no missing, no unknown ID)",
   set(sub.sample_id) == set(test.sample_id),
   f"missing {len(set(test.sample_id)-set(sub.sample_id))}, "
   f"unknown {len(set(sub.sample_id)-set(test.sample_id))}")
ck("id order matches sample_submission.csv",
   sub.sample_id.tolist() == sample.sample_id.tolist())

bad_len, bad_json, bad_fin, bad_type = [], [], [], []
stats = {}
for col in ["force_x", "force_y", "force_z"]:
    vals = []
    for i, cell in enumerate(sub[col].tolist()):
        if not isinstance(cell, str) or cell == "":
            bad_type.append((col, i)); continue
        try:
            arr = json.loads(cell)
        except Exception:
            bad_json.append((col, i)); continue
        if not isinstance(arr, list) or len(arr) != 256:
            bad_len.append((col, i, len(arr) if isinstance(arr, list) else None)); continue
        if not all(isinstance(x, (int, float)) for x in arr):
            bad_type.append((col, i)); continue
        v = np.asarray(arr, dtype=float)
        if not np.isfinite(v).all():
            bad_fin.append((col, i)); continue
        vals.append(v)
    stats[col] = np.array(vals)

ck("every cell is a valid JSON array", not bad_json, str(bad_json[:3]))
ck("every array has exactly 256 elements", not bad_len, str(bad_len[:3]))
ck("every element is numeric", not bad_type, str(bad_type[:3]))
ck("every element is finite", not bad_fin, str(bad_fin[:3]))
ck("all three axes parsed for all 322 rows",
   all(stats[c].shape == (322, 256) for c in stats),
   str({c: stats[c].shape for c in stats}))

# --- distribution sanity against the TRAIN targets (bug catching, not tuning) --
tg = pd.read_csv(public / "train_targets.csv")
print("\n  distribution sanity (predictions vs train targets):")
for a, col in enumerate(["force_x", "force_y", "force_z"]):
    tv = np.stack([json.loads(s) for s in tg[col].tolist()])
    pv = stats[col]
    print(f"    {col}: pred mean {pv.mean():+.4f} sd {pv.std():.4f} "
          f"min {pv.min():+.3f} max {pv.max():+.3f}  |  "
          f"train mean {tv.mean():+.4f} sd {tv.std():.4f} "
          f"min {tv.min():+.3f} max {tv.max():+.3f}")
ck("predictions are not degenerate/constant",
   all(stats[c].std() > 1e-4 for c in stats))
ck("force_y predictions are centred near body weight (0.4..1.2 BW)",
   0.4 < stats["force_y"].mean() < 1.2, f"{stats['force_y'].mean():.3f}")
ck("horizontal axes are near zero-mean (|mean| < 0.1 BW)",
   abs(stats["force_x"].mean()) < 0.1 and abs(stats["force_z"].mean()) < 0.1,
   f"x {stats['force_x'].mean():+.4f} z {stats['force_z'].mean():+.4f}")
ck("no predicted axis is identically zero for a row",
   not any((np.abs(stats[c]).sum(axis=1) == 0).any() for c in stats))

print(f"\n{checks - len(fails)}/{checks} checks passed")
if fails:
    raise SystemExit("STRUCTURAL VALIDATION FAILED: " + ", ".join(fails))
print("submission.csv is structurally valid for the KineProof grader")
