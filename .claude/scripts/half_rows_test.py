#!/usr/bin/env python3
"""Per-row independence test (stdlib only): predictions must not depend on which other test rows are present.

Usage: python3 half_rows_test.py SOLUTION.py PUBLIC_DIR [--tol 1e-6]

Builds a copy of PUBLIC_DIR whose test.csv and sample_submission.csv keep every second row (other files are symlinked), runs the solution on the
full public dir and on the halved one with the platform command, and compares the predictions of the kept rows by id. Numeric cells must agree within
--tol, other cells exactly. A difference means some statistic, rank, normalisation, threshold or cluster was computed across test rows (banned
whole-test calibration, CLAUDE.md §2.3.5), or the pipeline is nondeterministic (run determinism_check.py first). Data-dependent decode that is
legitimately per-bag/per-case is fine only if the test file keeps whole bags: pass --group-col COL to drop whole groups instead of alternate rows.
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def read(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def write(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 64
    sol, public = Path(argv[1]).resolve(), Path(argv[2]).resolve()
    tol = float(argv[argv.index("--tol") + 1]) if "--tol" in argv else 1e-6
    gcol = argv[argv.index("--group-col") + 1] if "--group-col" in argv else None
    tmp = Path(tempfile.mkdtemp(prefix="halfrows_"))
    half = tmp / "public"
    half.mkdir()
    for f in public.iterdir():
        if f.name not in ("test.csv", "sample_submission.csv"):
            os.symlink(f, half / f.name)
    test = read(public / "test.csv")
    hdr, rows = test[0], test[1:]
    if gcol:
        gi = hdr.index(gcol)
        groups = sorted({r[gi] for r in rows})
        keep_groups = set(groups[::2])
        keep_rows = [r for r in rows if r[gi] in keep_groups]
    else:
        keep_rows = rows[::2]
    idc = 0
    keep_ids = {r[idc] for r in keep_rows}
    write(half / "test.csv", [hdr] + keep_rows)
    smp = read(public / "sample_submission.csv")
    write(half / "sample_submission.csv", [smp[0]] + [r for r in smp[1:] if r[0] in keep_ids])
    outs = []
    for name, pub in (("full", public), ("half", half)):
        out = tmp / name / "submission.csv"
        out.parent.mkdir()
        r = subprocess.run([sys.executable, str(sol), str(pub), str(out)], cwd=str(sol.parent), capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            print(f"FAIL: run on {name} data exited {r.returncode}\n{r.stderr[-1500:]}")
            return 1
        outs.append(read(out))
    full = {r[0]: r for r in outs[0][1:]}
    bad, worst = 0, 0.0
    for r in outs[1][1:]:
        a = full.get(r[0])
        if a is None:
            bad += 1
            continue
        for x, y in zip(a, r):
            try:
                d = abs(float(x) - float(y))
                worst = max(worst, d)
                bad += d > tol
            except ValueError:
                bad += x != y
    shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        print(f"FAIL: {bad} cells differ between full-file and half-file predictions (max numeric diff {worst:.3g}). "
              "Some computation pools information across test rows (or the run is nondeterministic).")
        return 1
    print(f"PASS: kept rows ({len(keep_rows)}) predicted identically with and without the other rows (max diff {worst:.3g})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
