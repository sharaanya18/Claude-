#!/usr/bin/env python3
"""Run a challenge solution exactly like the platform, then scan and validate (stdlib only).

Usage: python3 local_run.py challenges/<slug>            # uses <slug>/dataset/public and <slug>/working/submission.csv
       python3 local_run.py path/to/solution.py PUBLIC_DIR [OUT_CSV]

Steps: (1) compliance_scan (abort on ERROR) (2) run `python3 solution.py <public> <out>` and time it
(3) validate_submission against sample_submission.csv (4) report runtime vs the 1 h worst-case budget.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 64
    a = Path(argv[1]).resolve()
    if a.is_dir():
        sol, public, out = a / "solution.py", a / "dataset" / "public", a / "working" / "submission.csv"
    else:
        sol = a
        public = Path(argv[2]).resolve() if len(argv) > 2 else a.parent / "dataset" / "public"
        out = Path(argv[3]).resolve() if len(argv) > 3 else a.parent / "working" / "submission.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    rc = subprocess.call([sys.executable, str(HERE / "compliance_scan.py"), str(sol)])
    if rc == 2:
        print("ABORT: fix compliance ERRORs first (a rejected solution scores nothing).")
        return 2
    t = time.time()
    rc = subprocess.call([sys.executable, str(sol), str(public), str(out)], cwd=str(sol.parent))
    dt = time.time() - t
    print(f"solution exit={rc} wall={dt:.0f}s ({dt / 60:.1f} min; budget: aim <= 50 min, hard ceiling 90 min)")
    if rc != 0:
        return rc
    sample = public / "sample_submission.csv"
    return subprocess.call([sys.executable, str(HERE / "validate_submission.py"), str(out), str(sample)])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
