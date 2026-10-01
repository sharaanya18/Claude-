#!/usr/bin/env python3
"""Run solution.py twice with the exact platform command and compare the outputs (stdlib only).

Usage: python3 determinism_check.py SOLUTION.py PUBLIC_DIR [--runs 2] [--workdir DIR] [--timeout 5400]
(any cwd; default workdir is <solution dir>/working/determinism)

Prints per-run wall time, sha256 of each submission and whether they are byte-identical. Differences are
reported with the first differing line so you can find the unseeded component. Platform runs use
`python3 solution.py <public_dir> <submission_out>`; this script uses the same form, with a clean output dir.
Byte-identity can legitimately fail on GPU kernels with atomics; then compare the rows (they must agree
on every row for discrete outputs and to ~1e-6 for floats) and document the residual noise.
"""
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 64
    sol, public = Path(argv[1]).resolve(), Path(argv[2]).resolve()
    runs = int(argv[argv.index("--runs") + 1]) if "--runs" in argv else 2
    work = (Path(argv[argv.index("--workdir") + 1]) if "--workdir" in argv else sol.parent / "working" / "determinism").resolve()
    timeout = int(argv[argv.index("--timeout") + 1]) if "--timeout" in argv else 5400
    outs = []
    for r in range(runs):
        d = work / f"run{r}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        out = d / "submission.csv"
        t = time.time()
        p = subprocess.run([sys.executable, str(sol), str(public), str(out)], cwd=str(sol.parent),
                           capture_output=True, text=True, timeout=timeout)
        dt = time.time() - t
        ok = out.exists()
        digest = hashlib.sha256(out.read_bytes()).hexdigest() if ok else "-"
        print(f"run {r}: exit={p.returncode} wall={dt:.0f}s sha256={digest[:16]} out={'yes' if ok else 'MISSING'}")
        if p.returncode != 0 or not ok:
            print(p.stderr[-2000:] or "solution exited 0 but wrote no submission at the given path")
            return 1
        outs.append(out)
    base = outs[0].read_bytes()
    same = all(o.read_bytes() == base for o in outs[1:])
    if same:
        print("PASS: outputs are byte-identical")
        return 0
    a = outs[0].read_text(encoding="utf-8").splitlines()
    for o in outs[1:]:
        b = o.read_text(encoding="utf-8").splitlines()
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"DIFF at line {i + 1}:\n  run0: {x[:200]}\n  other: {y[:200]}")
                break
        else:
            print(f"DIFF: different line counts {len(a)} vs {len(b)}")
    print("FAIL: outputs differ. Seed every RNG, fix thread counts and samplers, remove clock/hardware-dependent branches.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
