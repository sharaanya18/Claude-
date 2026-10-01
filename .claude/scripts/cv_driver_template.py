#!/usr/bin/env python3
"""validate.py template: run grouped-CV experiments against the self-contained solution.py without duplicating its code.

Pattern: `solution.py` stays the single source of truth and must be import-safe (work only under `if __name__ == "__main__":`).
It exposes
    CONFIG: dict                    # every fixed-plan constant an experiment may override (epochs, model name, features on/off ...)
    run_cv(config: dict) -> dict    # returns {"oof": list-or-array aligned with train rows, "fold_scores": [floats, one per fold x repeat],
                                    #          "slices": {name: float}, "secondary": [floats per fold] (optional, e.g. log-loss)}
This driver imports it, runs named experiments (config overrides), writes `reports/experiment_log.csv`, saves OOF under `reports/oof/<exp>.npy|csv`
and prints the paired fold-difference statistics versus the current best: mean gain, standard error, fraction of folds improved.

Usage:  python3 validate.py baseline                       # runs CONFIG as is
        python3 validate.py E1 key=value key2=value2       # runs with overrides (values parsed as JSON when possible)
        python3 validate.py --best baseline E1             # compare a run with the logged baseline
Decision rule (CLAUDE.md §4A, /eris-experiment): keep if the paired gain exceeds ~1 SE or is positive on most folds and split seeds. When the
primary metric is saturated (std 0 / all folds equal, e.g. noise-free synthetic data) the SE is undefined: fall back to the secondary continuous
metric (log-loss, Brier, margin) and say so in the log notes.
"""
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLUTION = HERE / "solution.py"
LOG = HERE / "reports" / "experiment_log.csv"
OOF_DIR = HERE / "reports" / "oof"


def load_solution():
    spec = importlib.util.spec_from_file_location("solution_mod", SOLUTION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_overrides(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def paired(a, b):
    """gain of b over a (positive = b better, higher-is-better scores) with SE and fraction of folds improved."""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / max(1, n - 1))
    se = sd / math.sqrt(n)
    return mean, se, sum(1 for x in d if x > 0) / n


def read_log():
    if not LOG.exists():
        return []
    with open(LOG, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 64
    exp = args[0]
    overrides = parse_overrides(args[1:])
    mod = load_solution()
    cfg = dict(getattr(mod, "CONFIG", {}))
    cfg.update(overrides)
    t = time.time()
    res = mod.run_cv(cfg)
    wall = time.time() - t
    scores = [float(x) for x in res["fold_scores"]]
    n = len(scores)
    mean = sum(scores) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in scores) / max(1, n - 1))
    sec = [float(x) for x in res.get("secondary", [])]
    print(f"{exp}: metric mean={mean:.5f} fold-std={std:.5f} over {n} fold scores; wall={wall:.0f}s; slices={res.get('slices', {})}")
    log = read_log()
    best = None
    for row in log:
        if row.get("kept", "").lower() in ("yes", "true", "1"):
            best = row
    note = ""
    if best is not None:
        prev = json.loads(best["per_fold"])
        if len(prev) == n:
            g, se, frac = paired(prev, scores)
            print(f"paired vs {best['exp_id']}: gain={g:+.5f} SE={se:.5f} folds improved={frac:.0%}")
            if std == 0 and se == 0 and best.get("secondary"):
                ps = json.loads(best["secondary"])
                if sec and len(ps) == len(sec):
                    g2, se2, f2 = paired([-x for x in ps], [-x for x in sec])      # secondary: lower is better
                    print(f"metric saturated; secondary paired gain={g2:+.5f} SE={se2:.5f} folds improved={f2:.0%}")
                    note = f"saturated primary; secondary gain {g2:+.5f} (SE {se2:.5f})"
        else:
            print("fold count differs from the best run: not comparable (use identical splits)")
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    (OOF_DIR / f"{exp}.json").write_text(json.dumps({"oof": [float(x) if not isinstance(x, (list, tuple)) else list(x) for x in res["oof"]]}) if "oof" in res else "{}")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["exp_id", "hypothesis", "change", "cv_mean", "cv_std", "per_fold", "secondary", "nested_or_holdout", "wall_s", "kept", "notes"])
        w.writerow([exp, "", json.dumps(overrides), f"{mean:.6f}", f"{std:.6f}", json.dumps(scores), json.dumps(sec), "", f"{wall:.0f}", "", note])
    print(f"logged to {LOG}. Fill hypothesis, nested/holdout number and kept=yes/no by hand after the decision.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
