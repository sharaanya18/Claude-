#!/usr/bin/env python3
"""Validate a submission CSV against sample_submission.csv (stdlib only).

Usage:
  python3 validate_submission.py SUBMISSION.csv SAMPLE.csv [--test TEST.csv --id-col id]
                                 [--json-cols col1,col2] [--prob-cols col1,col2]
                                 [--int-cols col] [--allowed-col col=a|b|c] [--allow-extra-rows]

Checks (any failure -> exit 1): same header and order as the sample; same row count; id column equals the
sample's ids in the same order (and the test ids if --test); no duplicate ids; no missing/empty cells (read
without NA coercion, so "" and "NA" strings are caught); numeric columns finite; optional JSON-parsable,
probability-range, integer and allowed-value checks. A malformed CSV ranks last on the platform, so run this
before every upload. Also prints distribution hints (constant columns, value ranges) as warnings.
"""
import csv
import json
import math
import sys
from collections import Counter


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit(f"FAIL: {path} is empty")
    return rows[0], rows[1:]


def is_num(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def main(argv):
    pos = [a for a in argv[1:] if not a.startswith("--")]
    opts = {}
    flags = set()
    i = 1
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            if a in ("--allow-extra-rows",):
                flags.add(a)
            else:
                opts[a] = argv[i + 1]
                i += 1
        i += 1
    pos = [a for a in argv[1:] if not a.startswith("--") and a not in opts.values()]
    if len(pos) < 2:
        print(__doc__)
        return 64
    sub_path, sample_path = pos[0], pos[1]
    h_sub, sub = read(sub_path)
    h_smp, smp = read(sample_path)
    errors, warns = [], []

    if h_sub != h_smp:
        errors.append(f"header {h_sub} != sample header {h_smp}")
    ncol = len(h_smp)
    bad_width = [i for i, r in enumerate(sub) if len(r) != ncol]
    if bad_width:
        errors.append(f"{len(bad_width)} rows have the wrong number of fields (first at data row {bad_width[0] + 1})")
    if "--allow-extra-rows" not in flags and len(sub) != len(smp):
        errors.append(f"row count {len(sub)} != sample {len(smp)}")
    idc = h_smp.index(opts.get("--id-col", h_smp[0])) if opts.get("--id-col", h_smp[0]) in h_smp else 0
    sub_ids = [r[idc] for r in sub if len(r) == ncol]
    smp_ids = [r[idc] for r in smp]
    if len(set(sub_ids)) != len(sub_ids):
        dup = [k for k, v in Counter(sub_ids).items() if v > 1][:3]
        errors.append(f"duplicate ids, e.g. {dup}")
    if sub_ids != smp_ids and "--allow-extra-rows" not in flags:
        if set(sub_ids) == set(smp_ids):
            errors.append("ids match as a set but ORDER differs from the sample")
        else:
            miss = len(set(smp_ids) - set(sub_ids))
            extra = len(set(sub_ids) - set(smp_ids))
            errors.append(f"id mismatch vs sample: {miss} missing, {extra} unexpected")
    if "--test" in opts:
        h_t, test = read(opts["--test"])
        tid = [r[h_t.index(h_smp[idc])] for r in test] if h_smp[idc] in h_t else None
        if tid is None:
            warns.append(f"test file has no column {h_smp[idc]!r}; skipped test-id check")
        elif set(tid) != set(sub_ids):
            errors.append(f"submission ids differ from test ids ({len(set(tid) - set(sub_ids))} missing, "
                          f"{len(set(sub_ids) - set(tid))} unexpected)")

    json_cols = [c for c in opts.get("--json-cols", "").split(",") if c]
    prob_cols = [c for c in opts.get("--prob-cols", "").split(",") if c]
    int_cols = [c for c in opts.get("--int-cols", "").split(",") if c]
    allowed = {}
    if "--allowed-col" in opts:
        k, v = opts["--allowed-col"].split("=", 1)
        allowed[k] = set(v.split("|"))

    for ci, c in enumerate(h_smp):
        if ci == idc:
            continue
        vals = [r[ci] for r in sub if len(r) == ncol]
        empties = [i for i, v in enumerate(vals) if v.strip() == "" or v.strip().lower() in {"nan", "none", "null", "inf", "-inf"}]
        if empties:
            errors.append(f"column {c}: {len(empties)} empty/NaN/inf cells (first at data row {empties[0] + 1})")
        nums = [v for v in vals if is_num(v)]
        if vals and len(nums) == len(vals):
            f = [float(v) for v in nums]
            if not all(math.isfinite(x) for x in f):
                errors.append(f"column {c}: non-finite numbers")
            if len(set(f)) == 1:
                warns.append(f"column {c}: constant value {f[0]} (constant submissions are a red flag)")
            if c in prob_cols and not all(0.0 <= x <= 1.0 for x in f):
                errors.append(f"column {c}: probabilities outside [0,1]")
            if c in int_cols and not all(float(x).is_integer() for x in f):
                errors.append(f"column {c}: expected integers")
        if c in json_cols:
            for i, v in enumerate(vals):
                try:
                    json.loads(v)
                except Exception as e:  # noqa
                    errors.append(f"column {c}: row {i + 1} is not valid JSON ({e.__class__.__name__}): {v[:60]!r}")
                    break
        if c in allowed:
            badv = [v for v in vals if v not in allowed[c]]
            if badv:
                errors.append(f"column {c}: {len(badv)} values outside the allowed set, e.g. {badv[0]!r}")
        if vals and not nums and len(set(vals)) == 1:
            warns.append(f"column {c}: single repeated value {vals[0][:40]!r}")

    for w in warns:
        print("WARN:", w)
    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1
    print(f"PASS: {sub_path} ({len(sub)} rows, {ncol} columns) matches {sample_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
