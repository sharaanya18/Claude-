#!/usr/bin/env python3
"""Print ONLY what you may legitimately read from the test files: schema, row count, id format, file size (stdlib only).

Usage: python3 test_schema_audit.py PUBLIC_DIR

Test feature values, text, distributions and cross-references to train are off limits for modelling decisions (CLAUDE.md §2.3.5,
Held-out Answer Ingestion). Use this helper instead of opening test.csv by hand, so a look at the test set cannot leak into design choices.
Column-level facts shown: name, whether the column also exists in train, dtype guess (int/float/str) from the header-adjacent first row is NOT shown;
only the schema comparison and the id pattern are printed.
"""
import csv
import re
import sys
from pathlib import Path


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 64
    d = Path(argv[1])
    train = d / "train.csv"
    tr_cols = None
    if train.exists():
        with open(train, newline="", encoding="utf-8") as fh:
            tr_cols = next(csv.reader(fh))
    for name in ("test.csv", "sample_submission.csv"):
        f = d / name
        if not f.exists():
            print(f"{name}: missing")
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            cols = next(rd)
            n, first_ids = 0, []
            for row in rd:
                n += 1
                if len(first_ids) < 3:
                    first_ids.append(row[0])
        pat = re.sub(r"[0-9]+", "N", re.sub(r"[A-Za-z]+", "A", first_ids[0])) if first_ids else ""
        print(f"{name}: {n} rows, {f.stat().st_size} bytes, columns={cols}")
        print(f"  id pattern (letters->A, digits->N): {pat!r}")
        if tr_cols is not None and name == "test.csv":
            print(f"  columns missing from test vs train (targets/groups): {[c for c in tr_cols if c not in cols]}")
            print(f"  columns only in test: {[c for c in cols if c not in tr_cols]}")
    others = sorted(p.name for p in d.iterdir() if p.name not in ("train.csv", "test.csv", "sample_submission.csv"))
    print(f"other entries in public dir: {others[:30]}{' ...' if len(others) > 30 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
