#!/usr/bin/env python3
"""Derive leakage-safe groups and group-aware folds from the TRAIN data only (stdlib only).

Why: a given group column is a floor, not a ceiling. Rows can leak through shared keys, exact or near-duplicate
text, sliding-window overlap, or paired views of one object. Build the transitive closure (union-find) of every
relation you can detect, then fold by whole groups. Never use test rows (use this on train.csv only).

Usage examples
  # union rows sharing any value in key columns, plus exact/near-duplicate text in text columns
  python3 make_groups.py train.csv --keys site_id,doc_id --text title,body --jaccard 0.8 --out groups.csv
  # also build 5 folds balanced by group size and (optionally) by a label column's rarest classes
  python3 make_groups.py train.csv --keys site_id --folds 5 --label label --seed 42 --out groups.csv
  # audit an existing fold column against the derived groups
  python3 make_groups.py train.csv --keys site_id --audit-fold fold --out /dev/null

Output CSV columns: row (0-based), group, fold (if --folds). Prints group-size stats; a giant group (>10% of
rows) means folds will be unbalanced: inspect which key links everything before capping or splitting it.
"""
import argparse
import csv
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict


class UF:
    def __init__(self, n):
        self.p = list(range(n))
        self.sz = [1] * n

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.sz[a] < self.sz[b]:
            a, b = b, a
        self.p[b] = a
        self.sz[a] += self.sz[b]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()


def shingles(s, k=5):
    s = norm(s)
    if len(s) <= k:
        return {s} if s else set()
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def h64(x):
    return int.from_bytes(hashlib.blake2b(x.encode("utf-8"), digest_size=8).digest(), "big")


def near_duplicate_pairs(texts, thresh, bands=16, rows=4, seed=0):
    """MinHash LSH (bands x rows) over char-5-gram shingles; verify candidate pairs with exact Jaccard."""
    rng = random.Random(seed)
    nh = bands * rows
    salts = [rng.getrandbits(64) for _ in range(nh)]
    sh = [shingles(t) for t in texts]
    buckets = defaultdict(list)
    for i, s in enumerate(sh):
        if not s:
            continue
        hs = [h64(x) for x in s]
        sig = [min((h ^ salt) * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF for h in hs) for salt in salts]
        for b in range(bands):
            key = (b, tuple(sig[b * rows:(b + 1) * rows]))
            buckets[key].append(i)
    seen = set()
    for members in buckets.values():
        if len(members) < 2 or len(members) > 200:      # skip pathological buckets (boilerplate text)
            continue
        for a_i in range(len(members)):
            for b_i in range(a_i + 1, len(members)):
                a, b = members[a_i], members[b_i]
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                inter = len(sh[a] & sh[b])
                union = len(sh[a] | sh[b])
                if union and inter / union >= thresh:
                    yield a, b


def build_groups(rows, keys, text_cols, jaccard):
    n = len(rows)
    uf = UF(n)
    for k in keys:
        first = {}
        for i, r in enumerate(rows):
            v = r.get(k, "").strip()
            if not v:
                continue
            if v in first:
                uf.union(i, first[v])
            else:
                first[v] = i
    for c in text_cols:
        texts = [r.get(c, "") for r in rows]
        first = {}
        for i, t in enumerate(texts):
            nt = norm(t)
            if not nt:
                continue
            if nt in first:
                uf.union(i, first[nt])
            else:
                first[nt] = i
        if jaccard < 1.0:
            for a, b in near_duplicate_pairs(texts, jaccard):
                uf.union(a, b)
    roots = {}
    gid = []
    for i in range(n):
        r = uf.find(i)
        roots.setdefault(r, len(roots))
        gid.append(roots[r])
    return gid


def make_folds(gid, k, labels=None, seed=42):
    """Greedy balanced assignment of whole groups to folds. Rare-label groups are placed first so every fold
    gets positives of every rare label (macro metrics need both classes in every fold)."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, g in enumerate(gid):
        groups[g].append(i)
    label_tot = Counter(labels) if labels else Counter()
    def rarity(members):
        if not labels:
            return 0
        return min(label_tot[labels[i]] for i in members)
    order = sorted(groups, key=lambda g: (rarity(groups[g]), -len(groups[g]), rng.random()))
    load = [0] * k
    lab_load = [Counter() for _ in range(k)]
    fold_of = {}
    for g in order:
        members = groups[g]
        cnt = Counter(labels[i] for i in members) if labels else Counter()
        best, best_cost = None, None
        for f in rng.sample(range(k), k):
            cost = load[f] + len(members)
            if labels:
                cost = (load[f] + len(members)) / max(1, len(gid)) * k
                for lab, c in cnt.items():
                    cost += (lab_load[f][lab] + c) / label_tot[lab] * k * 0.5
            if best_cost is None or cost < best_cost:
                best, best_cost = f, cost
        fold_of[g] = best
        load[best] += len(members)
        for lab, c in cnt.items():
            lab_load[best][lab] += c
    return [fold_of[g] for g in gid]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--keys", default="", help="comma list of columns; rows sharing a value are one group")
    ap.add_argument("--text", default="", help="comma list of text columns for exact/near duplicate links")
    ap.add_argument("--jaccard", type=float, default=0.85, help="near-duplicate char-5-gram Jaccard threshold; 1.0 = exact only")
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--label", default="", help="label column for rare-class-aware fold balancing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audit-fold", default="", help="existing fold column: report groups spanning several folds")
    ap.add_argument("--out", default="groups.csv")
    a = ap.parse_args()
    with open(a.csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    keys = [c for c in a.keys.split(",") if c]
    text = [c for c in a.text.split(",") if c]
    gid = build_groups(rows, keys, text, a.jaccard)
    sizes = Counter(gid)
    n = len(rows)
    big = max(sizes.values())
    print(f"rows={n} groups={len(sizes)} largest_group={big} ({100 * big / n:.1f}% of rows) "
          f"singletons={sum(1 for v in sizes.values() if v == 1)}")
    if big > 0.10 * n:
        print("WARN: one group holds >10% of rows; find which key/duplicate relation chains everything together. "
              "Templated text makes distinct rows look like near-duplicates: raise --jaccard (0.95-1.0) or drop that "
              "text column from --text, and decide deliberately whether the giant group is real leakage structure.")
    folds = None
    if a.folds:
        labels = [r[a.label] for r in rows] if a.label else None
        folds = make_folds(gid, a.folds, labels, a.seed)
        per = Counter(folds)
        print("fold sizes:", [per[f] for f in range(a.folds)])
        if labels:
            for lab, tot in sorted(Counter(labels).items(), key=lambda kv: kv[1])[:5]:
                print(f"  rare label {lab!r}: per-fold counts {[sum(1 for i in range(n) if folds[i] == f and labels[i] == lab) for f in range(a.folds)]} (total {tot})")
    if a.audit_fold:
        by_group = defaultdict(set)
        for i, r in enumerate(rows):
            by_group[gid[i]].add(r[a.audit_fold])
        leaks = [g for g, fs in by_group.items() if len(fs) > 1]
        print(f"AUDIT: {len(leaks)} derived groups span more than one fold in column {a.audit_fold!r} "
              f"({sum(sizes[g] for g in leaks)} rows affected)")
        if leaks:
            print("FAIL: the existing folds leak across derived groups; rebuild folds from groups.")
            return 1
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "group"] + (["fold"] if folds else []))
        for i in range(n):
            w.writerow([i, gid[i]] + ([folds[i]] if folds else []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
