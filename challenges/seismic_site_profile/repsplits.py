"""Extra station-held-out 5-fold splits for repeated CV.

The supplied folds mirror the evaluation split and are the primary harness, but
one 5-fold split over 239 stations leaves an OOF standard error of roughly
+-0.01-0.015, which is the size of the gains being chased. Additional splits
are generated the same way (whole stations held out, stratified on the station's
own profile) so a candidate can be judged on the mean over several splits.
Train material only; nothing here touches the evaluation records.
"""
import numpy as np, pandas as pd
from pathlib import Path

P = Path("dataset/public")


def make(n_splits=3, n_folds=5, seed=0):
    tr = pd.read_csv(P/"train.csv").merge(pd.read_csv(P/"folds.csv"), on="id")
    sites = tr.drop_duplicates("site")[["site", "profile"]].reset_index(drop=True)
    out = []
    for s in range(n_splits):
        rng = np.random.default_rng(1000 + seed + s)
        # stratify stations by their profile so every fold keeps all 4 bands
        # present in every cell, exactly as the supplied folds do
        assign = {}
        k = 0                      # carried across strata so fold sizes balance
        order = list(sites.groupby("profile"))
        rng.shuffle(order)
        for _prof, grp in order:
            idx = grp.site.values.copy()
            rng.shuffle(idx)
            for site in idx:
                assign[site] = k % n_folds
                k += 1
        f = tr.site.map(assign).to_numpy()
        out.append(f)
    return out


if __name__ == "__main__":
    tr = pd.read_csv(P/"train.csv").merge(pd.read_csv(P/"folds.csv"), on="id")
    Y = np.array([[("v1","v2","v3","v4").index(b) for b in p.split("|")]
                  for p in tr.profile])
    for i, f in enumerate(make()):
        ok = all(len(set(Y[f == k, c])) == 4 for k in range(5) for c in range(4))
        spans = pd.DataFrame({"s": tr.site, "f": f}).groupby("s").f.nunique().max()
        print(f"split{i}: fold sizes {np.bincount(f)} | all 4 bands in every "
              f"fold-cell: {ok} | max folds per station: {spans}")
