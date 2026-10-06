"""Feature-set choice decided on THREE station-held-out splits with the fast
model families. One split leaves an OOF standard error about the size of the
differences being compared, so a single-split ranking is not trustworthy."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import cv, models, repsplits

D = cv.load()
folds = [D["fold"]] + repsplits.make(2)
SETS = ["ABCDEFGHIJK", "ABCDEFGHIJKL", "ABCDEFGHIJKM", "ABCDEFGHIJKLM",
        "ABDEGHKLM", "ABCDEFGHKLM", "ADKLM"]
POOL = [("lr", models.LR(0.01)), ("pls", models.PLS(6)), ("ord", models.ORDINAL(0.03))]
tot = {}
for gs in SETS:
    n = int(np.isin(D["gtag"], list(gs)).sum())
    accum = []
    for mn, mk in POOL:
        r = cv.run_repeated(D, gs, mk, folds, tag=f"{gs:14s} {mn:4s} ({n}f)")
        accum.append(r["mean"])
    tot[gs] = float(np.mean(accum))
    print(f"  --> {gs:14s} mean over families {tot[gs]:.4f}\n", flush=True)
print("ranking:", sorted(tot.items(), key=lambda kv: -kv[1]))
