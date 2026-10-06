"""Cache out-of-fold probabilities for every ensemble candidate, on the supplied
split and two further station-held-out splits. Everything downstream (blend
weights, decode, stacking) is then decided on this cache, and decisions that
only hold on one split are visible as such."""
import numpy as np, warnings, time, sys
warnings.filterwarnings("ignore")
import cv, models, repsplits

D = cv.load(); Y = D["Y"]
F = "ABCDEFGHIJKLM"
Xa, Xte_a, _ = cv.sel(D, F)
splits = {"supplied": D["fold"]}
for i, f in enumerate(repsplits.make(2)):
    splits[f"rep{i}"] = f

POOL = {
    "hgb":  models.HGB(0.06, 15, 200, ff=0.4),
    "et":   models.ET(800, 3),
    "lr":   models.LR(0.01),
    "pls":  models.PLS(6),
    "ord":  models.ORDINAL(0.03),
    "ldak": models.LDA_KNN(30, 0.3),
}
out = {}
for sname, fold in splits.items():
    for mname, mk in POOL.items():
        t0 = time.time()
        P = np.zeros((len(Y), 4, 4))
        for f in sorted(set(fold)):
            va = fold == f
            P[va] = mk(Xa[~va], Y[~va], Xa[va], 0)
        out[f"{sname}|{mname}"] = P
        s, d = cv.score_oof(Y, P.argmax(axis=2))
        print(f"{sname:9s} {mname:5s} OOF {s:.4f} | cell "
              f"{' '.join(f'{x:.3f}' for x in d['cell_f1'])} | {time.time()-t0:.0f}s",
              flush=True)
np.savez_compressed("oof_pool.npz",
                    **{k: v for k, v in out.items()},
                    **{f"fold|{k}": v for k, v in splits.items()})
print("saved oof_pool.npz")
