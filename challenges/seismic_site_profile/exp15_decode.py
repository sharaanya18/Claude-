"""Settle the decode on the cached out-of-fold probabilities.

The structured decode re-weights each record's four independent per-cell
distributions by the empirical joint distribution over the profiles seen in
TRAINING, then takes each cell's marginal. The mechanism is measured, not
assumed: predicted cross-cell correlation (0.66-0.76) is LOWER than gold
(0.71-0.88), so the per-cell models under-couple the column and the joint prior
restores the coupling. It is fitted per fold on the other folds' labels only,
and at inference it is a fixed table applied to one record at a time.
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import cv, decode

Z = np.load("oof_pool.npz")
D = cv.load(); Y = D["Y"]
SPLITS = [k.split("|")[1] for k in Z.files if k.startswith("fold|")]
ALL = ["ordgb", "hgb", "ord", "et", "lr", "pls", "ldak"]
TOP3 = ["ordgb", "ord", "et"]


def blend(s, names):
    return np.mean([Z[f"{s}|{m}"] for m in names], axis=0)


def run(names, lam, bw=None, label=""):
    rows = []
    for s in SPLITS:
        fold = Z[f"fold|{s}"]; P = blend(s, names)
        Pred = np.zeros((len(Y), 4), dtype=int)
        for f in sorted(set(fold)):
            va = fold == f; tr = ~va
            Q = P[va]
            if lam is not None:
                prof, cnt = np.unique(Y[tr], axis=0, return_counts=True)
                Q = decode.joint_prior_decode(Q, prof, lam, cnt.astype(float))
            if bw is not None:
                Qtr = P[tr]
                if lam is not None:
                    prof, cnt = np.unique(Y[tr], axis=0, return_counts=True)
                    Qtr = decode.joint_prior_decode(Qtr, prof, lam, cnt.astype(float))
                W = decode.fit_decode(Qtr, Y[tr], bw)
                Pred[va] = decode.apply_decode(Q, W)
            else:
                Pred[va] = Q.argmax(axis=2)
        rows.append(cv.score_oof(Y, Pred)[0])
    print(f"  {label:44s} {' '.join(f'{x:.4f}' for x in rows)} mean {np.mean(rows):.4f}")
    return float(np.mean(rows))


print("== lambda sweep, all 7 families, equal weight ==")
best = {}
for lam in (None, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0):
    lbl = "argmax (no joint prior)" if lam is None else f"joint prior lam={lam}"
    best[lam] = run(ALL, lam, None, lbl)

print("\n== same sweep on the top-3 families ==")
for lam in (None, 1.0, 1.6, 2.0, 2.5):
    lbl = "argmax" if lam is None else f"joint prior lam={lam}"
    run(TOP3, lam, None, "top3 " + lbl)

bl = max((k for k in best if k is not None), key=lambda k: best[k])
print(f"\n== joint prior lam={bl} plus band weights on top ==")
for bw in ("prior", "uniform"):
    run(ALL, bl, bw, f"lam={bl} + {bw} band weights")
