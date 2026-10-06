"""Honest check on the two selection steps.

The ensemble subset and the joint-prior strength were both chosen on the
supplied split plus one repeat. Choosing twice on the same material is exactly
how a cross-validated number drifts above what a leaderboard returns, so the
shortlisted configurations are re-measured here on TWO station-held-out splits
that no selection has touched (a different generator seed entirely).
"""
import numpy as np, warnings, time
warnings.filterwarnings("ignore")
import cv, models, decode, repsplits

D = cv.load(); Y = D["Y"]
Xa, _, _ = cv.sel(D, "ABCDEFGHIJKM")
fresh = repsplits.make(2, seed=500)          # unused by any earlier decision
POOL = {
    "ordgb": models.ORDINAL_GB(0.06, 15, 200, 0.4),
    "hgb":   models.HGB(0.06, 15, 200, ff=0.4),
    "et":    models.ET(800, 3),
    "ord":   models.ORDINAL(0.03),
    "lr":    models.LR(0.01),
    "pls":   models.PLS(6),
    "ldak":  models.LDA_KNN(30, 0.3),
}
store = {}
for i, fold in enumerate(fresh):
    for m, mk in POOL.items():
        t0 = time.time()
        P = np.zeros((len(Y), 4, 4))
        for f in sorted(set(fold)):
            va = fold == f
            P[va] = mk(Xa[~va], Y[~va], Xa[va], 0)
        store[(i, m)] = P
        print(f"fresh{i} {m:6s} OOF {cv.score_oof(Y, P.argmax(axis=2))[0]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
np.savez_compressed("oof_fresh.npz",
                    **{f"{i}|{m}": v for (i, m), v in store.items()},
                    **{f"fold|{i}": f for i, f in enumerate(fresh)})

ALL = list(POOL); TOP3 = ["ordgb", "ord", "et"]


def run(names, lam, label):
    rows = []
    for i, fold in enumerate(fresh):
        P = np.mean([store[(i, m)] for m in names], axis=0)
        Pred = np.zeros((len(Y), 4), dtype=int)
        for f in sorted(set(fold)):
            va = fold == f; tr = ~va
            Q = P[va]
            if lam is not None:
                prof, cnt = np.unique(Y[tr], axis=0, return_counts=True)
                Q = decode.joint_prior_decode(Q, prof, lam, cnt.astype(float))
            Pred[va] = Q.argmax(axis=2)
        rows.append(cv.score_oof(Y, Pred)[0])
    print(f"  {label:38s} {' '.join(f'{x:.4f}' for x in rows)} mean {np.mean(rows):.4f}",
          flush=True)


print("\n== shortlisted configurations on the fresh splits ==")
run(ALL, None, "all 7, argmax")
run(ALL, 0.5, "all 7, joint prior lam=0.5")
run(ALL, 1.0, "all 7, joint prior lam=1.0")
run(TOP3, None, "top3, argmax")
run(TOP3, 0.7, "top3, joint prior lam=0.7")
run(TOP3, 1.0, "top3, joint prior lam=1.0")
run(TOP3, 1.3, "top3, joint prior lam=1.3")
