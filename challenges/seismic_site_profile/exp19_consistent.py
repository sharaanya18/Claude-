"""Re-measure the shipped configuration with float64 features, so the reported
cross-validated numbers are produced on exactly the dtype the shipped
solution.py computes (the experiment cache previously stored float32, which
shifts tree split thresholds slightly)."""
import numpy as np, warnings, time
warnings.filterwarnings("ignore")
import cv, models, decode, repsplits

D = cv.load(); Y = D["Y"]
Xa, _, _ = cv.sel(D, "ABCDEFGHIJKM")
print("feature dtype", Xa.dtype)
splits = {"supplied": D["fold"], "rep0": repsplits.make(1)[0],
          "rep1": repsplits.make(2)[1]}
POOL = {"ordgb": models.ORDINAL_GB(0.06, 15, 200, None, 1.0),
        "ord": models.ORDINAL(0.03),
        "et": models.ET(800, 3)}
store = {}
for s, fold in splits.items():
    for m, mk in POOL.items():
        t0 = time.time()
        P = np.zeros((len(Y), 4, 4))
        for f in sorted(set(fold)):
            va = fold == f
            P[va] = mk(Xa[~va], Y[~va], Xa[va], 0)
        store[(s, m)] = P
        print(f"{s:9s} {m:6s} OOF {cv.score_oof(Y, P.argmax(axis=2))[0]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

print("\n== shipped configuration: equal-weight top3 + joint decode lam=1.0 ==")
rows, cells = [], []
for s, fold in splits.items():
    P = np.mean([store[(s, m)] for m in POOL], axis=0)
    for lam, label in ((None, "argmax"), (1.0, "lam=1.0")):
        Pred = np.zeros((len(Y), 4), dtype=int)
        for f in sorted(set(fold)):
            va = fold == f; tr = ~va
            Q = P[va]
            if lam is not None:
                prof, cnt = np.unique(Y[tr], axis=0, return_counts=True)
                Q = decode.joint_prior_decode(Q, prof, lam, cnt.astype(float))
            Pred[va] = Q.argmax(axis=2)
        sc, det = cv.score_oof(Y, Pred)
        pf = [cv.score_oof(Y[fold == f], Pred[fold == f])[0] for f in sorted(set(fold))]
        print(f"  {s:9s} {label:8s} {sc:.4f} | folds {' '.join(f'{x:.3f}' for x in pf)} "
              f"sd {np.std(pf):.3f} | cells {' '.join(f'{x:.3f}' for x in det['cell_f1'])}")
        if lam is not None:
            rows.append(sc); cells.append(det["cell_f1"])
print(f"\n  MEAN over {len(rows)} station-held-out splits: {np.mean(rows):.4f} "
      f"+-{np.std(rows):.4f}")
print("  mean per-cell macro-F1:", np.round(np.mean(cells, axis=0), 3),
      "| cell_score", round(float(np.mean(cells)), 4))
