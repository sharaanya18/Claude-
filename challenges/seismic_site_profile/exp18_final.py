"""Re-measure the chosen pool and decode with the portable booster settings
(`max_features` omitted), then redo the honest select-on-one/score-on-other
estimate. `ord` and `et` are unaffected and reused from the exp10 cache."""
import numpy as np, warnings, time
warnings.filterwarnings("ignore")
import cv, models, decode, repsplits

D = cv.load(); Y = D["Y"]
Xa, _, _ = cv.sel(D, "ABCDEFGHIJKM")
Z = np.load("oof_pool.npz")
splits = {"supplied": D["fold"], "rep0": repsplits.make(1)[0]}
store = {}
for s, fold in splits.items():
    t0 = time.time()
    P = np.zeros((len(Y), 4, 4))
    mk = models.ORDINAL_GB(0.06, 15, 200, None, 1.0)
    for f in sorted(set(fold)):
        va = fold == f
        P[va] = mk(Xa[~va], Y[~va], Xa[va], 0)
    store[(s, "ordgb")] = P
    print(f"{s:9s} ordgb(portable) OOF {cv.score_oof(Y, P.argmax(axis=2))[0]:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    for m in ("ord", "et", "hgb", "lr", "pls", "ldak"):
        store[(s, m)] = Z[f"{s}|{m}"]
np.savez_compressed("oof_final.npz",
                    **{f"{s}|{m}": v for (s, m), v in store.items()},
                    **{f"fold|{s}": f for s, f in splits.items()})

def evaluate(s, names, lam):
    fold = splits[s]
    P = np.mean([store[(s, m)] for m in names], axis=0)
    Pred = np.zeros((len(Y), 4), dtype=int)
    for f in sorted(set(fold)):
        va = fold == f; tr = ~va
        Q = P[va]
        if lam is not None:
            prof, cnt = np.unique(Y[tr], axis=0, return_counts=True)
            Q = decode.joint_prior_decode(Q, prof, lam, cnt.astype(float))
        Pred[va] = Q.argmax(axis=2)
    return cv.score_oof(Y, Pred, )

SUBSETS = {"top3": ["ordgb", "ord", "et"], "all7": ["ordgb", "hgb", "ord", "et", "lr", "pls", "ldak"],
           "top2": ["ordgb", "ord"], "top4": ["ordgb", "ord", "et", "hgb"],
           "ordgb_only": ["ordgb"]}
LAMS = [None, 0.5, 0.7, 1.0, 1.3, 1.6]
table = {}
print("\n== grid with the portable booster ==")
print(f"  {'subset':11s} " + "  ".join(f"lam={str(l):>4}" for l in LAMS))
for k, v in SUBSETS.items():
    row = []
    for l in LAMS:
        table[(k, l)] = {s: evaluate(s, v, l)[0] for s in splits}
        row.append(np.mean(list(table[(k, l)].values())))
    print(f"  {k:11s} " + "  ".join(f"{x:8.4f}" for x in row))

print("\n== select on one split, score on the other ==")
tot = []
for s in splits:
    other = [x for x in splits if x != s]
    best = max(table, key=lambda kl: np.mean([table[kl][o] for o in other]))
    tot.append(table[best][s])
    print(f"  chose {best[0]:11s} lam={str(best[1]):>4} -> scored {tot[-1]:.4f} on {s}")
print(f"  honest mean: {np.mean(tot):.4f}")

print("\n== the shipped configuration (top3, lam=1.0) in detail ==")
for s in splits:
    sc, det = evaluate(s, SUBSETS["top3"], 1.0)
    print(f"  {s:9s} {sc:.4f} | per-cell macro-F1 "
          f"{' '.join(f'{x:.3f}' for x in det['cell_f1'])} | cell_score {det['cell_score']:.4f}")
