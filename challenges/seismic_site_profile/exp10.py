"""The decision run: out-of-fold probabilities for every candidate family on
three station-held-out splits, then blend and decode choices judged on all
three. Writes oof_pool.npz for later analysis."""
import numpy as np, warnings, time, sys
from collections import Counter
warnings.filterwarnings("ignore")
import cv, models, decode, repsplits

D = cv.load(); Y = D["Y"]
F = sys.argv[1] if len(sys.argv) > 1 else "ABCDEFGHIJKLM"
Xa, _, _ = cv.sel(D, F)
splits = {"supplied": D["fold"]}
for i, f in enumerate(repsplits.make(1)):
    splits[f"rep{i}"] = f

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
print(f"features {F} -> {Xa.shape[1]} columns\n")
for sname, fold in splits.items():
    for mname, mk in POOL.items():
        t0 = time.time()
        P = np.zeros((len(Y), 4, 4))
        for f in sorted(set(fold)):
            va = fold == f
            P[va] = mk(Xa[~va], Y[~va], Xa[va], 0)
        store[f"{sname}|{mname}"] = P
        s, d = cv.score_oof(Y, P.argmax(axis=2))
        print(f"{sname:9s} {mname:6s} OOF {s:.4f} | cell "
              f"{' '.join(f'{x:.3f}' for x in d['cell_f1'])} | {time.time()-t0:5.0f}s",
              flush=True)
np.savez_compressed("oof_pool.npz", **store,
                    **{f"fold|{k}": v for k, v in splits.items()})

names = list(POOL)
print("\n== per-model mean over the three splits ==")
means = {}
for m in names:
    r = [cv.score_oof(Y, store[f"{s}|{m}"].argmax(axis=2))[0] for s in splits]
    means[m] = float(np.mean(r))
    print(f"  {m:6s} {' '.join(f'{x:.4f}' for x in r)}  mean {means[m]:.4f}")
order = sorted(names, key=lambda m: -means[m])
print("  ranking:", order)

print("\n== equal-weight averages of the top-k families ==")
best = None
for k in range(1, len(order) + 1):
    sel = order[:k]
    r = [cv.score_oof(Y, np.mean([store[f"{s}|{m}"] for m in sel], axis=0).argmax(axis=2))[0]
         for s in splits]
    print(f"  top{k} ({','.join(sel)}): {' '.join(f'{x:.4f}' for x in r)} mean {np.mean(r):.4f}")
    if best is None or np.mean(r) > best[0]:
        best = (float(np.mean(r)), sel)
print(f"  -> best equal-weight set: {best[1]} mean {best[0]:.4f}")
SEL = best[1]

print("\n== greedy blend, selected per fold on the OTHER folds only ==")
def greedy(probs, fold, rounds=15):
    out = np.zeros_like(next(iter(probs.values())))
    for f in sorted(set(fold)):
        va = fold == f; tr = ~va
        cur, cnt = None, Counter()
        for _ in range(rounds):
            bb = None
            for n, P in probs.items():
                nn = sum(cnt.values())
                cand = P[tr] if cur is None else (cur * nn + P[tr]) / (nn + 1)
                sc = cv.score_oof(Y[tr], cand.argmax(axis=2))[0]
                if bb is None or sc > bb[0]:
                    bb = (sc, n, cand)
            cnt[bb[1]] += 1; cur = bb[2]
        w = {n: c / sum(cnt.values()) for n, c in cnt.items()}
        out[va] = sum(w[n] * probs[n][va] for n in w)
    return out

gr = {}
r = []
for s in splits:
    gr[s] = greedy({m: store[f"{s}|{m}"] for m in names}, splits[s])
    r.append(cv.score_oof(Y, gr[s].argmax(axis=2))[0])
print(f"  greedy: {' '.join(f'{x:.4f}' for x in r)} mean {np.mean(r):.4f}")

BL = {s: np.mean([store[f"{s}|{m}"] for m in SEL], axis=0) for s in splits}
print("\n== decode on the equal-weight blend (weights fitted on other folds) ==")
for mode, fn in (("argmax", None), ("prior", "prior"), ("uniform", "uniform"),
                 ("f1opt", "f1")):
    r = []
    for s in splits:
        fold = splits[s]; P = BL[s]
        Pred = np.zeros((len(Y), 4), dtype=int)
        for f in sorted(set(fold)):
            va = fold == f; tr = ~va
            if fn is None:
                Pred[va] = P[va].argmax(axis=2)
            elif fn == "f1":
                Pred[va] = decode.apply_decode(P[va], decode.fit_decode_f1(P[tr], Y[tr]))
            else:
                Pred[va] = decode.apply_decode(P[va], decode.fit_decode(P[tr], Y[tr], fn))
        r.append(cv.score_oof(Y, Pred)[0])
    print(f"  {mode:8s} {' '.join(f'{x:.4f}' for x in r)} mean {np.mean(r):.4f}")

print("\n== structured joint-profile decode on the blend ==")
for lam in (0.0, 0.3, 0.6, 1.0):
    r = []
    for s in splits:
        fold = splits[s]; P = BL[s]
        Pred = np.zeros((len(Y), 4), dtype=int)
        for f in sorted(set(fold)):
            va = fold == f; tr = ~va
            prof, cnt = np.unique(Y[tr], axis=0, return_counts=True)
            Pred[va] = decode.joint_prior_decode(P[va], prof, lam, cnt.astype(float)).argmax(axis=2)
        r.append(cv.score_oof(Y, Pred)[0])
    print(f"  lam={lam:<4} {' '.join(f'{x:.4f}' for x in r)} mean {np.mean(r):.4f}")
