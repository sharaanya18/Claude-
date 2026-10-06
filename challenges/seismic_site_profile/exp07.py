"""Blend, decode and stack on the cached OOF probabilities, judged on three
station-held-out splits so a one-split win is visible as one."""
import numpy as np, warnings
from collections import Counter
warnings.filterwarnings("ignore")
import cv, decode

Z = np.load("oof_pool.npz")
D = cv.load(); Y = D["Y"]
SPLITS = [k.split("|")[1] for k in Z.files if k.startswith("fold|")]
MODELS = sorted({k.split("|")[1] for k in Z.files if not k.startswith("fold|")})
print("splits", SPLITS, "models", MODELS)


def sc(P):
    return cv.score_oof(Y, P.argmax(axis=2))[0]


def greedy(probs, fold, rounds=20):
    """Caruana greedy selection with replacement, weights chosen WITHOUT the
    held-out fold's labels: for each fold the blend is selected on the other
    folds' OOF rows only."""
    out = np.zeros_like(next(iter(probs.values())))
    for f in sorted(set(fold)):
        va = fold == f; tr = ~va
        cur = None; cnt = Counter()
        for _ in range(rounds):
            best = None
            for n, P in probs.items():
                cand = P[tr] if cur is None else (cur * sum(cnt.values()) + P[tr]) / (sum(cnt.values()) + 1)
                s = cv.score_oof(Y[tr], cand.argmax(axis=2))[0]
                if best is None or s > best[0]:
                    best = (s, n, cand)
            cnt[best[1]] += 1; cur = best[2]
        w = {n: c / sum(cnt.values()) for n, c in cnt.items()}
        out[va] = sum(w[n] * probs[n][va] for n in w)
    return out


print("\n== single models, per split ==")
for m in MODELS:
    row = [sc(Z[f"{s}|{m}"]) for s in SPLITS]
    print(f"  {m:5s} {' '.join(f'{x:.4f}' for x in row)}  mean {np.mean(row):.4f}")

print("\n== simple average of all models ==")
for s in SPLITS:
    pass
rows = []
for s in SPLITS:
    P = np.mean([Z[f"{s}|{m}"] for m in MODELS], axis=0)
    rows.append(sc(P))
print(f"  mean-all {' '.join(f'{x:.4f}' for x in rows)}  mean {np.mean(rows):.4f}")

print("\n== greedy blend (selected per fold on the other folds only) ==")
blend = {}
rows = []
for s in SPLITS:
    probs = {m: Z[f"{s}|{m}"] for m in MODELS}
    P = greedy(probs, Z[f"fold|{s}"])
    blend[s] = P; rows.append(sc(P))
    print(f"  {s:9s} {rows[-1]:.4f}")
print(f"  mean {np.mean(rows):.4f}")

print("\n== macro-F1 decode on the blend (weights fitted on other folds only) ==")
for mode in ("prior", "uniform"):
    rows = []
    for s in SPLITS:
        fold = Z[f"fold|{s}"]; P = blend[s]
        Pred = np.zeros((len(Y), 4), dtype=int)
        for f in sorted(set(fold)):
            va = fold == f; tr = ~va
            W = decode.fit_decode(P[tr], Y[tr], mode)
            Pred[va] = decode.apply_decode(P[va], W)
        rows.append(cv.score_oof(Y, Pred)[0])
    print(f"  {mode:8s} {' '.join(f'{x:.4f}' for x in rows)}  mean {np.mean(rows):.4f}")

print("\n== structured joint-profile decode on the blend ==")
for lam in (0.0, 0.3, 0.6, 1.0):
    rows = []
    for s in SPLITS:
        fold = Z[f"fold|{s}"]; P = blend[s]
        Pred = np.zeros((len(Y), 4), dtype=int)
        for f in sorted(set(fold)):
            va = fold == f; tr = ~va
            prof, cnt = np.unique(Y[tr], axis=0, return_counts=True)
            Pred[va] = decode.joint_prior_decode(P[va], prof, lam, cnt.astype(float)).argmax(axis=2)
        rows.append(cv.score_oof(Y, Pred)[0])
    print(f"  lam={lam:<4} {' '.join(f'{x:.4f}' for x in rows)}  mean {np.mean(rows):.4f}")

print("\n== cross-cell stacking on the blend ==")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
for C in (0.1, 1.0):
    rows = []
    for s in SPLITS:
        fold = Z[f"fold|{s}"]
        S = np.concatenate([Z[f"{s}|{m}"].reshape(len(Y), -1) for m in MODELS], axis=1)
        Pred = np.zeros((len(Y), 4), dtype=int)
        for f in sorted(set(fold)):
            va = fold == f; tr = ~va
            for c in range(4):
                m2 = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
                m2.fit(S[tr], Y[tr, c])
                Pred[va, c] = m2.predict(S[va])
        rows.append(cv.score_oof(Y, Pred)[0])
    print(f"  stack C={C:<4} {' '.join(f'{x:.4f}' for x in rows)}  mean {np.mean(rows):.4f}")
