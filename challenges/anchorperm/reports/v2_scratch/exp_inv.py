import sys, os, numpy as np
sys.path.insert(0, "/home/user/Claude-/challenges/anchorperm/reports/v2_scratch")
from sims import *
from scipy.optimize import linear_sum_assignment as lsa
def cen(X): return X - X.mean(0)
# 1) rotation-invariant first-order item statistics: norm about the row centroid, count of extreme letters
nq, na, eq, ea = [], [], [], []
for r in fit:
    Q, A = cen(r["Q"]), cen(r["A"])[r["pi"]]
    nq += list(np.linalg.norm(Q, axis=1)); na += list(np.linalg.norm(A, axis=1))
    eq += list((np.abs(r["Q"]) == 2).sum(1)); ea += list((np.abs(r["A"][r["pi"]]) == 2).sum(1))
print("corr norm-about-centroid q vs partner a", np.corrcoef(nq, na)[0, 1].round(3), " extreme-letter count", np.corrcoef(eq, ea)[0, 1].round(3))
# within-row rank correlation of norms
rs = []
for r in fit:
    Q, A = cen(r["Q"]), cen(r["A"])[r["pi"]]
    a = np.argsort(np.argsort(np.linalg.norm(Q, axis=1))); b = np.argsort(np.argsort(np.linalg.norm(A, axis=1))); rs.append(np.corrcoef(a, b)[0, 1])
print("mean within-row Spearman of norms", np.nanmean(rs).round(3))
# 2) leaderboard diagnosis: linear cross-map fitted ONLY on anchor pairs from ~900 rows (the amount a transductive solver could pool from test anchors)
rng = np.random.default_rng(0)
def acc_W(W, rows):
    acc = []
    for r in rows:
        Q, A = cen(r["Q"]), cen(r["A"]); hq, ha = r["hq"], r["ha"]
        C = (((Q[hq] @ W)[:, None, :] - A[ha][None]) ** 2).sum(-1); ri, ci = lsa(C); acc.append(np.mean([ha[c] == t for c, t in zip(ci, r["y"])]))
    return np.mean(acc)
for nrows in (300, 900, 2900):
    sub = [fit[i] for i in rng.choice(len(fit), min(nrows, len(fit)), replace=False)]
    X = np.concatenate([cen(r["Q"])[[a for a, b in r["anc"]]] for r in sub]); Y = np.concatenate([cen(r["A"])[[b for a, b in r["anc"]]] for r in sub])
    out = {}
    for lam in (10, 50, 200):
        W = np.linalg.solve(X.T @ X + lam * np.eye(32), X.T @ Y); out[lam] = round(float(acc_W(W, va[:400])), 4)
    print(f"anchor pairs from {nrows} rows ({len(X)} pairs): hidden acc", out, flush=True)
