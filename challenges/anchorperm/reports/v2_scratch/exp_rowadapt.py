import sys, os, numpy as np
sys.path.insert(0, "/home/user/Claude-/challenges/anchorperm/reports/v2_scratch")
from sims import *
from dev_common import rot
from scipy.optimize import linear_sum_assignment as lsa
def cen(X): return X - X.mean(0)
XtX = sum(cen(r["Q"]).T @ cen(r["Q"]) for r in fit); XtY = sum(cen(r["Q"]).T @ cen(r["A"])[r["pi"]] for r in fit)
W1 = np.linalg.solve(XtX + 50 * np.eye(32), XtY)
s2 = np.mean(np.concatenate([((cen(r["A"])[r["pi"]] - cen(r["Q"]) @ W1) ** 2).ravel() for r in fit[:500]]))
def acc_of(r, W):
    Q, A = cen(r["Q"]), cen(r["A"]); hq, ha = r["hq"], r["ha"]
    C = ((Q[hq] @ W)[:, None, :] - A[ha][None]) ** 2; ri, ci = lsa(C.sum(-1)); return np.mean([ha[c] == t for c, t in zip(ci, r["y"])]), C.sum(-1)
def adapt(r, lam, mode, em_iters=5, tau=1.0):
    Q, A = cen(r["Q"]), cen(r["A"]); hq, ha = r["hq"], r["ha"]
    aq = [a for a, b in r["anc"]]; ab = [b for a, b in r["anc"]]
    if mode == "oracle": return W1 + np.linalg.solve(Q.T @ Q + lam * np.eye(32), Q.T @ (A[r["pi"]] - Q @ W1))
    W = W1
    for it in range(em_iters if mode == "em" else 1):
        Xs, Ys, ws = [Q[aq]], [A[ab]], [np.ones(len(aq))]
        if mode == "em" and it > 0:
            C = (((Q[hq] @ W)[:, None, :] - A[ha][None]) ** 2).sum(-1); L = -C / (2 * s2) / tau
            for _ in range(30): L = L - np.logaddexp.reduce(L, 1, keepdims=True); L = L - np.logaddexp.reduce(L, 0, keepdims=True)
            P = np.exp(L)
            # soft pairs: question i paired with expected answer under P
            Xs.append(Q[hq]); Ys.append(P @ A[ha]); ws.append(P.max(1))
        X = np.concatenate(Xs); Y = np.concatenate(Ys); w = np.concatenate(ws)[:, None]
        W = W1 + np.linalg.solve(X.T @ (w * X) + lam * np.eye(32), X.T @ (w * (Y - X @ W1)))
    return W
sets = {"clean": va[:400], "prot0.7": amp(va[:400], 1.0, 8, 4, 0.7), "prot1.0": amp(va[:400], 1.0, 8, 4, 1.0), "coordmix0.5": coordmix(va[:400], 0.5, 5)}
for name, rows in sets.items():
    out = {"global": np.mean([acc_of(r, W1)[0] for r in rows])}
    for lam in (3, 10, 30):
        out[f"anc{lam}"] = np.mean([acc_of(r, adapt(r, lam, "anc"))[0] for r in rows])
        out[f"em{lam}"] = np.mean([acc_of(r, adapt(r, lam, "em"))[0] for r in rows])
        out[f"ORACLE{lam}"] = np.mean([acc_of(r, adapt(r, lam, "oracle"))[0] for r in rows])
    print(name, {k: round(float(v), 4) for k, v in out.items()}, flush=True)
