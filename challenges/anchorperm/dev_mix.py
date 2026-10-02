import sys, torch, numpy as np
sys.path.insert(0, "."); import ap_adapt as AD
from dev_common import va
from dev_ridge import norm_row, acc_from
def row_xy(r):
    Qn, An = norm_row(r["Q"]), norm_row(r["A"]); return np.array([Qn[q] for q, a in r["anc"]]), np.array([An[a] for q, a in r["anc"]])
def fit_maps(rows, K, lam=100, iters=15, seed=0):
    g = np.random.default_rng(seed); xy = [row_xy(r) for r in rows]; assign = g.integers(0, K, len(rows))
    for _ in range(iters):
        Ws = []
        for k in range(K):
            idx = [i for i in range(len(rows)) if assign[i] == k]
            if len(idx) < 5: Ws.append(None); continue
            X = np.vstack([xy[i][0] for i in idx]); Y = np.vstack([xy[i][1] for i in idx]); Ws.append(np.linalg.solve(X.T @ X + lam * np.eye(32), X.T @ Y))
        new = []
        for i, (X, Y) in enumerate(xy):
            res = [np.inf if W is None else ((X @ W - Y) ** 2).sum() for W in Ws]; new.append(int(np.argmin(res)))
        new = np.array(new)
        if (new == assign).all(): break
        assign = new
    return Ws, assign
def scores_mix(rows, Ws, assign):
    out = []
    for r, a in zip(rows, assign):
        Qn, An = norm_row(r["Q"]), norm_row(r["A"]); P = Qn @ Ws[a]; out.append(-((P[:, None, :] - An[None]) ** 2).sum(-1))
    return out
from dev_common import va as VA
sets = {}
for name, ml in [("rot1.0", [AD.regime_maps(1.0, 2)]), ("3reg", [AD.regime_maps(0.7, 3), AD.regime_maps(1.0, 4), AD.regime_maps(1.5, 5)]), ("2reg+familiar", None)]:
    if ml is None:
        parts = [VA[0::3], VA[1::3], VA[2::3]]; mats = [AD.regime_maps(1.0, 6), AD.regime_maps(1.5, 7)]
        sets[name] = AD.apply_regime(parts[0], mats[0], seed=7) + AD.apply_regime(parts[1], mats[1], seed=7) + parts[2]; continue
    k = len(ml); parts = [VA[i::k] for i in range(k)]; sets[name] = [r for p, m in zip(parts, ml) for r in AD.apply_regime(p, m, seed=7)]
for n, rows in sets.items():
    res = {}
    for K in [1, 2, 3, 4]:
        best = None
        for seed in range(4):   # restarts: keep the fit with the lowest total anchor residual
            Ws, assign = fit_maps(rows, K, seed=seed)
            tot = sum(((row_xy(r)[0] @ Ws[a] - row_xy(r)[1]) ** 2).sum() for r, a in zip(rows, assign))
            if best is None or tot < best[0]: best = (tot, Ws, assign)
        res[K] = round(acc_from(scores_mix(rows, best[1], best[2]), rows), 4)
    print(n, "ridge-mixture accuracy by K:", res, flush=True)
