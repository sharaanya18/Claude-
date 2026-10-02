import sys, time, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M, ap_adapt as AD
from scipy.optimize import linear_sum_assignment
from dev_common import va, fit
from dev_ridge import norm_row, anchor_xy, ridge_scores
torch.set_num_threads(4)
def train_map(rows, lam=100.0):
    X = []; Y = []
    for r in rows:
        Qn, An = norm_row(r["Q"]), norm_row(r["A"])
        for q, a in r["anc"]: X.append(Qn[q]); Y.append(An[a])
        for q, a in zip(r["hq"], r["y"]): X.append(Qn[q]); Y.append(An[a])
    X, Y = np.array(X), np.array(Y); return np.linalg.solve(X.T @ X + lam * np.eye(32), X.T @ Y)
W0 = train_map(fit)
def ridge_prior_scores(rows, lam, W0):
    X, Y = anchor_xy(rows); W = np.linalg.solve(X.T @ X + lam * np.eye(32), X.T @ Y + lam * W0); out = []
    for r in rows:
        Qn, An = norm_row(r["Q"]), norm_row(r["A"]); P = Qn @ W
        out.append(-((P[:, None, :] - An[None]) ** 2).sum(-1))
    return out
def acc_blend(nn_logits, ridge_s, rows, beta):
    a = []
    for l, S2, r in zip(nn_logits, ridge_s, rows):
        hq, ha = r["hq"], r["ha"]; L = l[np.ix_(hq, ha)]
        S = 0.5 * ((L - np.logaddexp.reduce(L, 1, keepdims=True)) + (L - np.logaddexp.reduce(L, 0, keepdims=True)))
        R = S2[np.ix_(hq, ha)]; R = (R - R.mean()) / (R.std() + 1e-6)
        ri, ci = linear_sum_assignment(-(S + beta * R))
        a.append(np.mean([ha[c] == t for c, t in zip(ci, r["y"])]))
    return round(float(np.mean(a)), 4)
base = M.train_model(fit, epochs=40, gain_max=1.6, log=lambda *a: None)
sets = {}
for name, ml in [("clean", None), ("rot0.7", [AD.regime_maps(0.7, 1)]), ("rot1.0", [AD.regime_maps(1.0, 2)]), ("3reg", [AD.regime_maps(0.7, 3), AD.regime_maps(1.0, 4), AD.regime_maps(1.5, 5)])]:
    if ml is None: sets[name] = va; continue
    k = len(ml); parts = [va[i::k] for i in range(k)]; sets[name] = [r for p, m in zip(parts, ml) for r in AD.apply_regime(p, m, seed=7)]
for n, r in sets.items():
    ad = AD.adapt(base, fit, r, epochs=10, lr=5e-4, freeze=["head"]); lg = M.predict_logits(ad, r).numpy(); out = {}
    from dev_ridge import ridge_scores
    for tag, rs in [("ridge", ridge_scores(r, 100)), ("prior1000", ridge_prior_scores(r, 1000, W0)), ("prior3000", ridge_prior_scores(r, 3000, W0))]:
        out[tag] = {b: acc_blend(lg, rs, r, b) for b in [0, 0.5, 1.0, 2.0, 4.0]}
    print(n, out, flush=True)
