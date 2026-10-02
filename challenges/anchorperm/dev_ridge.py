import sys, time, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M, ap_adapt as AD
from scipy.optimize import linear_sum_assignment
from dev_common import va, fit
torch.set_num_threads(4)
def norm_row(X):
    Xc = X - X.mean(0); return Xc / (Xc.std() + 1e-6)
def anchor_xy(rows):
    X = []; Y = []
    for r in rows:
        Qn, An = norm_row(r["Q"]), norm_row(r["A"])
        for q, a in r["anc"]: X.append(Qn[q]); Y.append(An[a])
    return np.array(X), np.array(Y)
def ridge_scores(rows, lam):
    X, Y = anchor_xy(rows); W = np.linalg.solve(X.T @ X + lam * np.eye(32), X.T @ Y); out = []
    for r in rows:
        Qn, An = norm_row(r["Q"]), norm_row(r["A"]); P = Qn @ W
        out.append(-((P[:, None, :] - An[None]) ** 2).sum(-1))
    return out
def acc_from(scores, rows):
    a = []
    for S, r in zip(scores, rows):
        hq, ha = r["hq"], r["ha"]; M_ = S[np.ix_(hq, ha)]; ri, ci = linear_sum_assignment(-M_)
        a.append(np.mean([ha[c] == t for c, t in zip(ci, r["y"])]))
    return float(np.mean(a))
if __name__ == "__main__":
    base = M.train_model(fit, epochs=40, gain_max=1.6, log=lambda *a: None)
    sets = {}
    for name, ml in [("clean", None), ("rot0.7", [AD.regime_maps(0.7, 1)]), ("rot1.0", [AD.regime_maps(1.0, 2)]), ("3reg", [AD.regime_maps(0.7, 3), AD.regime_maps(1.0, 4), AD.regime_maps(1.5, 5)])]:
        if ml is None: sets[name] = va; continue
        k = len(ml); parts = [va[i::k] for i in range(k)]; sets[name] = [r for p, m in zip(parts, ml) for r in AD.apply_regime(p, m, seed=7)]
    for n, r in sets.items():
        res = {"base": round(float(M.evaluate(base, r)[0].mean()), 4)}
        for lam in [30, 100, 300]: res[f"ridge{lam}"] = round(acc_from(ridge_scores(r, lam), r), 4)
        ad = AD.adapt(base, fit, r, epochs=10, lr=5e-4, freeze=["head"]); res["adaptNN"] = round(float(M.evaluate(ad, r)[0].mean()), 4)
        print(n, res, flush=True)
