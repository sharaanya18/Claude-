import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Claude-/challenges/anchorperm"); import ap_model as M
from sklearn.mixture import GaussianMixture
df = pd.read_csv("/home/user/Claude-/challenges/anchorperm/dataset/public/train.csv", keep_default_na=False); rows = M.load_rows(df)
mq = np.array([r["Q"].mean(0) for r in rows]); ma = np.array([r["A"].mean(0) for r in rows])
sq = np.array([r["Q"].std(0).mean() for r in rows]); sa = np.array([r["A"].std(0).mean() for r in rows])
print("row std Q pct", np.percentile(sq, [1, 10, 50, 90, 99]).round(3), "A", np.percentile(sa, [1, 10, 50, 90, 99]).round(3))
rng = np.random.default_rng(0); perm = rng.permutation(len(rows)); tr, te = perm[:2900], perm[2900:]
for nm, X in (("Qmean", mq), ("Amean", ma), ("QAmean", np.hstack([mq, ma]))):
    out = []
    for k in (1, 2, 3, 4, 6):
        g = GaussianMixture(k, covariance_type="full", random_state=0, reg_covar=1e-3).fit(X[tr]); out.append((k, round(g.score(X[te]), 3), round(g.bic(X[tr]))))
    print(nm, out, flush=True)
# mixture of linear regressions a ~ W_k q over rows (within-row centred)
D = []
for r in rows:
    Q = r["Q"] - r["Q"].mean(0); A = r["A"][r["pi"]] - r["A"].mean(0); D.append((Q, A))
def fit_W(idx, w):
    XtX = np.zeros((32, 32)); XtY = np.zeros((32, 32))
    for i, wi in zip(idx, w):
        Q, A = D[i]; XtX += wi * Q.T @ Q; XtY += wi * Q.T @ A
    return np.linalg.solve(XtX + 50 * np.eye(32), XtY)
def ll_row(i, W, s2):
    Q, A = D[i]; R = A - Q @ W; return -0.5 * (R ** 2).sum() / s2 - 0.5 * R.size * np.log(2 * np.pi * s2)
for K in (1, 2, 3, 4):
    g = np.random.default_rng(K); resp = g.dirichlet(np.ones(K), len(tr))
    for it in range(25):
        Ws = [fit_W(tr, resp[:, k]) for k in range(K)]
        s2 = np.mean([((D[i][1] - D[i][0] @ Ws[0]) ** 2).mean() for i in tr[:500]])
        L = np.array([[ll_row(i, W, s2) for W in Ws] for i in tr]) + np.log(resp.mean(0) + 1e-9)
        resp = np.exp(L - L.max(1, keepdims=True)); resp /= resp.sum(1, keepdims=True)
    pis = resp.mean(0)
    Lte = np.array([[ll_row(i, W, s2) for W in Ws] for i in te]) + np.log(pis + 1e-9)
    m = Lte.max(1); held = np.mean(m + np.log(np.exp(Lte - m[:, None]).sum(1)))
    cors = [np.corrcoef(Ws[a].ravel(), Ws[b].ravel())[0, 1].round(2) for a in range(K) for b in range(a + 1, K)]
    print("K", K, "heldout ll/row", round(held, 2), "weights", pis.round(2), "W corr", cors, "mean max resp", resp.max(1).mean().round(3), flush=True)
