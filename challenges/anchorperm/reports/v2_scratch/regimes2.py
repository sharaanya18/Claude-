import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Claude-/challenges/anchorperm"); import ap_model as M
from scipy.optimize import linear_sum_assignment as lsa
df = pd.read_csv("/home/user/Claude-/challenges/anchorperm/dataset/public/train.csv", keep_default_na=False); rows = M.load_rows(df)
D = []
for r in rows:
    Q = r["Q"] - r["Q"].mean(0); A = r["A"] - r["A"].mean(0); D.append((Q, A, r["pi"]))
rng = np.random.default_rng(0); perm = rng.permutation(len(rows)); tr, te = perm[:2900], perm[2900:]
def fit_W(idx, w):
    XtX = np.zeros((32, 32)); XtY = np.zeros((32, 32))
    for i, wi in zip(idx, w):
        Q, A, pi = D[i]; XtX += wi * Q.T @ Q; XtY += wi * Q.T @ A[pi]
    return np.linalg.solve(XtX + 50 * np.eye(32), XtY)
def ll_row(i, W, s2, sel=None):
    Q, A, pi = D[i]; R = A[pi] - Q @ W
    if sel is not None: R = R[sel]
    return -0.5 * (R ** 2).sum() / s2
def em(K, seed, iters=30):
    g = np.random.default_rng(seed); resp = g.dirichlet(np.ones(K), len(tr))
    for it in range(iters):
        Ws = [fit_W(tr, resp[:, k]) for k in range(K)]
        L = np.array([[ll_row(i, W, s2) for W in Ws] for i in tr]) + np.log(resp.mean(0) + 1e-9)
        resp = np.exp(L - L.max(1, keepdims=True)); resp /= resp.sum(1, keepdims=True)
    return Ws, resp
W1 = fit_W(tr, np.ones(len(tr)))
s2 = np.mean([((D[i][1][D[i][2]] - D[i][0] @ W1) ** 2).mean() for i in tr])
print("s2", round(s2, 3))
K = 3
WsA, rA = em(K, 11); WsB, rB = em(K, 22)
C = np.array([[np.corrcoef(a.ravel(), b.ravel())[0, 1] for b in WsB] for a in WsA]); print("restart W corr matrix\n", C.round(2))
Ws = WsA; pis = rA.mean(0)
def match_acc(i, W):
    Q, A, pi = D[i]; r = rows[i]; hq, ha = r["hq"], r["ha"]
    P = Q[hq] @ W  # predicted answers
    cost = ((P[:, None, :] - A[ha][None]) ** 2).sum(-1)
    ri, ci = lsa(cost); return np.mean([ha[c] == t for c, t in zip(ci, r["y"])])
def post(i, sel):
    L = np.array([ll_row(i, W, s2, sel) for W in Ws]) + np.log(pis); L = np.exp(L - L.max()); return L / L.sum()
acc_g, acc_or, acc_anc, acc_mix = [], [], [], []
for i in te:
    r = rows[i]
    acc_g.append(match_acc(i, W1))
    po = post(i, None); acc_or.append(match_acc(i, Ws[int(po.argmax())]))
    aq = [a for a, b in r["anc"]]; pa = post(i, aq)
    acc_anc.append(match_acc(i, sum(p * W for p, W in zip(pa, Ws))))
print("linear global", np.mean(acc_g).round(4), "oracle regime", np.mean(acc_or).round(4), "anchor-posterior regime", np.mean(acc_anc).round(4))
# regime identifiability from role-only stats: per-regime within-row covariances
lab = rA.argmax(1)
for role in (0, 1):
    covs = []
    for k in range(K):
        X = np.concatenate([D[i][role] for i, l in zip(tr, lab) if l == k]); covs.append(np.cov(X.T))
    print("role", "QA"[role], "cov corr between regimes", [np.corrcoef(covs[a].ravel(), covs[b].ravel())[0, 1].round(3) for a in range(K) for b in range(a + 1, K)])
    # split-half noise reference
    X = np.concatenate([D[i][role] for i in tr]); h = len(X) // 2
    print("   split-half ref", np.corrcoef(np.cov(X[:h].T).ravel(), np.cov(X[h:].T).ravel())[0, 1].round(3))
