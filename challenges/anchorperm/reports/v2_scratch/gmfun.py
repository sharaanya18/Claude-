from scipy.optimize import linear_sum_assignment as lsa
def kern(X, kind):
    Xc = X - X.mean(0)
    if kind == "cos":
        Xn = Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-6); K = Xn @ Xn.T
    else:
        K = Xc @ Xc.T / Xc.shape[1]
    n = len(K); off = ~np.eye(n, dtype=bool)
    K = (K - K[off].mean()) / max(K[off].std(), 1e-6); np.fill_diagonal(K, 0); return K

def gm(r, kind="cos", iters=0, tau=1.0, lam=1.0):
    Kq, Ka = kern(r["Q"], kind), kern(r["A"], kind)
    hq, ha = r["hq"], r["ha"]; aq = [a for a, b in r["anc"]]; ab = [b for a, b in r["anc"]]
    S0 = Kq[np.ix_(hq, aq)] @ Ka[np.ix_(ha, ab)].T          # anchor term
    S = S0.copy()
    for _ in range(iters):
        L = S / tau
        for _ in range(30):
            L = L - np.logaddexp.reduce(L, 1, keepdims=True); L = L - np.logaddexp.reduce(L, 0, keepdims=True)
        P = np.exp(L)
        S = S0 + lam * Kq[np.ix_(hq, hq)] @ P @ Ka[np.ix_(ha, ha)].T
    ri, ci = lsa(-S)
    return np.mean([ha[c] == t for c, t in zip(ci, r["y"])])

