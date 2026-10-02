import numpy as np
from scipy.optimize import linear_sum_assignment

def zsim(X):
    Xc = X - X.mean(0); Xn = Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-6)
    S = Xn @ Xn.T; n = len(S); iu = ~np.eye(n, dtype=bool)
    return (S - S[iu].mean()) / max(S[iu].std(), 1e-6)

def qap_refine(r, S, lam, passes=6):
    """S: hidden-block score matrix [hq x ha]. Maximise sum U + lam * sum of similarity agreement (hidden-hidden and hidden-anchor)."""
    hq, ha = r['hq'], r['ha']; m = len(hq)
    Zq, Za = zsim(r['Q']), zsim(r['A'])
    ri, ci = linear_sum_assignment(-S); perm = list(ci)            # perm[i] = column index chosen for hidden question i
    aq = [a for a, b in r['anc']]; ab = [b for a, b in r['anc']]
    def total(p):
        t = S[np.arange(m), p].sum()
        pa = [ha[c] for c in p]
        pair = 0.0
        for i in range(m):
            for i2 in range(i + 1, m):
                pair += Zq[hq[i], hq[i2]] * Za[pa[i], pa[i2]]
            for a, b in zip(aq, ab):
                pair += Zq[hq[i], a] * Za[pa[i], b]
        return t + lam * pair
    best = total(perm)
    for _ in range(passes):
        improved = False
        for i in range(m):
            for i2 in range(i + 1, m):
                p2 = perm[:]; p2[i], p2[i2] = p2[i2], p2[i]
                v = total(p2)
                if v > best + 1e-9: perm, best, improved = p2, v, True
        if not improved: break
    return [ha[c] for c in perm]
