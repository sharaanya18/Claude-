"""Exact re-implementation of the challenge metric (from the description)."""
import numpy as np

INTERVALS = [(a, b) for a in range(4) for b in range(a, 4)]          # 10 intervals
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]           # 6 pairs


def counts(Z):
    """Z (n,4,4) -> C (n,4,10) interval concession counts."""
    cs = np.concatenate([np.zeros(Z.shape[:2] + (1,), int), np.cumsum(Z, 2)], 2)
    return np.stack([cs[:, :, b + 1] - cs[:, :, a] for a, b in INTERVALS], 2)


def comparisons(Z):
    """-> relation R (n,6,10), counts of both members Ci, Cj (n,6,10)."""
    C = counts(Z)
    Ci = np.stack([C[:, i] for i, _ in PAIRS], 1)
    Cj = np.stack([C[:, j] for _, j in PAIRS], 1)
    return np.sign(Ci - Cj), Ci, Cj


def macro_f1(tp, fp, fn):
    s = []
    for c in tp:
        d = 2 * tp[c] + fp[c] + fn[c]
        s.append(0.0 if d == 0 else 2 * tp[c] / d)
    return float(np.mean(s))


def kind_score(Zp, Zg, valid=None):
    n = len(Zg)
    valid = np.ones(n, bool) if valid is None else valid
    Rg, Cig, Cjg = comparisons(Zg)
    Rp, Cip, Cjp = comparisons(np.where(valid[:, None, None], Zp, 0))
    rec = (Rp == Rg) & (Cip == Cig) & (Cjp == Cjg) & valid[:, None, None]
    tp, fp, fn = {}, {}, {}
    for c in (-1, 0, 1):
        tp[c] = int(((Rg == c) & rec).sum())
        fn[c] = int(((Rg == c) & ~rec & valid[:, None, None]).sum())
        fp[c] = int(((Rp == c) & ~rec & valid[:, None, None]).sum())
        # invalid hearing: FN to the true class, FP to every alternative class
        inv = ~valid[:, None, None]
        fn[c] += int(((Rg == c) & inv).sum())
        fp[c] += int(((Rg != c) & inv).sum())
    f_order = macro_f1(tp, fp, fn)
    tpb, fpb, fnb = {}, {}, {}
    for c in (0, 1):
        g, p = Zg == c, Zp == c
        tpb[c] = int((g & p & valid[:, None, None]).sum())
        fpb[c] = int((p & ~g & valid[:, None, None]).sum() + ((Zg != c) & ~valid[:, None, None]).sum())
        fnb[c] = int((g & ~p & valid[:, None, None]).sum() + (g & ~valid[:, None, None]).sum())
    f_path = macro_f1(tpb, fpb, fnb)
    E = float(((Zp == Zg).all((1, 2)) & valid).mean())
    return 0.70 * f_order + 0.15 * f_path + 0.15 * E, dict(F_order=f_order, F_path=f_path, E=E)


def score(Zp, Zg, kinds):
    if np.array_equal(Zp, Zg):            # stated: a complete correct submission scores exactly 100
        return 100.0, {}
    qs, parts = [], {}
    for k in sorted(set(kinds)):
        s = np.asarray(kinds) == k
        q, p = kind_score(Zp[s], Zg[s])
        qs.append(q)
        parts[k] = dict(Q=q, **p)
    return float(np.clip(100 * np.mean(qs), 0.01, 100)), parts


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Z = (rng.random((50, 4, 4)) < 0.3).astype(int)
    k = np.array(["position"] * 25 + ["correction"] * 25)
    s, _ = score(Z, Z, k)
    assert abs(s - 100) < 1e-9, s
    z0 = np.zeros((10, 4, 4), int)                       # all-zero gold and pred: only class 0 present
    print("perfect", s, "| all-zero vs all-zero (only class 0 supported):", score(z0, z0, ["position"] * 10)[0])
