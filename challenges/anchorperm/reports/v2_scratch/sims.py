"""Shift simulations on TRAIN rows only (fit = training part, va = held-out rows)."""
import sys, os, numpy as np
sys.path.insert(0, "/home/user/Claude-/challenges/anchorperm"); os.chdir("/home/user/Claude-/challenges/anchorperm")
from scipy.linalg import expm
from dev_common import va, fit

def _pcs(rows, key):
    X = np.concatenate([r[key] - r[key].mean(0) for r in rows]); ev, V = np.linalg.eigh(np.cov(X.T)); return V[:, ::-1]
PC = {k: _pcs(fit, k) for k in ("Q", "A")}

def q(X): return np.clip(np.round(X), -2, 2).astype(np.float32)

def amp(rows, g, rdim, seed, theta=0.0, roles=("Q", "A")):
    """S3: amplify the top-rdim within-row principal directions (train-fit basis) by gain g, optional partial rotation theta of each role, requantise."""
    rng = np.random.default_rng(seed); out = []
    for r in rows:
        r = dict(r)
        for k in ("Q", "A"):
            X = r[k].astype(np.float64); mu = X.mean(0); V = PC[k][:, :rdim]
            Xc = X - mu; Xc = Xc + (g - 1) * (Xc @ V) @ V.T
            if theta > 0 and k in roles:
                G = rng.normal(size=(32, 32)) / 32 ** 0.5; S = (G - G.T) / 2 ** 0.5 * theta; Xc = Xc @ expm(S); mu = mu @ expm(S)
            r[k] = q(mu + Xc)
        out.append(r)
    return out

def coordmix(rows, frac, seed):
    """S4: for a random fraction of coordinates of each role, replace the coordinate by a random signed mixture of 3 other coordinates (graded per-coordinate agreement)."""
    rng = np.random.default_rng(seed); out = []
    for r in rows:
        r = dict(r)
        for k in ("Q", "A"):
            X = r[k].astype(np.float64); Y = X.copy(); cols = rng.choice(32, int(frac * 32), replace=False)
            for c in cols:
                src = rng.choice(32, 3, replace=False); w = rng.normal(size=3); Y[:, c] = X[:, src] @ w / np.sqrt((w ** 2).sum())
            r[k] = q(Y)
        out.append(r)
    return out
