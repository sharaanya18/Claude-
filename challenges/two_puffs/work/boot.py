"""Exact bootstrap machinery for the Two Puffs targets.

A draw picks 3 acceptable blows WITH REPLACEMENT from a session and takes the max FEV1
and the max FVC over those three blows (they may come from different blows, but it is the
same triple).  With m <= 10 acceptable blows there are only C(m+2,3) <= 220 distinct
multisets, so the whole bootstrap distribution of (best FEV1, best FVC) can be enumerated
EXACTLY with multinomial weights -- no Monte Carlo, fully deterministic.
"""
from itertools import combinations_with_replacement
from math import factorial
import numpy as np

ATS_ABS = 0.200   # litres
ATS_REL = 0.12    # 12 percent


def triple_distribution(fev1, fvc):
    """Exact distribution of (max FEV1, max FVC) over triples drawn with replacement.

    Returns (Fmax, Vmax, weight) arrays of length C(m+2,3)."""
    f = np.asarray(fev1, dtype=np.float64)
    v = np.asarray(fvc, dtype=np.float64)
    m = len(f)
    Fs, Vs, Ws = [], [], []
    denom = float(m) ** 3
    for combo in combinations_with_replacement(range(m), 3):
        c = np.bincount(combo, minlength=m)
        # multinomial count of ordered draws giving this multiset
        cnt = factorial(3)
        for k in c:
            cnt //= factorial(int(k))
        Fs.append(f[list(combo)].max())
        Vs.append(v[list(combo)].max())
        Ws.append(cnt / denom)
    return np.array(Fs), np.array(Vs), np.array(Ws)


class Participant:
    """Caches the per-participant bootstrap objects needed for the exact decode."""

    def __init__(self, fev1, fvc):
        self.F, self.V, self.W = triple_distribution(fev1, fvc)
        K = len(self.W)
        # pair weights for (post draw a, pre draw b)
        self.PW = (self.W[:, None] * self.W[None, :]).ravel()
        self.Fa = np.repeat(self.F, K)
        self.Fb = np.tile(self.F, K)
        self.Va = np.repeat(self.V, K)
        self.Vb = np.tile(self.V, K)
        # weighted CDF of the ratio pre/post for each arm: P(R*Fa >= s*Fb) = P(Fb/Fa <= R/s)
        self.rf_sorted, self.rf_cdf = _weighted_cdf(self.Fb / self.Fa, self.PW)
        self.rv_sorted, self.rv_cdf = _weighted_cdf(self.Vb / self.Va, self.PW)

    def p_fev1(self, R, thr):
        """P(best post FEV1 >= (1+thr) * best pre FEV1) for scalar/array R."""
        q = np.asarray(R, dtype=np.float64)[..., None] / (1.0 + np.asarray(thr))
        return _cdf_eval(self.rf_sorted, self.rf_cdf, q)

    def p_fvc(self, R, thr):
        q = np.asarray(R, dtype=np.float64)[..., None] / (1.0 + np.asarray(thr))
        return _cdf_eval(self.rv_sorted, self.rv_cdf, q)

    def p_ats(self, Rf, Rv):
        """Exact ATS probability for scalar response multipliers."""
        af = Rf * self.Fa
        av = Rv * self.Va
        fire = (((af - self.Fb) >= ATS_ABS) & (af >= (1.0 + ATS_REL) * self.Fb)) | \
               (((av - self.Vb) >= ATS_ABS) & (av >= (1.0 + ATS_REL) * self.Vb))
        return float(self.PW[fire].sum())

    def p_ats_mix(self, Rf, Rv, wts):
        """ATS probability marginalised over a weighted set of (Rf, Rv) draws."""
        Rf = np.asarray(Rf, dtype=np.float64)
        Rv = np.asarray(Rv, dtype=np.float64)
        af = Rf[:, None] * self.Fa[None, :]
        av = Rv[:, None] * self.Va[None, :]
        fire = (((af - self.Fb[None, :]) >= ATS_ABS) & (af >= (1.0 + ATS_REL) * self.Fb[None, :])) | \
               (((av - self.Vb[None, :]) >= ATS_ABS) & (av >= (1.0 + ATS_REL) * self.Vb[None, :]))
        per_draw = fire.astype(np.float64) @ self.PW
        return float(np.dot(per_draw, wts) / wts.sum())


def _weighted_cdf(x, w):
    o = np.argsort(x, kind="mergesort")
    xs = x[o]
    cw = np.cumsum(w[o])
    return xs, cw


def _cdf_eval(xs, cdf, q):
    """P(x <= q) under the weighted empirical distribution."""
    idx = np.searchsorted(xs, q, side="right")
    out = np.where(idx > 0, cdf[np.clip(idx - 1, 0, len(cdf) - 1)], 0.0)
    return out
