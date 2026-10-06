"""Generative decode: predicted posterior over the latent response -> the nine probabilities.

p(event) = E_{R ~ posterior} [ P_bootstrap(event | R, pre-session blows) ]

The bootstrap part is exact (enumerated triples).  The posterior part is the model's
predictive distribution for the latent log response, represented by the residual law
of the regression, which is where all the calibration lives.
"""
import numpy as np
from itertools import combinations_with_replacement
from math import factorial

ATS_ABS, ATS_REL = 0.200, 0.12
FEV1_THR = [0.00, 0.05, 0.10, 0.15, 0.20]
FVC_THR = [0.00, 0.05, 0.10]


class PreSession:
    """Everything about one participant's pre-session that the decode needs."""

    __slots__ = ("logr_f", "logr_v", "pw", "lt_f", "lt_v", "m")

    def __init__(self, fev1, fvc):
        f = np.asarray(fev1, float); v = np.asarray(fvc, float)
        m = len(f); self.m = m
        combos = list(combinations_with_replacement(range(m), 3))
        F = np.empty(len(combos)); V = np.empty(len(combos)); Wt = np.empty(len(combos))
        for i, c in enumerate(combos):
            cnt = factorial(3)
            for k in np.bincount(c, minlength=m):
                cnt //= factorial(int(k))
            F[i] = f[list(c)].max(); V[i] = v[list(c)].max(); Wt[i] = cnt / float(m) ** 3
        K = len(combos)
        Fa = np.repeat(F, K); Fb = np.tile(F, K)
        Va = np.repeat(V, K); Vb = np.tile(V, K)
        self.pw = (Wt[:, None] * Wt[None, :]).ravel()
        self.logr_f = np.log(Fb / Fa)       # post must beat pre by this ratio
        self.logr_v = np.log(Vb / Va)
        # ATS fires on an arm iff R >= max((pre+0.2)/post_unit, 1.12*pre/post_unit)
        self.lt_f = np.log(np.maximum((Fb + ATS_ABS) / Fa, (1.0 + ATS_REL) * Fb / Fa))
        self.lt_v = np.log(np.maximum((Vb + ATS_ABS) / Va, (1.0 + ATS_REL) * Vb / Va))


class ResidualLaw:
    """Joint predictive law of the two latent residuals, as 1-D CDFs plus a 2-D CDF table."""

    def __init__(self, ef, ev, scale_f=1.0, scale_v=1.0, n_grid=401, span=5.0):
        ef = np.asarray(ef, float) * scale_f
        ev = np.asarray(ev, float) * scale_v
        self.ef_sorted = np.sort(ef); self.ev_sorted = np.sort(ev)
        self.n = len(ef)
        sf = max(ef.std(), 1e-6); sv = max(ev.std(), 1e-6)
        self.gu = np.linspace(ef.mean() - span * sf, ef.mean() + span * sf, n_grid)
        self.gv = np.linspace(ev.mean() - span * sv, ev.mean() + span * sv, n_grid)
        iu = np.clip(np.searchsorted(self.gu, ef, side='left'), 0, n_grid - 1)
        iv = np.clip(np.searchsorted(self.gv, ev, side='left'), 0, n_grid - 1)
        H = np.zeros((n_grid + 1, n_grid + 1))
        np.add.at(H, (iu + 1, iv + 1), 1.0)
        self.C2 = np.cumsum(np.cumsum(H, axis=0), axis=1) / self.n   # C2[i,j] = P(ef<gu[i-1], ev<gv[j-1])

    def cdf_f(self, q):
        """P(ef < q)."""
        return np.searchsorted(self.ef_sorted, q, side='left') / self.n

    def cdf_v(self, q):
        return np.searchsorted(self.ev_sorted, q, side='left') / self.n

    def cdf2(self, qu, qv):
        """P(ef < qu AND ev < qv) by table lookup."""
        iu = np.searchsorted(self.gu, qu, side='left')
        iv = np.searchsorted(self.gv, qv, side='left')
        return self.C2[iu, iv]


def decode_one(ps, mf, mv, law):
    """Nine probabilities for one participant given predicted latent means and the residual law."""
    out = np.empty(9)
    for j, c in enumerate(FEV1_THR):
        q = np.log1p(c) + ps.logr_f - mf
        out[j] = 1.0 - float(np.dot(ps.pw, law.cdf_f(q)))
    for j, c in enumerate(FVC_THR):
        q = np.log1p(c) + ps.logr_v - mv
        out[5 + j] = 1.0 - float(np.dot(ps.pw, law.cdf_v(q)))
    nofire = law.cdf2(ps.lt_f - mf, ps.lt_v - mv)
    out[8] = 1.0 - float(np.dot(ps.pw, nofire))
    return np.clip(out, 0.0, 1.0)


def decode_all(sessions, mf, mv, law):
    P = np.empty((len(sessions), 9))
    for i, ps in enumerate(sessions):
        P[i] = decode_one(ps, mf[i], mv[i], law)
    return P
