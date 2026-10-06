"""Generalised latent parametrisation of the hidden post-inhaler session.

The post session's blows are modelled as  post_i = pre_i + delta * pre_i**LAM, with one global
exponent LAM and one latent scalar `delta` per participant per arm:
    LAM = 1  ->  purely proportional response   (post = (1+delta) * pre)
    LAM = 0  ->  purely absolute response       (post = pre + delta litres)
The transform is increasing in pre_i, so the best of a triple maps as
    post_best = Fa + delta * Fa**LAM
and every one of the nine events becomes a single threshold on `delta`:

    FEV1 gain >= c   <=>  delta >= ((1+c)*Fb - Fa) / Fa**LAM
    ATS on an arm    <=>  delta >= max(0.2 + Fb - Fa, 1.12*Fb - Fa) / Fa**LAM

Fa is the post-side triple's best, Fb the pre-side triple's best; the pre-side triple
distribution is enumerated exactly (at most C(12,3)=220 multisets), so the whole decode is
closed form and deterministic.

Why LAM is not fixed at 1: the inverted absolute change is almost uncorrelated with baseline
FEV1 (r = +0.05) while the inverted ratio is not (r = -0.21), i.e. the response looks closer to
a fixed volume than a fixed percentage.  LAM is chosen by cross-validation inside the script.
"""
from itertools import combinations_with_replacement
from math import factorial
import numpy as np

ATS_ABS, ATS_REL = 0.200, 0.12
FEV1_THR = (0.00, 0.05, 0.10, 0.15, 0.20)
FVC_THR = (0.00, 0.05, 0.10)


class GenSession:
    """Per-participant decode object for one exponent LAM.

    Stores, for each arm, the post-side best / pre-side best / scale, all divided by Fa**LAM,
    so every event threshold is one cheap arithmetic expression away.
    """

    __slots__ = ("pw", "af", "bf", "if_", "av", "bv", "iv", "m", "lam")

    def __init__(self, fev1, fvc, lam):
        f = np.asarray(fev1, float); v = np.asarray(fvc, float)
        m = len(f); self.m = m; self.lam = float(lam)
        combos = list(combinations_with_replacement(range(m), 3))
        K = len(combos)
        F = np.empty(K); V = np.empty(K); Wt = np.empty(K)
        for i, c in enumerate(combos):
            cnt = factorial(3)
            for k in np.bincount(c, minlength=m):
                cnt //= factorial(int(k))
            F[i] = f[list(c)].max(); V[i] = v[list(c)].max(); Wt[i] = cnt / float(m) ** 3
        Fa = np.repeat(F, K); Fb = np.tile(F, K)
        Va = np.repeat(V, K); Vb = np.tile(V, K)
        self.pw = (Wt[:, None] * Wt[None, :]).ravel()
        sf = Fa ** self.lam; sv = Va ** self.lam
        self.af = Fa / sf; self.bf = Fb / sf; self.if_ = 1.0 / sf
        self.av = Va / sv; self.bv = Vb / sv; self.iv = 1.0 / sv

    # thresholds on delta ------------------------------------------------
    def qf(self, c):
        return (1.0 + c) * self.bf - self.af

    def qv(self, c):
        return (1.0 + c) * self.bv - self.av

    def tf(self):
        return np.maximum(ATS_ABS * self.if_ + self.bf - self.af, (1.0 + ATS_REL) * self.bf - self.af)

    def tv(self):
        return np.maximum(ATS_ABS * self.iv + self.bv - self.av, (1.0 + ATS_REL) * self.bv - self.av)

    # exact probabilities for a point latent ------------------------------
    def probs_point(self, df, dv):
        out = np.empty(9)
        for j, c in enumerate(FEV1_THR):
            out[j] = float(self.pw[self.qf(c) <= df].sum())
        for j, c in enumerate(FVC_THR):
            out[5 + j] = float(self.pw[self.qv(c) <= dv].sum())
        out[8] = float(self.pw[(self.tf() <= df) | (self.tv() <= dv)].sum())
        return out

    # weighted CDFs, for inverting the published targets on a grid --------
    def cdfs(self, arm):
        """[(sorted thresholds, cumulative weights)] per event of this arm."""
        thrs = [self.qf(c) for c in FEV1_THR] if arm == 'f' else [self.qv(c) for c in FVC_THR]
        out = []
        for q in thrs:
            o = np.argsort(q, kind='mergesort')
            out.append((q[o], np.cumsum(self.pw[o])))
        return out

    def grid_probs(self, arm, dgrid):
        """(len(dgrid), n_events) exceedance probabilities over a grid of latent values."""
        cs = self.cdfs(arm)
        cols = []
        for qs, cw in cs:
            idx = np.searchsorted(qs, dgrid, side='right')
            cols.append(np.where(idx > 0, cw[np.clip(idx - 1, 0, len(cw) - 1)], 0.0))
        return np.stack(cols, axis=1)


def survival_curve(grid, S, ramp):
    s = np.minimum.accumulate(np.clip(S, 1e-6, 1 - 1e-6))
    xs = np.concatenate(([grid[0] - ramp], grid, [grid[-1] + ramp]))
    ys = np.concatenate(([1.0], s, [0.0]))
    return xs, ys


def decode_participant(gs, gf, Sf, gv, Sv, copula, ramp, norm_ppf):
    """Nine probabilities from the two conditional survival curves of the latent."""
    xf, yf = survival_curve(gf, Sf, ramp)
    xv, yv = survival_curve(gv, Sv, ramp)
    out = np.empty(9)
    for j, c in enumerate(FEV1_THR):
        out[j] = float(np.dot(gs.pw, np.interp(gs.qf(c), xf, yf)))
    for j, c in enumerate(FVC_THR):
        out[5 + j] = float(np.dot(gs.pw, np.interp(gs.qv(c), xv, yv)))
    Ff = 1.0 - np.interp(gs.tf(), xf, yf)
    Fv = 1.0 - np.interp(gs.tv(), xv, yv)
    out[8] = 1.0 - float(np.dot(gs.pw, copula(norm_ppf(Ff), norm_ppf(Fv))))
    return np.clip(out, 0.0, 1.0)


def invert(gs, y9, dgrid, refine=True):
    """Fit the latent (delta_fev1, delta_fvc) to one participant's published probabilities."""
    pf = gs.grid_probs('f', dgrid)
    pv = gs.grid_probs('v', dgrid)
    ef = ((pf - y9[0:5][None, :]) ** 2).sum(1)
    ev = ((pv - y9[5:8][None, :]) ** 2).sum(1)
    okf = np.where(ef <= ef.min() + 1e-12)[0]
    okv = np.where(ev <= ev.min() + 1e-12)[0]
    df = float(dgrid[okf].mean()); dv = float(dgrid[okv].mean())
    width = (float(dgrid[okf[-1]] - dgrid[okf[0]]), float(dgrid[okv[-1]] - dgrid[okv[0]]))
    if refine:
        # small joint refinement that also matches the ats probability
        step = dgrid[1] - dgrid[0]
        cf = df + step * np.arange(-12, 13) * 2
        cv = dv + step * np.arange(-12, 13) * 2
        pfc = gs.grid_probs('f', cf); pvc = gs.grid_probs('v', cv)
        efc = ((pfc - y9[0:5][None, :]) ** 2).sum(1); evc = ((pvc - y9[5:8][None, :]) ** 2).sum(1)
        tfa, tva = gs.tf(), gs.tv()
        bestobj = None
        for a in range(len(cf)):
            firef = tfa <= cf[a]
            for b in range(len(cv)):
                pa = float(gs.pw[firef | (tva <= cv[b])].sum())
                obj = efc[a] + evc[b] + (pa - y9[8]) ** 2
                if bestobj is None or obj < bestobj:
                    bestobj, df, dv = obj, cf[a], cv[b]
    return df, dv, width
