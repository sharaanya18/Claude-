"""Fragility-aware treatment of the ATS output.

FDS reads only `ats`, through fragility 4p(1-p).  The Brier-optimal submission is p = E[t|x],
but the fragility the grader compares against is E[4t(1-t)|x] = 4E[t] - 4E[t^2], which is
SMALLER than 4p(1-p) by 4*Var(t|x).  Where the posterior over the latent response is wide the
two orderings disagree, so the metric's two halves genuinely pull in different directions.

We compute both, turn the fragility estimate back into an equivalent probability on the same
side of 0.5, and blend with one cross-fitted weight.  The geometric mean decides the trade.
"""
import numpy as np
from scipy.special import ndtr
import dist_decode as dd

ATS_ABS, ATS_REL = 0.200, 0.12


def posterior_draws(rho, M=96, seed=20261006):
    """Fixed deterministic coupled uniform draws for the two latent arms."""
    rng = np.random.RandomState(seed)
    z1 = rng.standard_normal(M)
    z2 = rho * z1 + np.sqrt(max(1e-9, 1 - rho ** 2)) * rng.standard_normal(M)
    return ndtr(z1), ndtr(z2)


def invert_survival(grid, S, q, ramp=0.25):
    """u such that P(latent >= u) = 1-q, from the piecewise-linear survival curve."""
    xs, ys = dd.survival_curve(grid, S, ramp)
    # ys is decreasing in xs; F = 1-ys increasing
    F = 1.0 - ys
    return np.interp(q, F, xs)


def ats_moments(ps, gf, Sf, gv, Sv, qf, qv, ramp=0.25):
    """E[t] and E[t^2] of the ATS probability under the predicted latent posterior."""
    uf = invert_survival(gf, Sf, qf, ramp)
    uv = invert_survival(gv, Sv, qv, ramp)
    # fires on an arm iff log R >= lt
    firef = uf[:, None] >= ps.lt_f[None, :]
    firev = uv[:, None] >= ps.lt_v[None, :]
    t = (firef | firev).astype(np.float64) @ ps.pw
    return float(t.mean()), float((t ** 2).mean())


def frag_to_prob(phi, side_p):
    """Probability on the same side of 0.5 whose fragility equals phi."""
    phi = np.clip(phi, 0.0, 1.0)
    half = 0.5 * np.sqrt(np.clip(1.0 - phi, 0.0, 1.0))
    return np.where(side_p >= 0.5, 0.5 + half, 0.5 - half)
