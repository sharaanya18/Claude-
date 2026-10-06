"""Distributional latent model + exact-bootstrap decode.

For each arm we learn the whole conditional survival curve of the latent log response,
S(u | x) = P(log R >= u | x), with a ladder of 'at least u_k' binary heads.  That is
heteroscedastic and shape-free (a participant the features call certainly-unresponsive
gets S ~ 0 everywhere above zero), and it handles the interval-censored participants
for free: an upper-bounded latent still has a known 0/1 label at every threshold.

The nine probabilities then come out of the exact triple bootstrap:
    P(fev1 gain >= c) = sum_ab pw_ab * S_f( log(1+c) + log(preB/preA) | x )
and ATS from the joint of the two arms through a Gaussian copula with one parameter.
"""
import numpy as np

from scipy.special import ndtr, ndtri


def norm_cdf(x):
    return ndtr(x)


def norm_ppf(p):
    return ndtri(np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12))


class Copula:
    """Bivariate standard normal CDF at a fixed correlation (Drezner-Wesolowsky quadrature).

    Phi2(a,b;rho) = Phi(a)Phi(b) + 1/(2pi) * int_0^rho exp(-(a^2 - 2tab + b^2)/(2(1-t^2)))/sqrt(1-t^2) dt
    Exact to ~1e-10 with 24-node Gauss-Legendre; vectorised over (a, b) arrays.
    """

    def __init__(self, rho, n_node=24):
        self.rho = float(np.clip(rho, -0.95, 0.95))
        x, w = np.polynomial.legendre.leggauss(n_node)
        self.t = 0.5 * self.rho * (x + 1.0)          # map [-1,1] -> [0, rho]
        self.w = 0.5 * self.rho * w

    def __call__(self, a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        base = norm_cdf(a) * norm_cdf(b)
        if abs(self.rho) < 1e-9:
            return base
        t = self.t[:, None]
        num = a[None, :] ** 2 - 2.0 * t * a[None, :] * b[None, :] + b[None, :] ** 2
        integ = np.exp(-num / (2.0 * (1.0 - t ** 2))) / np.sqrt(1.0 - t ** 2)
        return np.clip(base + (self.w[:, None] * integ).sum(0) / (2.0 * np.pi), 0.0, 1.0)


def survival_curve(u_grid, s_vals, ramp=0.06):
    """Monotone, extended survival curve: 1 below the grid, 0 above it."""
    s = np.minimum.accumulate(np.clip(s_vals, 1e-6, 1 - 1e-6))
    xs = np.concatenate(([u_grid[0] - ramp], u_grid, [u_grid[-1] + ramp]))
    ys = np.concatenate(([1.0], s, [0.0]))
    return xs, ys


def decode_participant(ps, uf, sf, uv, sv, copula, thr_fev1, thr_fvc, ramp=0.06):
    """Nine probabilities for one participant from its two conditional survival curves."""
    xf, yf = survival_curve(uf, sf, ramp)
    xv, yv = survival_curve(uv, sv, ramp)
    out = np.empty(9)
    for j, c in enumerate(thr_fev1):
        out[j] = float(np.dot(ps.pw, np.interp(np.log1p(c) + ps.logr_f, xf, yf)))
    for j, c in enumerate(thr_fvc):
        out[5 + j] = float(np.dot(ps.pw, np.interp(np.log1p(c) + ps.logr_v, xv, yv)))
    # ATS: 1 - P(neither arm fires); dependence through a Gaussian copula on the two latents
    Ff = 1.0 - np.interp(ps.lt_f, xf, yf)
    Fv = 1.0 - np.interp(ps.lt_v, xv, yv)
    out[8] = 1.0 - float(np.dot(ps.pw, copula(norm_ppf(Ff), norm_ppf(Fv))))
    return np.clip(out, 0.0, 1.0)
