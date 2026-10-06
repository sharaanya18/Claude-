"""Consolidated pipeline, shared by the experiments and by the shipped solution.

Stages
  1  per-blow spirometric indices from the raw traces          (feats.py)
  2  normative reference equations fitted on the shipped pool  (harness.pool_reference)
  3  latent response multiplier per arm, inverted from the published train targets
  4  conditional survival of the latent, ordinal pooled GBDT, averaged over model variants
  5  structural calibration of the survival curves (shift / spread), fitted on train OOF
  6  copula correlation between the two arms, from train OOF PIT values
  7  exact triple-bootstrap decode -> the nine probabilities
  8  blend with a direct nine-target GBDT, weight fitted on train OOF
  9  fragility-ordering transfer for ats, isotonic, weight fitted on train OOF
"""
import sys
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
import spiro, dist_decode as dd, decode as dec

FEV1_THR = spiro.FEV1_THR; FVC_THR = spiro.FVC_THR
RAMP = 0.25

# model variants averaged for the latent survival (diversity of assumption, fixed plan)
VARIANTS = [
    dict(learning_rate=0.04, num_leaves=15, min_data_in_leaf=340, feature_fraction=0.45,
         bagging_fraction=0.80, lambda_l2=10.0, seed=1, n=400),
    dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=600, feature_fraction=0.30,
         bagging_fraction=0.70, lambda_l2=30.0, seed=7, n=500),
    dict(learning_rate=0.06, num_leaves=7,  min_data_in_leaf=200, feature_fraction=0.60,
         bagging_fraction=0.85, lambda_l2=5.0,  seed=13, n=300),
]
BASE = dict(objective='binary', bagging_freq=1, verbose=-1, num_threads=4,
            deterministic=True, force_row_wise=True)

QP = np.array([0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.875,0.925,0.96,0.98,0.99,0.995])


def ladder(y):
    g = np.maximum.accumulate(np.quantile(y, QP))
    for i in range(1, len(g)):
        if g[i] <= g[i-1]:
            g[i] = g[i-1] + 1e-4
    return g


def _stack(X, grid):
    K = len(grid); n = len(X)
    return np.hstack([np.repeat(X, K, axis=0), np.tile(grid, n)[:, None]])


def fit_survival(Xa, ya, grid, names, variants=VARIANTS):
    Z = _stack(Xa, grid)
    lab = (np.repeat(ya, len(grid)) >= np.tile(grid, len(ya))).astype(np.float64)
    mc = [0]*Xa.shape[1] + [-1]          # survival non-increasing in the threshold level
    ms = []
    for v in variants:
        p = dict(BASE); p.update({k: v[k] for k in v if k != 'n'}); p['monotone_constraints'] = mc
        ms.append(lgb.train(p, lgb.Dataset(Z, lab, feature_name=list(names)+['level']),
                            num_boost_round=v['n']))
    return ms


def predict_survival(ms, Xa, grid):
    Z = _stack(Xa, grid)
    n = len(Xa); K = len(grid)
    acc = np.zeros((n, K))
    for m in ms:                          # average on the logit scale, then back
        p = np.clip(m.predict(Z).reshape(n, K), 1e-6, 1-1e-6)
        acc += np.log(p/(1-p))
    acc /= len(ms)
    return 1.0/(1.0+np.exp(-acc))


def calibrate_survival(S, grid, shift=0.0, spread=1.0):
    """Structural recalibration: move and stretch the latent law on its own axis.

    S'(u) = S(shift + (u - shift)/spread) evaluated back on the original ladder.
    """
    if shift == 0.0 and spread == 1.0:
        return S, grid
    g2 = shift + (grid - shift) * spread
    return S, g2


def pit(S, grid, y, ramp=RAMP):
    out = np.empty(len(y))
    for i in range(len(y)):
        xs, ys = dd.survival_curve(grid, S[i], ramp)
        out[i] = 1.0 - np.interp(y[i], xs, ys)
    return np.clip(out, 1e-4, 1-1e-4)


def decode(sessions, gf, Sf, gv, Sv, rho, ramp=RAMP):
    C = dd.Copula(rho)
    return np.array([dd.decode_participant(sessions[i], gf, Sf[i], gv, Sv[i], C,
                                           FEV1_THR, FVC_THR, ramp) for i in range(len(sessions))])


def ats_moments_all(sessions, gf, Sf, gv, Sv, rho, M=128, ramp=RAMP, seed=20261006):
    """E[t] and E[t^2] of the ATS probability under the predicted latent posterior."""
    from scipy.special import ndtr
    rng = np.random.RandomState(seed)
    z1 = rng.standard_normal(M)
    z2 = rho*z1 + np.sqrt(max(1e-9, 1-rho**2))*rng.standard_normal(M)
    qf, qv = ndtr(z1), ndtr(z2)
    Et = np.empty(len(sessions)); Et2 = np.empty(len(sessions))
    for i, ps in enumerate(sessions):
        xs, ys = dd.survival_curve(gf, Sf[i], ramp); uf = np.interp(qf, 1.0-ys, xs)
        xs, ys = dd.survival_curve(gv, Sv[i], ramp); uv = np.interp(qv, 1.0-ys, xs)
        t = ((uf[:, None] >= ps.lt_f[None, :]) | (uv[:, None] >= ps.lt_v[None, :])).astype(np.float64) @ ps.pw
        Et[i] = t.mean(); Et2[i] = (t**2).mean()
    return Et, Et2


def fit_frag_iso(g_tr, p_tr, w_tr):
    iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
    iso.fit(g_tr, np.abs(p_tr-0.5), sample_weight=w_tr)
    return iso


def apply_frag_iso(iso, p, g, lo, hi, eps=1e-6):
    m = iso.predict(g)
    gr = (np.clip(g, lo, hi)-lo)/max(hi-lo, 1e-9)
    m = np.clip(m - eps*gr, 0.0, 0.5)
    side = np.where(p >= 0.5, 1.0, -1.0)
    return np.clip(0.5 + side*m, 0.0, 1.0)


DIRECT_PAR = dict(objective='l2', learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
                  feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  verbose=-1, num_threads=4, seed=1, deterministic=True, force_row_wise=True)


def fit_direct(Xa, Y, names, nr=500):
    """Direct nine-target gradient boosting: the reference rung, kept as a blend member."""
    return [lgb.train(DIRECT_PAR, lgb.Dataset(Xa, Y[:, j], feature_name=list(names)),
                      num_boost_round=nr) for j in range(9)]


def predict_direct(ms, Xa):
    return np.clip(np.column_stack([m.predict(Xa) for m in ms]), 0.0, 1.0)
