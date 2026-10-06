"""Shared CV harness: design matrix, pool normative reference, stratified folds."""
import sys
sys.path.insert(0, '/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd

W = '/home/user/Claude-/challenges/two_puffs/work'
D = '/home/user/Claude-/challenges/two_puffs/dataset'
ETH = ['non_hispanic_white', 'non_hispanic_black', 'mexican_american', 'other_hispanic', 'other']


def pool_reference(P, cols=('fev1_max', 'fvc_max', 'pef_max', 'fef2575_max', 'ratio_med')):
    """Normative reference equations fitted on the shipped POOL (no targets involved).

    The pool was never selected for the inhaler, so it is 'less obstructed by construction':
    a sex/age/height/ethnicity regression on it gives a predicted-normal value, and the
    observed/predicted ratio measures how obstructed a participant is.
    """
    P = P.copy()
    ok = P.height.notna() & P.age.notna()
    models = {}
    Z = _ref_design(P)
    for c in cols:
        y = np.log(np.clip(P[c].astype(float).values, 1e-3, None))
        m = ok.values & np.isfinite(y)
        A = Z[m]
        coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        models[c] = coef
    return models


def _ref_design(F):
    """Design matrix for the normative equations: log height, age splines, sex, ethnicity."""
    h = np.log(np.clip(F.height.astype(float).values, 50, None))
    a = F.age.astype(float).values
    cols = [np.ones(len(F)), h, a, a ** 2 / 100.0, np.log(np.clip(a, 4, None)),
            np.clip(a - 20, 0, None), np.clip(a - 40, 0, None),
            F.male.astype(float).values, F.male.astype(float).values * h]
    eth = F['ethnicity'].astype(str).values if 'ethnicity' in F else np.array(['other'] * len(F))
    for e in ETH[1:]:
        cols.append((eth == e).astype(float))
    A = np.column_stack(cols)
    return np.nan_to_num(A, nan=0.0)


def apply_reference(X, models):
    Z = _ref_design(X)
    out = {}
    for c, coef in models.items():
        pred = Z @ coef
        obs = np.log(np.clip(X[c].astype(float).values, 1e-3, None))
        out['ref_' + c] = pred
        out['pp_' + c] = obs - pred          # log(observed / predicted-normal)
    return pd.DataFrame(out, index=X.index)


def design(X, models):
    """Numeric design matrix: engineered features + ethnicity dummies + normative ratios."""
    num = X.drop(columns=['ethnicity']).astype(float)
    eth = pd.get_dummies(X['ethnicity'].astype(str)).reindex(columns=ETH, fill_value=0).astype(float)
    eth.columns = ['eth_' + c for c in eth.columns]
    ref = apply_reference(X, models)
    out = pd.concat([num, eth, ref], axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def strata(Y, n_acc, n_bins=6):
    """Fold strata: response band (ats) x acceptable-blow count, as the brief recommends."""
    ats = Y[:, 8]
    band = np.digitize(ats, [0.02, 0.10, 0.30, 0.60, 0.90])
    nb = np.clip(n_acc, 3, 6) - 3
    return band * 4 + nb


def folds(strat, n_splits=5, seed=0):
    """Stratified participant-disjoint folds (one row per participant, so always disjoint)."""
    rng = np.random.RandomState(seed)
    fold = np.full(len(strat), -1)
    for s in np.unique(strat):
        idx = np.where(strat == s)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % n_splits
    return fold
