"""Decode layer: probabilities -> band per cell, tuned for macro-F1.

Macro-F1 gives every band equal weight, so a band the model never predicts
scores 0 and costs a quarter of that cell. With four ordered bands and a
dominant "overall stiffness" latent, the middle bands sit between their
neighbours and a plain argmax under-predicts them. The fix is a per-band
multiplicative weight that pushes the PREDICTED marginal onto a target
marginal (the training prior), found by iterative proportional fitting.

Every parameter is derived from training-fold material only and is then a
fixed vector applied to one record at a time, so inference stays row-local.
"""
import numpy as np


def fit_weights(prob, target, iters=200):
    """Per-cell band weights w (4,) s.t. argmax(w*p) has marginal ~= target.

    prob: (n,4) probabilities for one cell from TRAINING-fold material.
    """
    w = np.ones(4)
    for _ in range(iters):
        pred = (prob * w).argmax(axis=1)
        freq = np.bincount(pred, minlength=4) / len(pred)
        # multiplicative update, damped; guard bands that vanish entirely
        w = w * ((target + 1e-3) / (freq + 1e-3)) ** 0.25
        w = w / w.mean()
    return w


def fit_decode(prob, Y, mode="prior"):
    """prob: (n,4,4) training-fold probabilities, Y: (n,4) labels."""
    W = np.ones((4, 4))
    if mode == "argmax":
        return W
    for c in range(4):
        if mode == "prior":
            target = np.bincount(Y[:, c], minlength=4) / len(Y)
        elif mode == "uniform":
            target = np.full(4, 0.25)
        else:
            raise ValueError(mode)
        W[c] = fit_weights(prob[:, c, :], target)
    return W


def apply_decode(prob, W):
    return (prob * W[None, :, :]).argmax(axis=2)


def joint_prior_decode(prob, profiles, lam=1.0, counts=None):
    """Structured decode: re-weight by the empirical joint distribution over the
    profiles seen in TRAINING, then take each cell's marginal. Only 68 of 256
    combinations occur, and cells correlate 0.71-0.88, so the joint prior
    carries real information the per-cell models do not share."""
    P = np.asarray(profiles)                      # (m,4) int
    logpri = np.log((counts / counts.sum()) if counts is not None
                    else np.full(len(P), 1.0 / len(P)))
    lp = np.log(prob + 1e-9)                      # (n,4,4)
    # score each candidate profile per record
    sc = lp[:, np.arange(4)[None, :], P].sum(axis=2) + lam * logpri[None, :]
    sc = sc - sc.max(axis=1, keepdims=True)
    post = np.exp(sc); post /= post.sum(axis=1, keepdims=True)
    out = np.zeros_like(prob)
    for c in range(4):
        for k in range(4):
            out[:, c, k] = post[:, P[:, c] == k].sum(axis=1)
    return out


def fit_weights_f1(prob, y, rounds=6, grid=None):
    """Alternative to prior matching: coordinate ascent on the four band weights
    to maximise macro-F1 of this cell directly. Targets the metric instead of a
    proxy, at the cost of four fitted numbers per cell (16 in total), so it has
    to be judged with the held-out fold's labels withheld."""
    import metric
    if grid is None:
        grid = np.exp(np.linspace(-1.2, 1.2, 13))
    w = np.ones(4)

    def mf1(w):
        p = (prob * w).argmax(axis=1)
        labels = np.unique(y)
        return float(np.mean([metric.f1_per_class(y, p, l) for l in labels]))

    best = mf1(w)
    for _ in range(rounds):
        improved = False
        for k in range(4):
            base = w[k]
            for g in grid:
                w[k] = base * g
                s = mf1(w)
                if s > best + 1e-9:
                    best, base, improved = s, w[k], True
            w[k] = base
        if not improved:
            break
    return w / w.mean()


def fit_decode_f1(prob, Y):
    W = np.ones((4, 4))
    for c in range(4):
        W[c] = fit_weights_f1(prob[:, c, :], Y[:, c])
    return W
