import sys; sys.path.insert(0, '.')
import numpy as np
from sim import *

MON = [0, 1, 3, 6, 9, 10]


def per_image_grads(w, X, y):
    return np.stack([L.gradient(w, X[k:k + 1], y[k:k + 1]) for k in range(len(y))])


def request_features(d):
    """Everything below uses ONLY public per-request inputs (weights, images, labels, diagnostics, request limits)."""
    w = d["weights"]; req = d["request"]
    Xb, yb = d["x_batches"], d["batch_labels"]
    G = np.stack([L.gradient(w, Xb[j], yb[j]) for j in range(8)])                  # (8, P)
    Gi = [per_image_grads(w, Xb[j], yb[j]) for j in range(8)]                        # 8 x (4, P)
    gm = G.mean(0); gmon = L.gradient(w, d["x_monitor"], d["monitor_labels"])
    gd = np.stack([L.gradient(w, d["x_diag"][j], d["diagnostic_labels"][j]) for j in range(2)])
    nrm = np.linalg.norm(G, axis=1) + 1e-12
    P = np.stack([L.probabilities(w, Xb[j]) for j in range(8)])
    loss = np.array([L.cross_entropy(w, Xb[j], yb[j]) for j in range(8)])
    acc = np.array([(P[j].argmax(1) == yb[j]).mean() for j in range(8)])
    maxp = P.max(2).mean(1)
    fisher = np.array([np.mean(np.linalg.norm(Gi[j], axis=1) ** 2) for j in range(8)])
    B = np.stack([
        nrm, loss, acc, maxp, fisher,
        (G @ gm) / (nrm * np.linalg.norm(gm) + 1e-12),
        (G @ gmon) / (nrm * np.linalg.norm(gmon) + 1e-12), (G @ gmon) / (np.linalg.norm(gmon) + 1e-12),
        (G @ gd[0]) / (nrm * np.linalg.norm(gd[0]) + 1e-12), (G @ gd[1]) / (nrm * np.linalg.norm(gd[1]) + 1e-12),
        np.array([np.mean([np.mean(Gi[j] @ Gi[j].T) for _ in [0]]) for j in range(8)]),
    ], 1)                                                                            # (8, nb)
    Kc = (G @ G.T) / (nrm[:, None] * nrm[None])                                      # cosine Gram
    Kd = (G @ G.T) / np.mean(nrm ** 2)                                               # scaled dot Gram
    obs, p0 = d["diagnostics"], L.probabilities(w, d["x_monitor"])[MON]
    dp = obs - p0[None, None]                                                        # (2,2,6,4)
    base_ce = L.cross_entropy(w, d["x_monitor"], d["monitor_labels"])
    r = [np.log(req["divergence_limit"]), np.log(req["capability_limit"]), base_ce, req["capability_limit"] / base_ce,
         np.linalg.norm(dp[0, 0]), np.linalg.norm(dp[0, 1]), np.linalg.norm(dp[1, 0]), np.linalg.norm(dp[1, 1]),
         np.linalg.norm(dp[0] - dp[1]), np.linalg.norm(dp[:, 0] - dp[:, 1]), np.linalg.norm(gm), np.linalg.norm(gmon), np.linalg.norm(gd[0]), np.linalg.norm(gd[1]),
         float(np.mean(nrm)), float(np.std(nrm) / np.mean(nrm))]
    return dict(B=B, Kc=Kc, Kd=Kd, r=np.array(r))


def order_features(orders, rf):
    """orders: (n, 8) int array of permutations. Returns (n, F) design matrix."""
    orders = np.asarray(orders); n = len(orders); B, Kc, Kd = rf["B"], rf["Kc"], rf["Kd"]
    pos = np.arange(8)
    wts = {"first3": (pos < 3).astype(float), "last3": (pos >= 5).astype(float), "ramp": (pos - 3.5) / 3.5, "decay": 0.9 ** (7 - pos)}
    cols = []
    Bo = B[orders]                                                                   # (n, 8, nb)
    for nm, wv in wts.items(): cols.append(np.einsum("nkb,k->nb", Bo, wv))
    for K in (Kc, Kd):
        Ko = K[orders[:, :, None], orders[:, None, :]]                               # (n, 8, 8)
        adj = np.stack([Ko[:, k, k + 1] for k in range(7)], 1)
        cols.append(adj.mean(1, keepdims=True))
        for gam in (0.5, 0.9):
            W = np.triu(gam ** (np.abs(pos[:, None] - pos[None]) ), 1)
            cols.append((Ko * W[None]).sum((1, 2))[:, None] / W.sum())
        cols.append(np.stack([Ko[:, :4, :4].sum((1, 2)), Ko[:, 4:, 4:].sum((1, 2)), Ko[:, :4, 4:].sum((1, 2))], 1))
    X = np.concatenate(cols, 1)
    return np.concatenate([X, np.repeat(rf["r"][None], n, 0)], 1)
