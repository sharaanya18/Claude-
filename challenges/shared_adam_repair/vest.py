"""Learned estimate of the common second-moment vector v (per parameter) from public per-request inputs."""
import sys; sys.path.insert(0,'.')
import numpy as np
from sim import *
GROUPS = np.concatenate([np.zeros(2304), np.ones(12), np.full(48, 2), np.full(4, 3)]).astype(int)

def per_image_grads2(w, X, y):
    return np.stack([L.gradient(w, X[k:k+1], y[k:k+1]) for k in range(len(y))])

def factor_feats(d):
    """Rank-1 (pixel energy x hidden-delta energy) smoothing of the per-image squared gradient, for W1; other groups reuse per-image g2."""
    w=d["weights"]; a,b,c,dd=L.unpack(w)
    X=np.concatenate([d["x_batches"].reshape(-1,192),d["x_monitor"],d["x_diag"].reshape(-1,192)])
    y=np.concatenate([d["batch_labels"].ravel(),d["monitor_labels"],d["diagnostic_labels"].ravel()])
    h=np.tanh(X@a+b); z=h@c+dd; z-=z.max(1,keepdims=True); q=np.exp(z); q/=q.sum(1,keepdims=True)
    q[np.arange(len(y)),y]-=1
    dh=(q@c.T)*(1-h*h)                                  # (n,12) per-image hidden deltas (no 1/n)
    S=np.log((X**2).mean(0)+1e-12); T=np.log((dh**2).mean(0)+1e-12)
    f=np.zeros((L.P,3),np.float32)
    f[:2304,0]=np.repeat(S,12); f[:2304,1]=np.tile(T,192); f[:2304,2]=f[:2304,0]+f[:2304,1]
    f[2304:2316,1]=T; f[2304:2316,2]=T
    hq=np.log((h**2).mean(0)+1e-12); Tq=np.log((q**2).mean(0)+1e-12)
    f[2316:2364,0]=np.repeat(hq,4); f[2316:2364,1]=np.tile(Tq,12); f[2316:2364,2]=f[2316:2364,0]+f[2316:2364,1]
    f[2364:,1]=Tq; f[2364:,2]=Tq
    return f

def v_features(d):
    """(P, F) parameter-level features + (R,) request-level scalars; public inputs only."""
    w = d["weights"]
    Gb = np.stack([L.gradient(w, d["x_batches"][j], d["batch_labels"][j]) for j in range(8)])
    Gd = np.stack([L.gradient(w, d["x_diag"][j], d["diagnostic_labels"][j]) for j in range(2)])
    Gm = L.gradient(w, d["x_monitor"], d["monitor_labels"])
    Gi = np.concatenate([per_image_grads2(w, d["x_batches"][j], d["batch_labels"][j]) for j in range(8)])
    Gmi = per_image_grads2(w, d["x_monitor"], d["monitor_labels"])
    e = 1e-12
    f_b = np.log((Gb ** 2).mean(0) + e); f_d = np.log((Gd ** 2).mean(0) + e); f_m = np.log(Gm ** 2 + e)
    f_i = np.log((Gi ** 2).mean(0) + e); f_mi = np.log((Gmi ** 2).mean(0) + e); f_bm = np.log(((Gb.mean(0)) ** 2) + e)
    f_w = np.abs(w)
    scal = np.array([np.log((Gb ** 2).mean() + e), np.log((Gi ** 2).mean() + e), np.log((Gmi ** 2).mean() + e),
                     L.cross_entropy(w, d["x_monitor"], d["monitor_labels"]),
                     np.mean([L.cross_entropy(w, d["x_batches"][j], d["batch_labels"][j]) for j in range(8)]),
                     np.log(d["request"]["divergence_limit"]), np.log(d["request"]["capability_limit"]), np.log(np.linalg.norm(w))])
    F = np.stack([f_b, f_d, f_m, f_i, f_mi, f_bm, f_w, f_b - scal[0], f_i - scal[1]], 1)
    F = np.concatenate([F, GROUPS[:, None], np.repeat(scal[None], L.P, 0), factor_feats(d)], 1)
    return F.astype(np.float32)
