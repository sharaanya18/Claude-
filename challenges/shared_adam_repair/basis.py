import sys; sys.path.insert(0,'.')
import numpy as np
from sim import *

def grad_basis(d, per_image=False):
    w = d["weights"]; cols = []
    for j in range(8): cols.append(L.gradient(w, d["x_batches"][j], d["batch_labels"][j]))
    for j in range(2): cols.append(L.gradient(w, d["x_diag"][j], d["diagnostic_labels"][j]))
    cols.append(L.gradient(w, d["x_monitor"], d["monitor_labels"]))
    if per_image:
        for j in range(8):
            for k in range(4): cols.append(L.gradient(w, d["x_batches"][j][k:k+1], d["batch_labels"][j][k:k+1]))
        for k in range(12): cols.append(L.gradient(w, d["x_monitor"][k:k+1], d["monitor_labels"][k:k+1]))
    return np.stack(cols).T

def ols_state(d, G=None):
    G = grad_basis(d) if G is None else G
    out = np.zeros((2, 2, G.shape[0]))
    G2 = np.concatenate([G ** 2, np.ones((G.shape[0], 1))], 1)
    for b in range(2):
        cm, *_ = np.linalg.lstsq(G, d["moments"][b, 0], rcond=None); out[b, 0] = G @ cm
        cv, *_ = np.linalg.lstsq(G2, d["moments"][b, 1], rcond=None); out[b, 1] = np.maximum(G2 @ cv, 1e-12)
    return out
