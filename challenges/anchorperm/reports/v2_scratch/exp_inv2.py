import sys, os, numpy as np
sys.path.insert(0, "/home/user/Claude-/challenges/anchorperm/reports/v2_scratch")
from sims import *
from dev_common import rot
from dev_noise import noisy
exec(open("/home/user/Claude-/challenges/anchorperm/reports/v2_scratch/gmfun.py").read())
def zv(v): return (v - v.mean()) / (v.std() + 1e-9)
def inv_acc(r, beta, iters=10, tau=2.0):
    Kq, Ka = kern(r["Q"], "cos"), kern(r["A"], "cos")
    hq, ha = r["hq"], r["ha"]; aq = [a for a, b in r["anc"]]; ab = [b for a, b in r["anc"]]
    nq = zv(np.linalg.norm(r["Q"] - r["Q"].mean(0), axis=1)); na = zv(np.linalg.norm(r["A"] - r["A"].mean(0), axis=1))
    U = beta * np.outer(nq[hq], na[ha])
    S0 = Kq[np.ix_(hq, aq)] @ Ka[np.ix_(ha, ab)].T + U; S = S0.copy()
    for _ in range(iters):
        L = S / tau
        for _ in range(30): L = L - np.logaddexp.reduce(L, 1, keepdims=True); L = L - np.logaddexp.reduce(L, 0, keepdims=True)
        S = S0 + Kq[np.ix_(hq, hq)] @ np.exp(L) @ Ka[np.ix_(ha, ha)].T
    ri, ci = lsa(-S); return np.mean([ha[c] == t for c, t in zip(ci, r["y"])])
N = 500
sets = {"clean": va[:N], "rotfull": rot(va[:N], 2), "prot1.0": amp(va[:N], 1.0, 8, 4, 1.0), "coordmix0.5": coordmix(va[:N], 0.5, 5), "noise0.5": noisy(va[:N], 0.5, 3)}
for beta in (0.0, 1.0, 2.0, 4.0):
    print("beta", beta, {k: round(float(np.mean([inv_acc(r, beta) for r in v])), 4) for k, v in sets.items()}, flush=True)
