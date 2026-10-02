import sys, os, numpy as np, torch
sys.path.insert(0, "/home/user/Claude-/challenges/anchorperm/reports/v2_scratch")
from sims import *
import ap_model as M
from dev_common import rot
from dev_noise import noisy
torch.set_num_threads(4)
exec(open("/home/user/Claude-/challenges/anchorperm/reports/v2_scratch/gmfun.py").read())
def gm_S(r, iters=10, tau=2.0, lam=1.0):
    Kq, Ka = kern(r["Q"], "cos"), kern(r["A"], "cos")
    hq, ha = r["hq"], r["ha"]; aq = [a for a, b in r["anc"]]; ab = [b for a, b in r["anc"]]
    S0 = Kq[np.ix_(hq, aq)] @ Ka[np.ix_(ha, ab)].T; S = S0.copy()
    for _ in range(iters):
        L = S / tau
        for _ in range(30): L = L - np.logaddexp.reduce(L, 1, keepdims=True); L = L - np.logaddexp.reduce(L, 0, keepdims=True)
        S = S0 + lam * Kq[np.ix_(hq, hq)] @ np.exp(L) @ Ka[np.ix_(ha, ha)].T
    return S, Kq, Ka
def z(S): return (S - S.mean()) / (S.std() + 1e-9)
def gwcons(Kq, Ka, r, assign):
    """label-free per-row structural consistency of a full assignment (anchors + predicted hidden): corr of Kq and permuted Ka."""
    n = r["n"]; p = np.zeros(n, int)
    for a, b in r["anc"]: p[a] = b
    for qi, aj in zip(r["hq"], assign): p[qi] = aj
    iu = np.triu_indices(n, 1); return np.corrcoef(Kq[iu], Ka[np.ix_(p, p)][iu])[0, 1]
model = M.train_model(fit, epochs=40, gain_max=1.6, log=lambda *a: None)
sets = {"clean": va, "noise0.7": noisy(va, 0.7, 3), "prot0.5": amp(va, 1.0, 8, 4, 0.5), "prot0.7": amp(va, 1.0, 8, 4, 0.7), "prot1.0": amp(va, 1.0, 8, 4, 1.0), "rotfull": rot(va, 2)}
W = [0.0, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0]
from scipy.optimize import linear_sum_assignment as lsa
for name, rows in sets.items():
    lg = M.predict_logits(model, rows).numpy(); acc = {w: [] for w in W}; cons_t, acc_t = [], []
    for l, r in zip(lg, rows):
        pred, St = M.decode_row(l, r); Sg, Kq, Ka = gm_S(r)
        cons_t.append(gwcons(Kq, Ka, r, pred)); acc_t.append(np.mean([p == t for p, t in zip(pred, r["y"])]))
        for w in W:
            ri, ci = lsa(-(z(St) + w * z(Sg))); acc[w].append(np.mean([r["ha"][c] == t for c, t in zip(ci, r["y"])]))
    print(name, {w: round(float(np.mean(a)), 4) for w, a in acc.items()}, "corr(GWcons, acc_tower)", round(float(np.corrcoef(cons_t, acc_t)[0, 1]), 3), flush=True)
