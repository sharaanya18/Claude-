import sys, os, numpy as np, torch
sys.path.insert(0, "/home/user/Claude-/challenges/anchorperm/reports/v2_scratch")
from sims import *
import ap_model as M
from dev_common import rot
from scipy.optimize import linear_sum_assignment as lsa
torch.set_num_threads(4)
N = 500
sets = {"clean": va[:N], "prot0.7": amp(va[:N], 1.0, 8, 4, 0.7), "prot1.0": amp(va[:N], 1.0, 8, 4, 1.0), "coordmix0.5": coordmix(va[:N], 0.5, 5), "rotfull": rot(va[:N], 2)}
F = M.train_model(fit, epochs=40, gain_max=1.6, seed=0, log=lambda *a: None)
R = M.train_model(fit, epochs=40, gain_max=1.6, rot_prob=0.3, seed=1, log=lambda *a: None)
def z(S): return (S - S.mean()) / (S.std() + 1e-9)
for name, rows in sets.items():
    lf, lr_ = M.predict_logits(F, rows).numpy(), M.predict_logits(R, rows).numpy(); res = {"F": [], "R": [], "avg": [], "oracle": [], "agree": []}
    for a, b, r in zip(lf, lr_, rows):
        pf, Sf = M.decode_row(a, r); pr, Sr = M.decode_row(b, r)
        af = np.mean([p == t for p, t in zip(pf, r["y"])]); ar = np.mean([p == t for p, t in zip(pr, r["y"])])
        ri, ci = lsa(-(z(Sf) + z(Sr))); aa = np.mean([r["ha"][c] == t for c, t in zip(ci, r["y"])])
        res["F"].append(af); res["R"].append(ar); res["avg"].append(aa); res["oracle"].append(max(af, ar)); res["agree"].append(np.mean([p == q for p, q in zip(pf, pr)]))
    print(name, {k: round(float(np.mean(v)), 4) for k, v in res.items()}, flush=True)
