import sys, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M
from scipy.optimize import linear_sum_assignment
from dev_common import va, fit
from dev_noise import noisy
torch.set_num_threads(4)
def sinkhorn(L, it=60):
    P = L - L.max(); K = P.copy()
    for _ in range(it):
        K = K - np.logaddexp.reduce(K, axis=1, keepdims=True)
        K = K - np.logaddexp.reduce(K, axis=0, keepdims=True)
    return np.exp(K)
def dec(logit, r, mode, tau=1.0):
    hq, ha = r["hq"], r["ha"]; L = logit[np.ix_(hq, ha)] / tau
    if mode == "sink":
        P = sinkhorn(L); ri, ci = linear_sum_assignment(-P)
    else:
        S = 0.5 * ((L - np.logaddexp.reduce(L, 1, keepdims=True)) + (L - np.logaddexp.reduce(L, 0, keepdims=True))); ri, ci = linear_sum_assignment(-S)
    return [ha[c] for c in ci]
model = M.train_model(fit, epochs=40, gain_max=1.6, log=lambda *a: None)
for name, rows in {"clean": va, "noise0.5": noisy(va, 0.5, 3), "noise0.7": noisy(va, 0.7, 3)}.items():
    lg = M.predict_logits(model, rows).numpy(); res = {}
    for mode, tau in [("hung", 1.0), ("sink", 1.0), ("sink", 2.0), ("sink", 0.5)]:
        res[f"{mode}{tau}"] = round(float(np.mean([np.mean([p == t for p, t in zip(dec(l, r, mode, tau), r["y"])]) for l, r in zip(lg, rows)])), 4)
    print(name, res, flush=True)
