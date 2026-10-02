import sys, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M
from scipy.optimize import linear_sum_assignment
from dev_common import va, fit
from dev_partial import prot
from dev_noise import noisy
torch.set_num_threads(4)
M.K_MAX = 12
def sinkhorn(L, it=40):
    K = L - L.max()
    for _ in range(it):
        K = K - np.logaddexp.reduce(K, axis=1, keepdims=True); K = K - np.logaddexp.reduce(K, axis=0, keepdims=True)
    return np.exp(K)
def two_pass(model, rows, n_pseudo):
    lg = M.predict_logits(model, rows).numpy(); new_rows = []
    for l, r in zip(lg, rows):
        pred, S = M.decode_row(l, r)
        hq, ha = r["hq"], r["ha"]; P = sinkhorn(l[np.ix_(hq, ha)]); ci = [ha.index(a) for a in pred]
        conf = P[np.arange(len(hq)), ci]; take = np.argsort(-conf)[:n_pseudo]
        r2 = dict(r); r2["anc"] = list(r["anc"]) + [(hq[i], pred[i]) for i in take]
        r2["hq"] = [q for j, q in enumerate(hq) if j not in set(take)]; keep = [j for j in range(len(hq)) if j not in set(take)]
        r2["_orig"] = (hq, pred, take); new_rows.append(r2)
    lg2 = M.predict_logits(model, [dict(r, anc=r2["anc"]) for r, r2 in zip(rows, new_rows)]).numpy(); accs = []
    for l, r, r2 in zip(lg2, rows, new_rows):
        hq, pred, take = r2["_orig"]; ta = set(take.tolist()); rem = [j for j in range(len(hq)) if j not in ta]
        ha2 = [a for a in r["ha"] if a not in {pred[i] for i in take}]; hq2 = [hq[j] for j in rem]
        if not hq2: accs.append(np.mean([p == t for p, t in zip(pred, r["y"])])); continue
        L = l[np.ix_(hq2, ha2)]; S = 0.5 * ((L - np.logaddexp.reduce(L, 1, keepdims=True)) + (L - np.logaddexp.reduce(L, 0, keepdims=True)))
        ri, ci = linear_sum_assignment(-S); final = list(pred)
        for jj, c in zip(rem, ci): final[jj] = ha2[c]
        accs.append(np.mean([p == t for p, t in zip(final, r["y"])]))
    return float(np.mean(accs))
model = M.train_model(fit, epochs=40, gain_max=1.6, log=lambda *a: None)
for name, rows in {"clean": va, "noise0.5": noisy(va, 0.5, 3), "rot0.5": prot(va, 0.5, 1), "rot0.7": prot(va, 0.7, 2)}.items():
    print(name, {p: round(two_pass(model, rows, p), 4) if p else round(float(M.evaluate(model, rows)[0].mean()), 4) for p in [0, 1, 2, 3]}, flush=True)
