import sys, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M
from dev_common import va, fit
torch.set_num_threads(4)
def prot(rows, theta, seed):
    g = np.random.default_rng(seed); out = []
    for r in rows:
        r = dict(r)
        for k in ("Q", "A"):
            G = g.normal(size=(32, 32)) / 32 ** 0.5; S = (G - G.T) / 2 ** 0.5 * theta
            from scipy.linalg import expm
            r[k] = np.clip(np.round(r[k] @ expm(S)), -2, 2).astype(np.float32)
        out.append(r)
    return out
sets = {"clean": va, "rot0.3": prot(va, 0.3, 1), "rot0.7": prot(va, 0.7, 2), "rot1.5": prot(va, 1.5, 3)}
for name, kw in [("v1 (no rot)", dict()), ("full rot 0.3", dict(rot_prob=0.3)), ("partial<=1.0 p0.6", dict(rot_prob=0.6, partial=1.0)), ("partial<=2.0 p0.6", dict(rot_prob=0.6, partial=2.0))]:
    model = M.train_model(fit, epochs=40, gain_max=1.6, log=lambda *a: None, **kw)
    print(name, {k: round(float(M.evaluate(model, v)[0].mean()), 3) for k, v in sets.items()}, flush=True)
