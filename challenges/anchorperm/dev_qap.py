import sys, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M, ap_qap as Q
from dev_common import va, fit, rot
from dev_noise import noisy
torch.set_num_threads(4)
model = M.train_model(fit, epochs=40, gain_max=1.6, log=lambda *a: None)
sets = {"clean": va, "noise0.7": noisy(va, 0.7, 3), "rotated": rot(va, 1)}
for name, rows in sets.items():
    lg = M.predict_logits(model, rows).numpy(); out = {}
    for lam in [0.0, 0.02, 0.05, 0.1, 0.2]:
        acc = []
        for l, r in zip(lg, rows):
            pred0, S = M.decode_row(l, r)
            pred = pred0 if lam == 0 else Q.qap_refine(r, S, lam)
            acc.append(np.mean([p == t for p, t in zip(pred, r["y"])]))
        out[lam] = round(float(np.mean(acc)), 4)
    print(name, out, flush=True)
