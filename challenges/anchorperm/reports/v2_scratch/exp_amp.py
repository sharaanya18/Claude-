import sys, os, numpy as np, torch
sys.path.insert(0, "/home/user/Claude-/challenges/anchorperm/reports/v2_scratch")
from sims import *
import ap_model as M
from dev_common import rot
torch.set_num_threads(4)
exec(open("/home/user/Claude-/challenges/anchorperm/reports/v2_scratch/gmfun.py").read())
sets = {"clean": va, "amp1.8r8": amp(va, 1.8, 8, 1), "amp1.8r8+rotfull": rot(amp(va, 1.8, 8, 1), 2),
        "amp1.8r8+prot0.7": amp(va, 1.8, 8, 3, 0.7), "prot0.7": amp(va, 1.0, 8, 4, 0.7), "coordmix0.5": coordmix(va, 0.5, 5)}
for k, v in sets.items():
    print(k, "row std", round(float(np.mean([r["Q"].std(0).mean() for r in v])), 3), "GM0", round(float(np.mean([gm(r, "cos", 0) for r in v])), 4), "GM10", round(float(np.mean([gm(r, "cos", 10, 2.0, 1.0) for r in v])), 4), flush=True)
model = M.train_model(fit, epochs=40, gain_max=1.6, log=lambda *a: None)
print("model v3cfg", {k: round(float(M.evaluate(model, v)[0].mean()), 4) for k, v in sets.items()}, flush=True)
