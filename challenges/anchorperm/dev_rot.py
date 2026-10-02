import sys, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M
from dev_common import va, fit, rot
from dev_noise import noisy
for rp in [0.0, 0.3, 0.5]:
    model = M.train_model(fit, epochs=40, gain_max=1.6, rot_prob=rp, log=lambda *a: None)
    print("rot_prob", rp, "clean", M.evaluate(model, va)[0].mean().round(4), "rotated", M.evaluate(model, rot(va, 1))[0].mean().round(4), "noise0.7", M.evaluate(model, noisy(va, 0.7, 3))[0].mean().round(4), flush=True)
