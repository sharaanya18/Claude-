import sys, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M
from dev_common import va, fit
from dev_noise import noisy
for rich in [0, 64, 128]:
    model = M.train_model(fit, epochs=40, gain_max=1.6, rich=rich, log=lambda *a: None)
    print("rich", rich, "clean", M.evaluate(model, va)[0].mean().round(4), "noise0.7", M.evaluate(model, noisy(va, 0.7, 3))[0].mean().round(4), flush=True)
