import sys, time, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M
from dev_common import rot, scale, va, fit
for cl in [0, 1, 2]:
    t = time.time(); model = M.train_model(fit, epochs=40, gain_max=1.6, ctx_layers=cl, log=lambda *a: None)
    print("ctx_layers", cl, "clean", M.evaluate(model, va)[0].mean().round(4), "g1.5n0.3", M.evaluate(model, scale(va, 1.5, 0.3, 2))[0].mean().round(4), "s", round(time.time() - t))
