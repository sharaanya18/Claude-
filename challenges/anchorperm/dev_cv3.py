import sys, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M
from dev_common import rot, scale, va, fit
for gm, nm_ in [(1.0, 0.6), (1.6, 0.6), (2.0, 0.8)]:
    model = M.train_model(fit, epochs=40, gain_max=gm, noise_max=nm_, log=lambda *a: None)
    print("gain_max", gm, "noise", nm_, "clean", M.evaluate(model, va)[0].mean().round(4),
          "g1.5n0.3", M.evaluate(model, scale(va, 1.5, 0.3, 2))[0].mean().round(4), "g1.8n0.5", M.evaluate(model, scale(va, 1.8, 0.5, 3))[0].mean().round(4))
