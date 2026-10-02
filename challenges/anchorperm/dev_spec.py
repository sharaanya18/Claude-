import sys, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M
from dev_common import va, fit
from dev_noise import noisy
for nmin, nmax in [(0.0, 1.0), (0.5, 0.9), (0.6, 0.8)]:
    model = M.train_model(fit, epochs=40, noise_min=nmin, noise_max=nmax, requant=True, log=lambda *a: None)
    res = [M.evaluate(model, va)[0].mean()] + [M.evaluate(model, noisy(va, s, 3))[0].mean() for s in (0.5, 0.7)]
    print(f"train noise [{nmin},{nmax}] requant: clean {res[0]:.4f} sigma0.5 {res[1]:.4f} sigma0.7 {res[2]:.4f}", flush=True)
