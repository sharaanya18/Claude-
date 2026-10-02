import sys, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M
from dev_common import va, fit
from dev_noise import noisy
torch.set_num_threads(4); vn = noisy(va, 0.7, 3)
for frac in [0.25, 0.5, 1.0]:
    sub = fit[:int(len(fit) * frac)]
    model = M.train_model(sub, epochs=int(40 / frac ** 0.5), gain_max=1.6, log=lambda *a: None)
    print("train rows", len(sub), "clean", M.evaluate(model, va)[0].mean().round(4), "noise0.7", M.evaluate(model, vn)[0].mean().round(4), flush=True)
