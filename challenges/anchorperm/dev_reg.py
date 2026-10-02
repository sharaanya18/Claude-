import sys, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M
from dev_common import va, fit, scale
torch.set_num_threads(2); sub = fit[:700]
for drop, wd, ep in [(0.0, 1e-2, 40), (0.2, 1e-2, 60), (0.3, 5e-2, 80), (0.3, 0.2, 80)]:
    model = M.train_model(fit, epochs=ep, gain_max=1.6, drop=drop, wd=wd, log=lambda *a: None)
    a, b = M.evaluate(model, sub)[0].mean(), M.evaluate(model, va)[0].mean()
    print(f"drop {drop} wd {wd} ep {ep}: train {a:.4f} val {b:.4f} gap {a - b:+.4f} stressed {M.evaluate(model, scale(va, 1.5, 0.3, 2))[0].mean():.4f}", flush=True)
