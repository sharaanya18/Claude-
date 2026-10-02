import sys, time, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M, ap_adapt as AD
from dev_common import va, fit
torch.set_num_threads(4)
base = M.train_model(fit, epochs=40, gain_max=1.6, log=lambda *a: None)
sets = {}
for name, ml in [("rot0.7", [AD.regime_maps(0.7, 1)]), ("rot1.0", [AD.regime_maps(1.0, 2)]), ("3reg", [AD.regime_maps(0.7, 3), AD.regime_maps(1.0, 4), AD.regime_maps(1.5, 5)])]:
    k = len(ml); parts = [va[i::k] for i in range(k)]; sets[name] = [r for p, m in zip(parts, ml) for r in AD.apply_regime(p, m, seed=7)]
b = {n: M.evaluate(base, r)[0].mean() for n, r in sets.items()}; print("base", {n: round(float(v), 4) for n, v in b.items()}, flush=True)
for cfg in [dict(epochs=10, lr=5e-4), dict(epochs=25, lr=5e-4), dict(epochs=25, lr=1e-3), dict(epochs=50, lr=1e-3), dict(epochs=25, lr=2e-3), dict(epochs=40, lr=2e-3)]:
    t = time.time(); res = {}
    for n, r in sets.items():
        ad = AD.adapt(base, fit, r, **cfg); res[n] = round(float(M.evaluate(ad, r)[0].mean() - b[n]), 4)
    print(cfg, "gain", res, f"[{time.time() - t:.0f}s]", flush=True)
