import sys, time, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M, ap_adapt as AD
from dev_common import va, fit
torch.set_num_threads(4)
base = M.train_model(fit, epochs=40, gain_max=1.6, log=lambda *a: None)
print("base clean", M.evaluate(base, va)[0].mean().round(4), flush=True)
for name, mats_list in [("shared rot0.7", [AD.regime_maps(0.7, 1)]), ("shared rot1.0", [AD.regime_maps(1.0, 2)]), ("3 regimes rot0.7/1.0/1.5", [AD.regime_maps(0.7, 3), AD.regime_maps(1.0, 4), AD.regime_maps(1.5, 5)])]:
    k = len(mats_list); parts = [va[i::k] for i in range(k)]
    shifted = [AD.apply_regime(p, m, seed=7) for p, m in zip(parts, mats_list)]
    allrows = [r for s in shifted for r in s]
    b = M.evaluate(base, allrows)[0].mean()
    t = time.time(); accs = []
    # adapt once on the pooled rows of ALL regimes (the solution cannot know regimes)
    ad = AD.adapt(base, fit, allrows, epochs=10)
    a = M.evaluate(ad, allrows)[0].mean()
    print(f"{name}: base {b:.4f} -> adapted {a:.4f}  (+{a - b:.4f})  [{time.time() - t:.0f}s]", flush=True)
