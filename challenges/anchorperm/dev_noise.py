import sys, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M
from dev_common import va, fit
def noisy(rs, s, seed):
    g = np.random.default_rng(seed); out = []
    for r in rs:
        r = dict(r)
        for k in ("Q", "A"): r[k] = np.clip(np.round(r[k] + g.normal(0, s, r[k].shape)), -2, 2).astype(np.float32)
        out.append(r)
    return out
if __name__ == "__main__":
  for nmax, gmax in [(0.6, 1.6), (1.0, 1.0), (1.2, 1.0), (1.4, 1.0)]:
      model = M.train_model(fit, epochs=40, noise_max=nmax, gain_max=gmax, log=lambda *a: None)
      res = [M.evaluate(model, va)[0].mean()] + [M.evaluate(model, noisy(va, s, 3))[0].mean() for s in (0.5, 0.7)]
      print(f"train noise_max {nmax} gain_max {gmax}: clean {res[0]:.4f} sigma0.5 {res[1]:.4f} sigma0.7 {res[2]:.4f}", flush=True)
