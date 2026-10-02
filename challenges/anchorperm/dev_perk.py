import sys, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M
from dev_common import va, fit
def perk(rows, seed, gl=0.7, gh=1.8, bs=0.4, signflip=False, perm=False):
    g = np.random.default_rng(seed)
    gq, ga = g.uniform(gl, gh, 32), g.uniform(gl, gh, 32); bq, ba = g.normal(0, bs, 32), g.normal(0, bs, 32)
    sq = np.where(g.random(32) < 0.5, -1, 1) if signflip else np.ones(32); sa = np.where(g.random(32) < 0.5, -1, 1) if signflip else np.ones(32)
    pq = g.permutation(32) if perm else np.arange(32); pa = g.permutation(32) if perm else np.arange(32)
    out = []
    for r in rows:
        r = dict(r)
        r["Q"] = np.clip(np.round(r["Q"][:, pq] * gq * sq + bq), -2, 2).astype(np.float32); r["A"] = np.clip(np.round(r["A"][:, pa] * ga * sa + ba), -2, 2).astype(np.float32); out.append(r)
    return out
if __name__ == "__main__":
    model = M.train_model(fit, epochs=40, gain_max=1.6, log=lambda *a: None)
    print("clean", M.evaluate(model, va)[0].mean().round(4))
    for name, kw in [("per-coord gain+bias", {}), ("gain only", dict(bs=0.0)), ("sign flips", dict(gl=1, gh=1, bs=0, signflip=True)), ("permuted coords", dict(gl=1, gh=1, bs=0, perm=True))]:
        print(name, M.evaluate(model, perk(va, 5, **kw))[0].mean().round(4), flush=True)
