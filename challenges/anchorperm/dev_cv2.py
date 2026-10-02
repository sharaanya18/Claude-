import sys, time, torch, numpy as np, pandas as pd
sys.path.insert(0, "."); import ap_model as M
torch.set_num_threads(4)
df = pd.read_csv("dataset/public/train.csv", keep_default_na=False); rows = M.load_rows(df)
rng = np.random.default_rng(0); idx = rng.permutation(len(rows)); va = [rows[i] for i in idx[:700]]; fit = [rows[i] for i in idx[700:]]
def rot(rs, seed):
    g = np.random.default_rng(seed); out = []
    for r in rs:
        r = dict(r); R1 = np.linalg.qr(g.normal(size=(32, 32)))[0].astype(np.float32); R2 = np.linalg.qr(g.normal(size=(32, 32)))[0].astype(np.float32)
        r["Q"] = r["Q"] @ R1; r["A"] = r["A"] @ R2; out.append(r)
    return out
def scale(rs, g_, noise, seed):
    g = np.random.default_rng(seed); out = []
    for r in rs:
        r = dict(r); r["Q"] = np.clip(np.round(r["Q"] * g_ + g.normal(0, noise, r["Q"].shape)), -2, 2).astype(np.float32); r["A"] = np.clip(np.round(r["A"] * g_ + g.normal(0, noise, r["A"].shape)), -2, 2).astype(np.float32); out.append(r)
    return out
for tower in [True, False]:
    model = M.train_model(fit, epochs=40, use_tower=tower, log=lambda *a: None)
    print("tower", tower, "clean", M.evaluate(model, va)[0].mean().round(4), "rotated", M.evaluate(model, rot(va, 1))[0].mean().round(4),
          "gain1.5+noise", M.evaluate(model, scale(va, 1.5, 0.3, 2))[0].mean().round(4))
