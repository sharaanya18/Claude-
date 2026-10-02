import sys, time, torch, numpy as np, pandas as pd
sys.path.insert(0, "."); import ap_model as M
torch.set_num_threads(4)
df = pd.read_csv("dataset/public/train.csv", keep_default_na=False); rows = M.load_rows(df)
rng = np.random.default_rng(0); idx = rng.permutation(len(rows)); va = [rows[i] for i in idx[:700]]; fit = [rows[i] for i in idx[700:]]
ep = int(sys.argv[1]) if len(sys.argv) > 1 else 10
t = time.time(); model = M.train_model(fit, epochs=ep); print("train s", round(time.time() - t))
acc, _ = M.evaluate(model, va); ks = np.array([r["k"] for r in va])
print("val acc", acc.mean().round(4), {k: acc[ks == k].mean().round(3) for k in [2, 3, 4, 5]})
