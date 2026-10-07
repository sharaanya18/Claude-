"""Sweep listwise ranker configurations on cached evidence profiles.
Model selection (architecture, capacity, epoch) uses the INNER-VAL split only; the outer-val
number is reported alongside but never used to choose, so it stays an honest estimate."""
import sys, argparse, itertools, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from common import metrics_from_matrix, score_from_matrix

ap = argparse.ArgumentParser()
ap.add_argument("cache"); ap.add_argument("--grid", type=str, default="")
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--seed", type=int, default=0)
G = ap.parse_args()
torch.set_num_threads(4)
Z = np.load(G.cache)
Ftr, Fiv, Fov = Z["Ftr"], Z["Fiv"], Z["Fov"]

def featurise(Fp, mu, sd, poolrel=True):
    z = np.log1p(np.abs(Fp) * 100.0) * np.sign(Fp)
    zs = (z - mu) / sd
    if not poolrel: return zs.astype(np.float32)
    pm = z.mean(-2, keepdims=True); ps = z.std(-2, keepdims=True) + 1e-6
    return np.concatenate([zs, (z - pm) / ps], -1).astype(np.float32)

zz = np.log1p(np.abs(Ftr) * 100.0) * np.sign(Ftr)
MU, SD = zz.reshape(-1, zz.shape[-1]).mean(0), zz.reshape(-1, zz.shape[-1]).std(0) + 1e-6
print("raw cosine baselines:  inner-val %.4f   outer-val %.4f" %
      (np.mean([score_from_matrix(Fiv[i][:, :, Z["ncellL"]]) for i in range(len(Fiv))]),
       np.mean([score_from_matrix(Fov[i][:, :, Z["ncellL"]]) for i in range(len(Fov))])))

class Net(nn.Module):
    def __init__(self, nin, hid, drop, depth, setattn):
        super().__init__()
        if depth == 0:
            self.body = nn.Identity(); d = nin
        else:
            L = []; d = nin
            for _ in range(depth):
                L += [nn.Linear(d, hid), nn.GELU(), nn.Dropout(drop)]; d = hid
            self.body = nn.Sequential(*L)
        self.setattn = setattn
        if setattn:
            self.att = nn.MultiheadAttention(d, 4, batch_first=True, dropout=drop); self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
    def forward(self, x):
        h = self.body(x)
        if self.setattn: h = self.ln(h + self.att(h, h, h, need_weights=False)[0])
        return self.out(h).squeeze(-1)

def run(hid, drop, depth, setattn, lr, wd, poolrel, bs=64, epochs=G.epochs, seed=0, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = featurise(Ftr, MU, SD, poolrel).reshape(-1, 20, Ftr.shape[-1] * (2 if poolrel else 1))
    Xiv = [torch.from_numpy(featurise(v, MU, SD, poolrel)) for v in Fiv]
    Xov = [torch.from_numpy(featurise(v, MU, SD, poolrel)) for v in Fov]
    net = Net(Xtr.shape[-1], hid, drop, depth, setattn)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    spe = max(len(Xtr) // bs, 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * spe, pct_start=0.2)
    Xt = torch.from_numpy(Xtr); rng = np.random.default_rng(seed)
    def sc(Xs):
        net.eval()
        with torch.no_grad():
            r = float(np.mean([score_from_matrix(net(x).numpy()) for x in Xs]))
        net.train(); return r
    best = (9, 9, -1)
    for ep in range(epochs):
        for _ in range(spe):
            sel = torch.from_numpy(rng.choice(len(Xt), size=bs, replace=False))
            loss = F.cross_entropy(net(Xt[sel]), torch.zeros(bs, dtype=torch.long))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if (ep + 1) % 4 == 0:
            iv, ov = sc(Xiv), sc(Xov)
            if iv < best[0]: best = (iv, ov, ep + 1)
            if verbose: print(f"    ep{ep+1:3d} inner {iv:.4f} outer {ov:.4f}")
    return best

GRID = [
    dict(hid=0,   drop=0.0, depth=0, setattn=0, lr=3e-3, wd=1e-3, poolrel=0),
    dict(hid=0,   drop=0.0, depth=0, setattn=0, lr=3e-3, wd=1e-3, poolrel=1),
    dict(hid=32,  drop=0.3, depth=1, setattn=0, lr=2e-3, wd=1e-2, poolrel=1),
    dict(hid=64,  drop=0.3, depth=1, setattn=0, lr=2e-3, wd=1e-2, poolrel=1),
    dict(hid=64,  drop=0.3, depth=2, setattn=0, lr=2e-3, wd=1e-2, poolrel=1),
    dict(hid=128, drop=0.3, depth=2, setattn=0, lr=2e-3, wd=1e-2, poolrel=1),
    dict(hid=64,  drop=0.3, depth=2, setattn=1, lr=2e-3, wd=1e-2, poolrel=1),
    dict(hid=128, drop=0.5, depth=2, setattn=1, lr=2e-3, wd=3e-2, poolrel=1),
    dict(hid=64,  drop=0.1, depth=1, setattn=0, lr=2e-3, wd=1e-3, poolrel=1),
]
print(f"{'hid':>4s} {'dep':>3s} {'drop':>5s} {'att':>3s} {'wd':>6s} {'prel':>4s} | {'inner':>7s} {'outer':>7s} {'ep':>4s}  t")
for c in GRID:
    t0 = time.time(); iv, ov, ep = run(**c)
    print(f"{c['hid']:4d} {c['depth']:3d} {c['drop']:5.2f} {c['setattn']:3d} {c['wd']:6.3f} {c['poolrel']:4d} | "
          f"{iv:7.4f} {ov:7.4f} {ep:4d}  {time.time()-t0:.0f}s")
