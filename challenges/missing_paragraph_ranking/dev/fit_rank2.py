"""Train the listwise ranker on cached evidence2 profiles, per fold, with early stopping on
the inner-val split only. The outer-val number is reported but never used for any choice."""
import sys, argparse, time, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from common import score_from_matrix, metrics_from_matrix

ap = argparse.ArgumentParser()
ap.add_argument("cachedir"); ap.add_argument("--folds", type=str, default="0,1,2,3,4")
ap.add_argument("--hid", type=int, default=48); ap.add_argument("--depth", type=int, default=2)
ap.add_argument("--drop", type=float, default=0.3); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--setattn", type=int, default=1)
ap.add_argument("--nseed", type=int, default=3); ap.add_argument("--rank", type=int, default=1)
ap.add_argument("--tag", type=str, default="r2"); ap.add_argument("--quiet", type=int, default=0)
G = ap.parse_args(); torch.set_num_threads(4)

def featurise(Fp, mu, sd, use_rank):
    z = np.log1p(np.abs(Fp) * 100.0) * np.sign(Fp)
    parts = [(z - mu) / sd]
    pm = z.mean(-2, keepdims=True); ps = z.std(-2, keepdims=True) + 1e-6
    parts.append((z - pm) / ps)
    if use_rank:
        o = np.argsort(np.argsort(z, -2), -2).astype(np.float32)
        parts.append(o / (z.shape[-2] - 1.0) - 0.5)
    return np.concatenate(parts, -1).astype(np.float32)

class Net(nn.Module):
    def __init__(self, nin, hid, drop, depth, setattn):
        super().__init__()
        L = []; d = nin
        for _ in range(depth):
            L += [nn.Linear(d, hid), nn.GELU(), nn.Dropout(drop)]; d = hid
        self.body = nn.Sequential(*L); self.setattn = setattn
        if setattn:
            self.att = nn.MultiheadAttention(d, 4, batch_first=True, dropout=drop)
            self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
    def forward(self, x):
        h = self.body(x)
        if self.setattn: h = self.ln(h + self.att(h, h, h, need_weights=False)[0])
        return self.out(h).squeeze(-1)

def fit_one(Xv, Xiv, Xov, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    NVIEW, NQ = Xv.shape[0], Xv.shape[1]
    net = Net(Xv.shape[-1], G.hid, G.drop, G.depth, G.setattn)
    opt = torch.optim.AdamW(net.parameters(), lr=G.lr, weight_decay=G.wd)
    spe = max(NQ // G.bs, 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=G.lr, total_steps=G.epochs * spe, pct_start=0.2)
    Xt = torch.from_numpy(Xv); rng = np.random.default_rng(seed)
    def pred(Xs):
        net.eval()
        with torch.no_grad(): r = [net(torch.from_numpy(x)).numpy() for x in Xs]
        net.train(); return r
    best = (9.0, None, 0)
    for ep in range(G.epochs):
        for _ in range(spe):
            q = torch.from_numpy(rng.choice(NQ, size=G.bs, replace=False))
            v = torch.from_numpy(rng.integers(0, NVIEW, size=G.bs))
            loss = F.cross_entropy(net(Xt[v, q]), torch.zeros(G.bs, dtype=torch.long))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if (ep + 1) % 3 == 0:
            iv = float(np.mean([score_from_matrix(p) for p in pred(Xiv)]))
            if iv < best[0]: best = (iv, [p.copy() for p in pred(Xov)], ep + 1)
    return best

rows = []
for f in [int(x) for x in G.folds.split(",")]:
    Z = np.load(Path(G.cachedir) / f"e2_f{f}.npz")
    Ftr, Fiv, Fov = Z["Ftr"], Z["Fiv"], Z["Fov"]
    zz = np.log1p(np.abs(Ftr) * 100.0) * np.sign(Ftr)
    mu, sd = zz.reshape(-1, zz.shape[-1]).mean(0), zz.reshape(-1, zz.shape[-1]).std(0) + 1e-6
    Xv = featurise(Ftr, mu, sd, G.rank)
    Xiv = [featurise(v, mu, sd, G.rank) for v in Fiv]
    Xov = [featurise(v, mu, sd, G.rank) for v in Fov]
    cos_iv = float(np.mean([score_from_matrix(v[:, :, 0]) for v in Fiv]))
    cos_ov = float(np.mean([score_from_matrix(v[:, :, 0]) for v in Fov]))
    accum, ivs, eps = None, [], []
    for s in range(G.nseed):
        iv, povs, ep = fit_one(Xv, Xiv, Xov, 1000 + s)
        ivs.append(iv); eps.append(ep)
        r = [(-np.argsort(np.argsort(-p, 1), 1)).astype(np.float64) for p in povs]  # rank-average seeds
        accum = r if accum is None else [a + b for a, b in zip(accum, r)]
    ov = float(np.mean([score_from_matrix(p) for p in accum]))
    m = metrics_from_matrix(np.concatenate(accum, 0))
    rows.append(dict(fold=f, cos_iv=cos_iv, cos_ov=cos_ov, iv=float(np.mean(ivs)), ov=ov,
                     ep=float(np.mean(eps)), **{k: m[k] for k in ("top1", "top3", "top5", "mean_rank")}))
    if not G.quiet:
        print(f"  fold {f}: cosine iv {cos_iv:.4f} ov {cos_ov:.4f} | ranker iv {np.mean(ivs):.4f} "
              f"ov {ov:.4f} (ep~{np.mean(eps):.0f})")
C = np.mean([r["cos_ov"] for r in rows]); R = np.mean([r["ov"] for r in rows])
print(f"[{G.tag}] hid={G.hid} depth={G.depth} drop={G.drop} wd={G.wd} attn={G.setattn} rank={G.rank} "
      f"seeds={G.nseed} ep={G.epochs}")
print(f"   cosine  outer {C:.4f} +- {np.std([r['cos_ov'] for r in rows]):.4f}")
print(f"   RANKER  outer {R:.4f} +- {np.std([r['ov'] for r in rows]):.4f}   gain {C-R:+.4f}   "
      f"top1 {np.mean([r['top1'] for r in rows]):.3f} top5 {np.mean([r['top5'] for r in rows]):.3f} "
      f"mrank {np.mean([r['mean_rank'] for r in rows]):.2f}  ep~{np.mean([r['ep'] for r in rows]):.0f}")
