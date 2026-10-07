"""Model B: character-level CONVOLUTIONAL dual encoder trained from random initialisation.
Two objectives, both on training abstracts only:
  (1) MATCH  - listwise softmax (InfoNCE) of opening_i against same-subfield contributions,
  (2) DENOISE - two independently corrupted views of the SAME snippet must embed together.
(2) uses only the documented corruption process and the training snippets, and teaches the
encoder noise invariance without consuming the scarce opening->contribution supervision."""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("public")
ap.add_argument("--dim", type=int, default=192); ap.add_argument("--emb", type=int, default=48)
ap.add_argument("--layers", type=int, default=3); ap.add_argument("--kern", type=int, default=5)
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--wd", type=float, default=1e-2)
ap.add_argument("--drop", type=float, default=0.1); ap.add_argument("--den", type=float, default=0.5)
ap.add_argument("--augmin", type=float, default=0.20); ap.add_argument("--augmax", type=float, default=0.45)
ap.add_argument("--folds", type=str, default="0"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", type=str, default="cnn"); ap.add_argument("--sfb", type=int, default=1)
ap.add_argument("--pool", type=str, default="attn"); ap.add_argument("--wide", type=int, default=0); ap.add_argument("--quiet", type=int, default=0)
G = ap.parse_args()
torch.manual_seed(G.seed); np.random.seed(G.seed); torch.set_num_threads(4)

D = Data(G.public)
fold = make_folds(D.subv, 5, seed=0)
L = 100
CHARS = sorted({c for t in list(D.A) + list(D.B) for c in t})
C2I = {c: i + 1 for i, c in enumerate(CHARS)}            # 0 = pad/unknown
NV = len(CHARS) + 1
LOW = np.array([C2I[c] for c in "abcdefghijklmnopqrstuvwxyz"], dtype=np.int64)
def encode(texts):
    M = np.zeros((len(texts), L), dtype=np.int64)
    for n, t in enumerate(texts):
        for k, c in enumerate(t[:L]): M[n, k] = C2I.get(c, 0)
    return M
EA, EB = encode(list(D.A)), encode(list(D.B))
ISLOW = np.isin(np.arange(NV), LOW)                      # which codes are corruptible letters

def gpu_corrupt(M, rate, rng):
    """Apply the documented corruption directly on code arrays (fast, same process as common.corrupt)."""
    out = M.copy()
    m = (rng.random(M.shape) < rate) & ISLOW[M]
    k = int(m.sum())
    if k: out[m] = LOW[rng.integers(0, 26, k)]
    return out

class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(NV, G.emb, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, L, G.emb))
        ch = [G.emb] + [G.dim] * G.layers
        self.convs = nn.ModuleList([nn.Conv1d(ch[i], ch[i+1], G.kern, padding=G.kern//2)
                                    for i in range(G.layers)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, ch[i+1]) for i in range(G.layers)])
        self.drop = nn.Dropout(G.drop)
        # A wide final layer acts as a bank of learned, corruption-tolerant n-gram detectors.
        # Every low-dimensional bottleneck measured on this data LOSES to the sparse cosine,
        # so the pooled representation is kept deliberately high-dimensional.
        self.wide = nn.Conv1d(G.dim, G.wide, 1) if G.wide else None
        od = G.wide if G.wide else G.dim
        self.od = od
        if G.pool == "attn": self.att = nn.Conv1d(od, 1, 1)
    def forward(self, x):
        h = (self.emb(x) + self.pos).transpose(1, 2)
        for i, (cv, nm) in enumerate(zip(self.convs, self.norms)):
            y = F.gelu(nm(cv(h)))
            h = y if y.shape[1] != h.shape[1] else h + y
            h = self.drop(h)
        if self.wide is not None: h = F.gelu(self.wide(h))
        if G.pool == "attn":
            w = torch.softmax(self.att(h), dim=-1)        # learned per-position importance
            z = torch.cat([(h * w).sum(-1), h.max(-1).values], 1)
        else:
            z = torch.cat([h.mean(-1), h.max(-1).values], 1)
        return z

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Enc()
        od = self.enc.od
        self.hq = nn.Linear(2 * od, od); self.hc = nn.Linear(2 * od, od)
        self.hd = nn.Linear(2 * od, od)                   # head for the denoising objective
    def forward(self, x, which):
        z = self.enc(x)
        h = {"q": self.hq, "c": self.hc, "d": self.hd}[which](z)
        return F.normalize(h, dim=-1)

def info_nce(u, v, scale):
    lg = scale * (u @ v.T); t = torch.arange(len(u))
    return 0.5 * (F.cross_entropy(lg, t) + F.cross_entropy(lg.T, t))

def run_fold(f):
    tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
    sub_tr = D.subv[tri]; subs = np.unique(sub_tr)
    Atr, Btr = EA[tri], EB[tri]
    allsnip = np.concatenate([Atr, Btr], 0)
    net = Net()
    ls = nn.Parameter(torch.tensor(float(np.log(1 / 0.05))))
    opt = torch.optim.AdamW(list(net.parameters()) + [ls], lr=G.lr, weight_decay=G.wd)
    spe = max(len(tri) // G.bs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=G.lr, total_steps=G.epochs * spe, pct_start=0.2)
    rng = np.random.default_rng(777 + f)

    pools = build_pools(vai, D.subv, seed=100 + f)
    pos = {g: k for k, g in enumerate(vai)}
    pl = np.array([[pos[j] for j in row] for row in pools])
    def evaluate(effs=(0.30,)):
        net.eval(); res = {}
        for eff in effs:
            ms = []
            for ns in (1, 2):
                r2 = np.random.default_rng(5000 * ns + f); q_ = extra_rate(0.20, eff)
                va = gpu_corrupt(EA[vai], q_, r2); vb = gpu_corrupt(EB[vai], q_, r2)
                with torch.no_grad():
                    u = net(torch.from_numpy(va), "q").numpy(); v = net(torch.from_numpy(vb), "c").numpy()
                ms.append(metrics_from_matrix((u @ v.T)[np.arange(len(vai))[:, None], pl]))
            res[eff] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        net.train(); return res

    t0 = time.time()
    for ep in range(G.epochs):
        tot = 0.0
        for _ in range(spe):
            if G.sfb:
                s = subs[rng.integers(len(subs))]; pop = np.where(sub_tr == s)[0]
                sel = rng.choice(pop, size=min(G.bs, len(pop)), replace=False)
            else:
                sel = rng.choice(len(tri), size=G.bs, replace=False)
            ra = G.augmin + (G.augmax - G.augmin) * rng.random()
            rb = G.augmin + (G.augmax - G.augmin) * rng.random()
            xa = torch.from_numpy(gpu_corrupt(Atr[sel], extra_rate(0.20, ra), rng))
            xb = torch.from_numpy(gpu_corrupt(Btr[sel], extra_rate(0.20, rb), rng))
            sc = ls.exp().clamp(max=100.0)
            loss = info_nce(net(xa, "q"), net(xb, "c"), sc)
            if G.den > 0:                                  # denoising view-consistency on train snippets
                ds = rng.choice(len(allsnip), size=G.bs, replace=False)
                r1 = G.augmin + (G.augmax - G.augmin) * rng.random()
                r2_ = G.augmin + (G.augmax - G.augmin) * rng.random()
                v1 = torch.from_numpy(gpu_corrupt(allsnip[ds], extra_rate(0.20, r1), rng))
                v2 = torch.from_numpy(gpu_corrupt(allsnip[ds], extra_rate(0.20, r2_), rng))
                loss = loss + G.den * info_nce(net(v1, "d"), net(v2, "d"), sc)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if not G.quiet and (ep + 1) % max(G.epochs // 6, 1) == 0:
            print(f"    ep{ep+1:3d} loss {tot/spe:.4f} val@0.30 {evaluate()[0.30]['score']:.4f} ({time.time()-t0:.0f}s)")
    return evaluate((0.20, 0.25, 0.30, 0.35))

allres = [run_fold(f) for f in [int(x) for x in G.folds.split(",")]]
print(f"[{G.tag}] dim={G.dim} emb={G.emb} L={G.layers} k={G.kern} ep={G.epochs} lr={G.lr} wd={G.wd} "
      f"drop={G.drop} den={G.den} aug={G.augmin}-{G.augmax} pool={G.pool} sfb={G.sfb}")
for eff in (0.20, 0.25, 0.30, 0.35):
    sc = [r[eff]["score"] for r in allres]
    print(f"   noise {eff:.2f}: SCORE {np.mean(sc):.4f} +- {np.std(sc):.4f}   "
          f"top1 {np.mean([r[eff]['top1'] for r in allres]):.3f} "
          f"top5 {np.mean([r[eff]['top5'] for r in allres]):.3f} "
          f"mrank {np.mean([r[eff]['mean_rank'] for r in allres]):.2f}")
