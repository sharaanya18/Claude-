"""Model C: LISTWISE neural ranker over the pair evidence profile.
The 20 candidates of a query are scored jointly and trained with softmax cross-entropy on the
true continuation -- the loss the official metric actually cares about. Each candidate is
described by its evidence profile and by the SAME profile standardised within its own pool
(pool-relative evidence), which is information the query itself supplies. No statistic ever
crosses from one query to another, and nothing is fitted on evaluation text."""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from evidence import Evidence

ap = argparse.ArgumentParser()
ap.add_argument("public"); ap.add_argument("--hid", type=int, default=128)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--wd", type=float, default=1e-3)
ap.add_argument("--drop", type=float, default=0.2); ap.add_argument("--naug", type=int, default=4)
ap.add_argument("--npool", type=int, default=3); ap.add_argument("--augmin", type=float, default=0.20)
ap.add_argument("--augmax", type=float, default=0.40); ap.add_argument("--folds", type=str, default="0")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", type=str, default="rank")
ap.add_argument("--spec", type=str, default="w2:2-8,w3:3-8,w4:4-8")
ap.add_argument("--skel", type=str, default="w3:3-6,w4:4-7"); ap.add_argument("--nbuck", type=int, default=6)
ap.add_argument("--setattn", type=int, default=1); ap.add_argument("--quiet", type=int, default=0)
G = ap.parse_args()
torch.manual_seed(G.seed); np.random.seed(G.seed); torch.set_num_threads(4)
D = Data(G.public); fold = make_folds(D.subv, 5, seed=0)
A, B = np.array(D.A, dtype=object), np.array(D.B, dtype=object)

def train_pools(idx, subv, nneg=19, seed=0):
    rng = np.random.default_rng(seed)
    bysub = {s: idx[subv[idx] == s] for s in np.unique(subv[idx])}
    P = np.empty((len(idx), nneg + 1), dtype=np.int64)
    for r, i in enumerate(idx):
        pop = bysub[subv[i]]; pop = pop[pop != i]
        P[r, 0] = i; P[r, 1:] = rng.choice(pop, size=nneg, replace=False)
    return P

class Ranker(nn.Module):
    """Scores all 20 candidates of a pool together; optional set-attention lets a candidate
    be judged against the rest of its own pool before the final score."""
    def __init__(self, nin, hid, drop, setattn):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nin, hid), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Dropout(drop))
        self.setattn = setattn
        if setattn:
            self.att = nn.MultiheadAttention(hid, 4, batch_first=True, dropout=drop)
            self.ln = nn.LayerNorm(hid)
        self.out = nn.Linear(hid, 1)
    def forward(self, x):                      # x: (Q, P, nin)
        h = self.inp(x)
        if self.setattn:
            h = self.ln(h + self.att(h, h, h, need_weights=False)[0])
        return self.out(h).squeeze(-1)

def featurise(Fp, mu, sd):
    """log-compress, standardise with TRAIN statistics, and append pool-relative versions."""
    z = np.log1p(np.abs(Fp) * 100.0) * np.sign(Fp)
    zs = (z - mu) / sd
    pm = z.mean(1, keepdims=True); ps = z.std(1, keepdims=True) + 1e-6
    return np.concatenate([zs, (z - pm) / ps], -1).astype(np.float32)

def run_fold(f):
    tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
    rs = np.random.default_rng(5 + f); R30 = extra_rate(0.20, 0.30)
    fit = list(A[tri]) + list(B[tri]); fit = fit + corrupt(fit, R30, rs)
    ev = Evidence(G.spec, G.skel, nbuck=G.nbuck).fit(fit)
    t0 = time.time()
    # training views: fresh corruption x fresh same-subfield pools, all from the train fold
    pos_tr = {g: k for k, g in enumerate(tri)}
    banks = []
    for a in range(G.naug):
        p = G.augmin + (G.augmax - G.augmin) * a / max(G.naug - 1, 1)
        q = extra_rate(0.20, p)
        ta = corrupt(list(A[tri]), q, rs) if q > 0 else list(A[tri])
        tb = corrupt(list(B[tri]), q, rs) if q > 0 else list(B[tri])
        Xq, Xc = ev.vectors(ta), ev.vectors(tb)
        for s in range(G.npool):
            P = train_pools(tri, D.subv, seed=900 + 37 * s + f)
            banks.append(ev.profiles(Xq, Xc, np.vectorize(pos_tr.get)(P)))
    Fall = np.concatenate(banks, 0)
    zz = np.log1p(np.abs(Fall) * 100.0) * np.sign(Fall)
    mu, sd = zz.reshape(-1, zz.shape[-1]).mean(0), zz.reshape(-1, zz.shape[-1]).std(0) + 1e-6
    Xtr = np.concatenate([featurise(b, mu, sd) for b in banks], 0)
    if not G.quiet: print(f"  fold {f}: built {Xtr.shape} in {time.time()-t0:.0f}s")

    vpools = build_pools(vai, D.subv, seed=100 + f)
    pos_va = {g: k for k, g in enumerate(vai)}
    pl = np.vectorize(pos_va.get)(vpools)
    vsets = []
    for ns in (1, 2):
        r2 = np.random.default_rng(5000 * ns + f)
        va = corrupt(list(A[vai]), R30, r2); vb = corrupt(list(B[vai]), R30, r2)
        vsets.append(featurise(ev.profiles(ev.vectors(va), ev.vectors(vb), pl), mu, sd))

    net = Ranker(Xtr.shape[-1], G.hid, G.drop, G.setattn)
    opt = torch.optim.AdamW(net.parameters(), lr=G.lr, weight_decay=G.wd)
    spe = max(len(Xtr) // G.bs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=G.lr, total_steps=G.epochs * spe, pct_start=0.2)
    Xt = torch.from_numpy(Xtr); rng = np.random.default_rng(31 + f)
    def ev_score():
        net.eval(); out = []
        with torch.no_grad():
            for V in vsets:
                out.append(metrics_from_matrix(net(torch.from_numpy(V)).numpy()))
        net.train(); return {k: float(np.mean([o[k] for o in out])) for k in out[0]}
    for ep in range(G.epochs):
        tot = 0.0
        for _ in range(spe):
            sel = torch.from_numpy(rng.choice(len(Xt), size=G.bs, replace=False))
            s = net(Xt[sel])
            loss = F.cross_entropy(s, torch.zeros(len(sel), dtype=torch.long))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if not G.quiet and (ep + 1) % max(G.epochs // 5, 1) == 0:
            print(f"    ep{ep+1:3d} loss {tot/spe:.4f} val@0.30 {ev_score()['score']:.4f}")
    return ev_score()

res = [run_fold(f) for f in [int(x) for x in G.folds.split(",")]]
print(f"[{G.tag}] hid={G.hid} ep={G.epochs} lr={G.lr} wd={G.wd} drop={G.drop} aug={G.naug}x pool={G.npool} "
      f"setattn={G.setattn} nbuck={G.nbuck} spec={G.spec} skel={G.skel}")
print(f"   @0.30 SCORE {np.mean([r['score'] for r in res]):.4f} +- {np.std([r['score'] for r in res]):.4f}  "
      f"top1 {np.mean([r['top1'] for r in res]):.3f} top3 {np.mean([r['top3'] for r in res]):.3f} "
      f"top5 {np.mean([r['top5'] for r in res]):.3f} mrank {np.mean([r['mean_rank'] for r in res]):.2f}")
