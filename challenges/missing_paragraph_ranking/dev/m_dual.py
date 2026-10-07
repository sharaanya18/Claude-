"""Model A: trained sparse->dense DUAL ENCODER.
Character n-gram counts (vocabulary + idf fitted on the TRAINING FOLD ONLY) are projected by
two LEARNED towers (opening side / contribution side) into a shared space. Both towers start
from the SAME random projection, so at step 0 the score is an (undistorted) random sketch of
the raw cosine; every improvement past that point is learned from the training pairs.
Trained with listwise softmax (InfoNCE) against SAME-SUBFIELD in-batch negatives, with the
challenge's own letter corruption applied as augmentation across and beyond the eval level."""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from sklearn.feature_extraction.text import TfidfVectorizer

ap = argparse.ArgumentParser()
ap.add_argument("public"); ap.add_argument("--dim", type=int, default=256)
ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--wd", type=float, default=1e-5)
ap.add_argument("--naug", type=int, default=8); ap.add_argument("--augmax", type=float, default=0.45)
ap.add_argument("--augmin", type=float, default=0.20); ap.add_argument("--maxnnz", type=int, default=640)
ap.add_argument("--ng", type=str, default="2,5"); ap.add_argument("--mindf", type=int, default=3)
ap.add_argument("--analyzer", type=str, default="char_wb"); ap.add_argument("--tied", type=int, default=1)
ap.add_argument("--folds", type=str, default="0"); ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", type=str, default="dual")
ap.add_argument("--sfb", type=int, default=1); ap.add_argument("--head", type=int, default=1)
ap.add_argument("--quiet", type=int, default=0)
G = ap.parse_args()
torch.manual_seed(G.seed); np.random.seed(G.seed); torch.set_num_threads(4)

D = Data(G.public)
fold = make_folds(D.subv, 5, seed=0)
A, B = np.array(D.A, dtype=object), np.array(D.B, dtype=object)
NG = tuple(int(x) for x in G.ng.split(","))

def pad_bag(X, maxnnz):
    """csr -> dense (N, maxnnz) index/weight tensors; index 0 is a zero-weight pad slot."""
    X = X.tocsr(); N = X.shape[0]
    I = np.zeros((N, maxnnz), dtype=np.int64); W = np.zeros((N, maxnnz), dtype=np.float32)
    for r in range(N):
        s, e = X.indptr[r], X.indptr[r + 1]
        k = min(e - s, maxnnz)
        if k:
            d = X.data[s:e]
            if e - s > maxnnz:                      # keep the heaviest terms if ever truncated
                o = np.argsort(-d)[:maxnnz]; I[r, :k] = X.indices[s:e][o] + 1; W[r, :k] = d[o]
            else:
                I[r, :k] = X.indices[s:e] + 1; W[r, :k] = d
    return torch.from_numpy(I), torch.from_numpy(W)

class Tower(nn.Module):
    def __init__(self, emb, d, drop, head):
        super().__init__()
        self.emb = emb; self.drop = nn.Dropout(drop); self.head = head
        if head:
            self.ln = nn.LayerNorm(d); self.fc = nn.Linear(d, d)
            nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)   # starts as identity
    def forward(self, I, W):
        h = (self.emb(I) * W.unsqueeze(-1)).sum(1)
        h = self.drop(h)
        if self.head:
            h = h + self.fc(F.gelu(self.ln(h)))
        return F.normalize(h, dim=-1)

def run_fold(f):
    tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
    rs = np.random.default_rng(1234 + f)
    fit_txt = list(A[tri]) + list(B[tri])
    fit_txt += corrupt(fit_txt, extra_rate(0.20, 0.30), np.random.default_rng(99 + f))
    vec = TfidfVectorizer(analyzer=G.analyzer, ngram_range=NG, min_df=G.mindf,
                          sublinear_tf=True, dtype=np.float32).fit(fit_txt)
    V = len(vec.vocabulary_)
    def feat(txts): return pad_bag(vec.transform(txts), G.maxnnz)
    augA, augB = [], []
    for k in range(G.naug):
        p = G.augmin + (G.augmax - G.augmin) * k / max(G.naug - 1, 1)
        q = extra_rate(0.20, p)
        augA.append(feat(corrupt(list(A[tri]), q, rs) if q > 0 else list(A[tri])))
        augB.append(feat(corrupt(list(B[tri]), q, rs) if q > 0 else list(B[tri])))
    sub_tr = D.subv[tri]
    gen = torch.Generator().manual_seed(G.seed)
    base = nn.Embedding(V + 1, G.dim, padding_idx=0)
    with torch.no_grad():
        base.weight.normal_(0, 1.0 / np.sqrt(G.dim), generator=gen); base.weight[0].zero_()
    embq = base
    embc = base if G.tied else nn.Embedding.from_pretrained(base.weight.clone(), freeze=False, padding_idx=0)
    model = nn.ModuleDict({"q": Tower(embq, G.dim, G.drop, G.head),
                           "c": Tower(embc, G.dim, G.drop, G.head)})
    logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.05))))
    opt = torch.optim.AdamW(list(model.parameters()) + [logit_scale], lr=G.lr, weight_decay=G.wd)
    spe = max(len(tri) // G.bs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=G.lr, total_steps=G.epochs * spe, pct_start=0.15)
    rng = np.random.default_rng(777 + f); subs = np.unique(sub_tr)

    pools = build_pools(vai, D.subv, seed=100 + f)
    pos = {g: k for k, g in enumerate(vai)}
    pl = np.array([[pos[j] for j in row] for row in pools])
    def evaluate(effs=(0.30,)):
        model.eval(); res = {}
        for eff in effs:
            ms = []
            for ns in (1, 2):
                r2 = np.random.default_rng(5000 * ns + f); q_ = extra_rate(0.20, eff)
                va = corrupt(list(A[vai]), q_, r2) if q_ > 0 else list(A[vai])
                vb = corrupt(list(B[vai]), q_, r2) if q_ > 0 else list(B[vai])
                with torch.no_grad():
                    u = model["q"](*feat(va)).numpy(); v = model["c"](*feat(vb)).numpy()
                ms.append(metrics_from_matrix((u @ v.T)[np.arange(len(vai))[:, None], pl]))
            res[eff] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        model.train(); return res

    if not G.quiet:
        print(f"  fold {f} V={V} init(JL sketch of raw cosine) @0.30={evaluate()[0.30]['score']:.4f}")
    t0 = time.time()
    for ep in range(G.epochs):
        tot = 0.0
        for _ in range(spe):
            if G.sfb:
                s = subs[rng.integers(len(subs))]; pop = np.where(sub_tr == s)[0]
                sel = rng.choice(pop, size=min(G.bs, len(pop)), replace=False)
            else:
                sel = rng.choice(len(tri), size=G.bs, replace=False)
            t = torch.from_numpy(sel)
            ka, kb = rng.integers(G.naug), rng.integers(G.naug)
            u = model["q"](augA[ka][0][t], augA[ka][1][t])
            v = model["c"](augB[kb][0][t], augB[kb][1][t])
            lg = logit_scale.exp().clamp(max=100.0) * (u @ v.T)
            tg = torch.arange(len(sel))
            loss = 0.5 * (F.cross_entropy(lg, tg) + F.cross_entropy(lg.T, tg))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if not G.quiet and (ep + 1) % max(G.epochs // 5, 1) == 0:
            print(f"    ep{ep+1:3d} loss {tot/spe:.4f} val@0.30 {evaluate()[0.30]['score']:.4f} "
                  f"({time.time()-t0:.0f}s)")
    return evaluate((0.20, 0.25, 0.30, 0.35))

allres = [run_fold(f) for f in [int(x) for x in G.folds.split(",")]]
print(f"[{G.tag}] dim={G.dim} ep={G.epochs} lr={G.lr} ng={G.ng} {G.analyzer} mindf={G.mindf} "
      f"tied={G.tied} head={G.head} aug={G.augmin}-{G.augmax}x{G.naug} sfb={G.sfb} drop={G.drop} wd={G.wd}")
for eff in (0.20, 0.25, 0.30, 0.35):
    sc = [r[eff]["score"] for r in allres]
    print(f"   noise {eff:.2f}: SCORE {np.mean(sc):.4f} +- {np.std(sc):.4f}   "
          f"top1 {np.mean([r[eff]['top1'] for r in allres]):.3f} "
          f"top5 {np.mean([r[eff]['top5'] for r in allres]):.3f} "
          f"mrank {np.mean([r[eff]['mean_rank'] for r in allres]):.2f}")
