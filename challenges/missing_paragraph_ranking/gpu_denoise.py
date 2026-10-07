"""DEV-ONLY GPU experiment (never submitted).
Idea: spaced seeds make matching tolerant of corruption, but they never try to UNDO it.
A character sequence carries a lot of context - the surrounding characters strongly constrain
what a corrupted letter originally was. So train, from random initialisation on the training
snippets alone, a character denoiser: feed it a snippet pushed to the evaluation noise level
(30%) and ask it to reproduce the same snippet at its own training noise level (20%). The
extra corruption is generated with the challenge's documented process, so every (input,target)
pair is made from real training text - no synthetic data, no evaluation text, no pretraining.

Then measure whether matching DENOISED text beats matching raw text, on held-out abstracts at
the evaluation noise level. The measured noise curve says 20% vs 30% is worth ~0.045 of score,
so a denoiser that recovers even part of that is the largest remaining lever."""
import sys, time
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F

PUB = Path(sys.argv[1] if len(sys.argv) > 1 else "./dataset/public")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.0f}s] {m}", flush=True)
log(f"device {DEV} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

train = pd.read_csv(PUB/"train.csv", keep_default_na=False)
cand = pd.read_csv(PUB/"candidates.csv", keep_default_na=False)
lab = pd.read_csv(PUB/"train_labels.csv", keep_default_na=False)
cmap = dict(zip(cand.candidate_id, cand.snippet_b)); tgt = dict(zip(lab.query_id, lab.target_id))
pools_tr = {q: c.split() for q, c in zip(train.query_id, train.candidates)}
A = list(train.snippet_a); B = [cmap[tgt[q]] for q in train.query_id]
cands = sorted({c for v in pools_tr.values() for c in v}); idx = {c: i for i, c in enumerate(cands)}
par = list(range(len(cands)))
def find(a):
    while par[a] != a: par[a] = par[par[a]]; a = par[a]
    return a
for pool in pools_tr.values():
    ii = [idx[c] for c in pool]
    for j in ii[1:]:
        ra, rb = find(ii[0]), find(j)
        if ra != rb: par[ra] = rb
roots = {}
for c in cands: roots.setdefault(find(idx[c]), len(roots))
sub = np.array([roots[find(idx[tgt[q]])] for q in train.query_id])
log(f"{len(A)} abstracts, {len(np.unique(sub))} subfields")

def make_folds(sv, k, seed=0):
    rng = np.random.default_rng(seed); f = np.empty(len(sv), dtype=int)
    for s in np.unique(sv):
        i = np.where(sv == s)[0]; i = i[rng.permutation(len(i))]; f[i] = np.arange(len(i)) % k
    return f
def build_pools(vi, sv, seed):
    rng = np.random.default_rng(seed); by = {s: vi[sv[vi] == s] for s in np.unique(sv[vi])}
    P = np.empty((len(vi), 20), dtype=np.int64)
    for r, i in enumerate(vi):
        pop = by[sv[i]]; pop = pop[pop != i]
        P[r, 0] = i; P[r, 1:] = rng.choice(pop, 19, replace=False)
    return P
def pool_score(S):
    s0 = S[:, :1]
    return float(((S[:, 1:] > s0).sum(1) + 0.5*(S[:, 1:] == s0).sum(1)).mean()/19.0)
def extra_rate(a, b): return 1.0 - (1.0-b)/(1.0-a)
R30 = extra_rate(0.20, 0.30)

CH = sorted({c for t in A+B for c in t}); C2I = {c: i+1 for i, c in enumerate(CH)}
I2C = {v: k for k, v in C2I.items()}; NV = len(CH)+1
LOWC = np.array([C2I[c] for c in "abcdefghijklmnopqrstuvwxyz"]); ISLOW = np.isin(np.arange(NV), LOWC)
def enc(texts):
    M = np.zeros((len(texts), 100), dtype=np.int64)
    for n, t in enumerate(texts):
        for k, c in enumerate(t[:100]): M[n, k] = C2I.get(c, 0)
    return M
def dec(M): return ["".join(I2C.get(int(c), " ") for c in row) for row in M]
def ccorrupt(M, rate, rng):
    o = M.copy(); m = (rng.random(M.shape) < rate) & ISLOW[M]; k = int(m.sum())
    if k: o[m] = LOWC[rng.integers(0, 26, k)]
    return o
EA, EB = enc(A), enc(B)

# ---------------- lexical scorer (spaced seeds), used to judge raw vs denoised text
def masks(w, W):
    if w == W: return [tuple(range(W))]
    return [(0,)+tuple(m)+(W-1,) for m in combinations(range(1, W-1), w-2)]
def seed_lib(spec):
    out = []
    for part in spec.split(","):
        w, r = part.split(":"); w = int(w[1:]); lo, hi = (int(x) for x in r.split("-"))
        for W in range(lo, hi+1): out += masks(w, W)
    return out
SEEDS = seed_lib("w2:2-8,w3:3-8,w4:4-8"); H = 1 << 22
def rawfeat(M):
    n, L = M.shape; per = H//len(SEEDS); rows, cols = [], []
    for sid, off in enumerate(SEEDS):
        npos = L-off[-1]; v = np.full((n, npos), (sid+1)*2654435761, dtype=np.int64)
        for o in off: v = (v*131 + M[:, o:o+npos]) & 0x7FFFFFFFFFFF
        cols.append((sid*per + (((v*0x9E3779B1) >> 13) % per)).ravel()); rows.append(np.repeat(np.arange(n), npos))
    X = sp.coo_matrix((np.ones(sum(len(c) for c in cols), dtype=np.float32),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(n, H)).tocsr()
    X.sum_duplicates(); return X
def lex_score(fitM, qaM, cbM, pl):
    X = rawfeat(fitM); df = np.asarray((X > 0).sum(0)).ravel(); nn_ = X.shape[0]
    keep = df >= 2; idf = np.zeros(H, dtype=np.float32)
    idf[keep] = np.log((1.0+nn_)/(1.0+df[keep])) + 1.0
    def tr(M):
        Y = rawfeat(M).tocsr(); Y.data = 1.0+np.log(Y.data)
        Y = Y.multiply(idf[None, :]).tocsr(); Y.eliminate_zeros()
        nr = np.sqrt(np.asarray(Y.multiply(Y).sum(1))).ravel(); nr[nr == 0] = 1.0
        return sp.diags(1.0/nr) @ Y
    return np.asarray((tr(qaM) @ tr(cbM).T).todense())[np.arange(qaM.shape[0])[:, None], pl]

# ---------------- the denoiser: a from-scratch character transformer
class Denoiser(nn.Module):
    def __init__(s, d=256, layers=6, heads=8, ff=1024, drop=0.1):
        super().__init__()
        s.emb = nn.Embedding(NV, d); s.pos = nn.Parameter(torch.zeros(1, 100, d))
        nn.init.normal_(s.pos, std=0.02)
        L = nn.TransformerEncoderLayer(d, heads, ff, dropout=drop, batch_first=True,
                                       norm_first=True, activation="gelu")
        s.tr = nn.TransformerEncoder(L, layers)
        s.ln = nn.LayerNorm(d); s.out = nn.Linear(d, NV)
    def forward(s, x):
        return s.out(s.ln(s.tr(s.emb(x) + s.pos)))

def train_denoiser(tri, steps, bs, lr, d, layers, heads, ff, drop, in_noise, seed=0):
    """Input: a training snippet pushed to `in_noise`. Target: the same snippet at its own
    20% level. The model learns to undo the extra corruption from context alone."""
    torch.manual_seed(seed)
    data = np.concatenate([EA[tri], EB[tri]], 0)
    net = Denoiser(d, layers, heads, ff, drop).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(1234 + seed)
    net.train()
    for st in range(steps):
        sel = rng.choice(len(data), size=bs, replace=False)
        y = data[sel]
        q = extra_rate(0.20, in_noise[0] + (in_noise[1] - in_noise[0]) * rng.random())
        x = ccorrupt(y, q, rng)
        xb = torch.from_numpy(x).to(DEV); yb = torch.from_numpy(y).to(DEV)
        loss = F.cross_entropy(net(xb).reshape(-1, NV), yb.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sch.step()
        if (st + 1) % 2000 == 0: log(f"    denoise step {st+1} loss {loss.item():.4f}")
    net.eval(); return net

@torch.no_grad()
def denoise(net, M, bs=512):
    out = np.zeros_like(M)
    for i in range(0, len(M), bs):
        xb = torch.from_numpy(M[i:i+bs]).to(DEV)
        out[i:i+bs] = net(xb).argmax(-1).cpu().numpy()
    return out

def char_acc(pred, truth, mask):
    return float((pred[mask] == truth[mask]).mean())

def run(fold_id, cfg):
    fold = make_folds(sub, 5, 0); tri = np.where(fold != fold_id)[0]; vai = np.where(fold == fold_id)[0]
    P = build_pools(vai, sub, 100+fold_id); pos = {g: k for k, g in enumerate(vai)}
    pl = np.vectorize(pos.get)(P)
    rv = np.random.default_rng(5000+fold_id)
    VA = ccorrupt(EA[vai], R30, rv); VB = ccorrupt(EB[vai], R30, rv)
    rf = np.random.default_rng(5+fold_id)
    FIT = np.concatenate([EA[tri], EB[tri]], 0)
    FIT = np.concatenate([FIT, ccorrupt(FIT, R30, rf)], 0)
    base = pool_score(lex_score(FIT, VA, VB, pl))
    log(f"  fold {fold_id}: RAW text score {base:.4f}")
    net = train_denoiser(tri, **cfg)
    DA, DB = denoise(net, VA), denoise(net, VB)
    letters = ISLOW[EA[vai]]
    log(f"  recovery on held-out openings: observed {char_acc(VA, EA[vai], letters):.3f} "
        f"-> denoised {char_acc(DA, EA[vai], letters):.3f} (of the 20%-noise text)")
    DFIT = denoise(net, FIT)
    for name, fitM, qa, cb in (("denoised", DFIT, DA, DB),
                               ("denoised-fit-raw", FIT, DA, DB)):
        log(f"  fold {fold_id}: {name} score {pool_score(lex_score(fitM, qa, cb, pl)):.4f}")
    return base

CFG = dict(steps=20000, bs=256, lr=3e-4, d=256, layers=6, heads=8, ff=1024, drop=0.1,
           in_noise=(0.28, 0.34))
for f in (0, 1):
    run(f, CFG)
