"""DEV-ONLY GPU experiment (never submitted). Settles one open question: does a character
encoder trained properly from scratch add anything over the spaced-seed lexical ranker?
Reports, on identical held-out-abstract folds at the 30% evaluation noise level:
  (a) spaced-seed tf-idf cosine, (b) the trained char encoder alone, (c) their blend."""
import sys, time
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F

PUB = Path(sys.argv[1] if len(sys.argv) > 1 else "./dataset/public")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", DEV, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.0f}s] {m}", flush=True)

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
print("abstracts", len(A), "subfields", len(np.unique(sub)), flush=True)

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
def corrupt(texts, rate, rng):
    out = []
    for t in texts:
        a = np.frombuffer(t.encode("ascii","ignore"), dtype=np.uint8).copy()
        m = (rng.random(a.shape[0]) < rate) & (a >= 97) & (a <= 122)
        if m.any(): a[m] = 97 + rng.integers(0, 26, int(m.sum()))
        out.append(a.tobytes().decode("ascii"))
    return out
R30 = extra_rate(0.20, 0.30)

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
def codes(texts, L=100):
    M = np.zeros((len(texts), L), dtype=np.int64)
    for n, t in enumerate(texts):
        b = np.frombuffer(t[:L].encode("ascii","ignore"), dtype=np.uint8); M[n, :len(b)] = b
    return M
def rawfeat(M):
    n, L = M.shape; per = H//len(SEEDS); rows, cols = [], []
    for sid, off in enumerate(SEEDS):
        npos = L-off[-1]; v = np.full((n, npos), (sid+1)*2654435761, dtype=np.int64)
        for o in off: v = (v*131 + M[:, o:o+npos]) & 0x7FFFFFFFFFFF
        cols.append((sid*per + (((v*0x9E3779B1) >> 13) % per)).ravel()); rows.append(np.repeat(np.arange(n), npos))
    X = sp.coo_matrix((np.ones(sum(len(c) for c in cols), dtype=np.float32),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(n, H)).tocsr()
    X.sum_duplicates(); return X
def lex_scores(fit_texts, qa, cb, pl):
    X = rawfeat(codes(fit_texts)); df = np.asarray((X > 0).sum(0)).ravel(); nn_ = X.shape[0]
    keep = df >= 2; idf = np.zeros(H, dtype=np.float32)
    idf[keep] = np.log((1.0+nn_)/(1.0+df[keep])) + 1.0
    def tr(t):
        Y = rawfeat(codes(t)).tocsr(); Y.data = 1.0+np.log(Y.data)
        Y = Y.multiply(idf[None, :]).tocsr(); Y.eliminate_zeros()
        nr = np.sqrt(np.asarray(Y.multiply(Y).sum(1))).ravel(); nr[nr == 0] = 1.0
        return sp.diags(1.0/nr) @ Y
    return np.asarray((tr(qa) @ tr(cb).T).todense())[np.arange(len(qa))[:, None], pl]

CH = sorted({c for t in A+B for c in t}); C2I = {c: i+1 for i, c in enumerate(CH)}; NV = len(CH)+1
LOWC = np.array([C2I[c] for c in "abcdefghijklmnopqrstuvwxyz"])
ISLOW = np.isin(np.arange(NV), LOWC)
def enc(texts):
    M = np.zeros((len(texts), 100), dtype=np.int64)
    for n, t in enumerate(texts):
        for k, c in enumerate(t[:100]): M[n, k] = C2I.get(c, 0)
    return M
EA, EB = enc(A), enc(B)
def ccorrupt(M, rate, rng):
    o = M.copy(); m = (rng.random(M.shape) < rate) & ISLOW[M]; k = int(m.sum())
    if k: o[m] = LOWC[rng.integers(0, 26, k)]
    return o

class Enc(nn.Module):
    def __init__(s, emb, dim, wide, layers, kern, drop):
        super().__init__()
        s.emb = nn.Embedding(NV, emb, padding_idx=0); s.pos = nn.Parameter(torch.zeros(1, 100, emb))
        ch = [emb]+[dim]*layers
        s.cv = nn.ModuleList([nn.Conv1d(ch[i], ch[i+1], kern, padding=kern//2) for i in range(layers)])
        s.nm = nn.ModuleList([nn.GroupNorm(1, ch[i+1]) for i in range(layers)])
        s.wide = nn.Conv1d(dim, wide, 1); s.att = nn.Conv1d(wide, 1, 1); s.drop = nn.Dropout(drop); s.od = wide
    def forward(s, x):
        h = (s.emb(x)+s.pos).transpose(1, 2)
        for cv, nm in zip(s.cv, s.nm):
            y = F.gelu(nm(cv(h))); h = y if y.shape[1] != h.shape[1] else h+y; h = s.drop(h)
        h = F.gelu(s.wide(h)); w = torch.softmax(s.att(h), -1)
        return torch.cat([(h*w).sum(-1), h.max(-1).values], 1)
class Net(nn.Module):
    def __init__(s, **kw):
        super().__init__(); s.e = Enc(**kw); d = s.e.od
        s.hq = nn.Linear(2*d, d); s.hc = nn.Linear(2*d, d); s.hd = nn.Linear(2*d, d)
    def forward(s, x, w):
        z = s.e(x); return F.normalize({"q": s.hq, "c": s.hc, "d": s.hd}[w](z), dim=-1)
def nce(u, v, sc):
    lg = sc*(u@v.T); t = torch.arange(len(u), device=u.device)
    return 0.5*(F.cross_entropy(lg, t)+F.cross_entropy(lg.T, t))

def run(fold_id, steps, bs, lr, emb, dim, wide, layers, kern, drop, den, amin, amax, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    fold = make_folds(sub, 5, 0); tri = np.where(fold != fold_id)[0]; vai = np.where(fold == fold_id)[0]
    P = build_pools(vai, sub, 100+fold_id); pos = {g: k for k, g in enumerate(vai)}
    pl = np.vectorize(pos.get)(P)
    rv = np.random.default_rng(5000+fold_id)
    va_txt = corrupt([A[i] for i in vai], R30, rv); vb_txt = corrupt([B[i] for i in vai], R30, rv)
    fitt = [A[i] for i in tri]+[B[i] for i in tri]
    fitt = fitt + corrupt(fitt, R30, np.random.default_rng(5+fold_id))
    Slex = lex_scores(fitt, va_txt, vb_txt, pl)
    log(f"  fold {fold_id} lexical cosine {pool_score(Slex):.4f}")
    Atr, Btr = EA[tri], EB[tri]; allsn = np.concatenate([Atr, Btr], 0); sub_tr = sub[tri]
    subs = np.unique(sub_tr)
    net = Net(emb=emb, dim=dim, wide=wide, layers=layers, kern=kern, drop=drop).to(DEV)
    ls = nn.Parameter(torch.tensor(float(np.log(1/0.05)), device=DEV))
    opt = torch.optim.AdamW(list(net.parameters())+[ls], lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(777+fold_id)
    Eva = torch.from_numpy(enc(va_txt)).to(DEV); Evb = torch.from_numpy(enc(vb_txt)).to(DEV)
    z = lambda S: (S-S.mean(1, keepdims=True))/(S.std(1, keepdims=True)+1e-9)
    best = (9.0, 9.0, 0)
    for st in range(steps):
        s_ = subs[rng.integers(len(subs))]; pop = np.where(sub_tr == s_)[0]
        sel = rng.choice(pop, size=min(bs, len(pop)), replace=False)
        ra = amin+(amax-amin)*rng.random(); rb = amin+(amax-amin)*rng.random()
        xa = torch.from_numpy(ccorrupt(Atr[sel], extra_rate(0.20, ra), rng)).to(DEV)
        xb = torch.from_numpy(ccorrupt(Btr[sel], extra_rate(0.20, rb), rng)).to(DEV)
        sc = ls.exp().clamp(max=100.0)
        loss = nce(net(xa, "q"), net(xb, "c"), sc)
        if den > 0:
            ds = rng.choice(len(allsn), size=bs, replace=False)
            r1 = amin+(amax-amin)*rng.random(); r2 = amin+(amax-amin)*rng.random()
            v1 = torch.from_numpy(ccorrupt(allsn[ds], extra_rate(0.20, r1), rng)).to(DEV)
            v2 = torch.from_numpy(ccorrupt(allsn[ds], extra_rate(0.20, r2), rng)).to(DEV)
            loss = loss + den*nce(net(v1, "d"), net(v2, "d"), sc)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if (st+1) % 1000 == 0:
            net.eval()
            with torch.no_grad():
                u = net(Eva, "q").cpu().numpy(); v = net(Evb, "c").cpu().numpy()
            net.train()
            Snn = (u@v.T)[np.arange(len(vai))[:, None], pl]
            sn = pool_score(Snn)
            bl = min(pool_score(z(Slex)+w*z(Snn)) for w in (0.25, 0.5, 0.75, 1.0))
            if sn < best[0]: best = (sn, bl, st+1)
            log(f"    step {st+1:6d} cnn {sn:.4f} blend {bl:.4f} (lex {pool_score(Slex):.4f})")
    return best

CFGS = [
    dict(steps=12000, bs=256, lr=2e-3, emb=48, dim=192, wide=768, layers=3, kern=5, drop=0.1, den=0.5, amin=0.20, amax=0.45),
    dict(steps=12000, bs=256, lr=2e-3, emb=48, dim=256, wide=2048, layers=4, kern=5, drop=0.1, den=0.5, amin=0.20, amax=0.45),
    dict(steps=12000, bs=256, lr=3e-3, emb=48, dim=192, wide=768, layers=3, kern=7, drop=0.2, den=1.0, amin=0.20, amax=0.50),
]
for i, c in enumerate(CFGS):
    log(f"CONFIG {i}: {c}")
    b = run(0, **c)
    log(f"CONFIG {i} RESULT: cnn {b[0]:.4f} blend {b[1]:.4f} at step {b[2]}")
