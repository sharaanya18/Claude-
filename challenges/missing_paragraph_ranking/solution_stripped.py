import os, sys, time, random
PUBLIC_DIR = sys.argv[1] if len(sys.argv) > 1 else './dataset/public'
SUBMISSION_OUT = sys.argv[2] if len(sys.argv) > 2 else './working/submission.csv'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
PUBLIC_DIR = Path(PUBLIC_DIR)
SUBMISSION_OUT = Path(SUBMISSION_OUT)
SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)
SEED = 42
DEVICE = torch.device('cpu')
POOL = 20
N_FOLDS = 5
N_SEEDS = 3
EPOCHS = 60
EVAL_EVERY = 3
BATCH = 64
LR = 0.002
WD = 0.01
HID = 48
DEPTH = 2
DROP = 0.3
N_AUG = 4
N_POOLSAMP = 3
AUG_MIN = 0.2
AUG_MAX = 0.42
TRAIN_NOISE = 0.2
EVAL_NOISE = 0.3
SEED_SPEC = 'w2:2-8,w3:3-8,w4:4-8'
SKEL_SPEC = 'w3:3-6,w4:4-7'
NBITS = 22
MIN_DF = 2
N_RARE = 3
T0 = time.time()

def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
LOWER = np.arange(97, 123, dtype=np.uint8)

def corrupt(texts, rate, rng):
    out = []
    for t in texts:
        a = np.frombuffer(t.encode('ascii', 'ignore'), dtype=np.uint8).copy()
        hit = rng.random(a.shape[0]) < rate
        lo = hit & (a >= 97) & (a <= 122)
        up = hit & (a >= 65) & (a <= 90)
        if lo.any():
            a[lo] = 97 + rng.integers(0, 26, int(lo.sum()))
        if up.any():
            a[up] = 65 + rng.integers(0, 26, int(up.sum()))
        out.append(a.tobytes().decode('ascii'))
    return out

def extra_rate(p_from, p_to):
    return 1.0 - (1.0 - p_to) / (1.0 - p_from)

def _masks(w, W):
    from itertools import combinations
    if w == W:
        return [tuple(range(W))]
    if w < 2 or w > W:
        return []
    return [(0,) + tuple(m) + (W - 1,) for m in combinations(range(1, W - 1), w - 2)]

def seed_lib(spec):
    out, seen = ([], set())
    for part in spec.split(','):
        w, rng_ = part.split(':')
        w = int(w[1:])
        lo, hi = (int(x) for x in rng_.split('-'))
        for W in range(lo, hi + 1):
            for m in _masks(w, W):
                if m not in seen:
                    seen.add(m)
                    out.append(m)
    return out

def codes(texts, L=100):
    M = np.zeros((len(texts), L), dtype=np.int64)
    for n, t in enumerate(texts):
        b = np.frombuffer(t[:L].encode('ascii', 'ignore'), dtype=np.uint8)
        M[n, :len(b)] = b
    return M

def skeletonise(M):
    out = M.copy()
    out[(M >= 97) & (M <= 122)] = 1
    return out

class PatternBank:

    def __init__(self, spec, skel_spec, nbits, min_df, n_rare):
        self.seeds = seed_lib(spec)
        self.skseeds = seed_lib(skel_spec)
        self.H = 1 << nbits
        self.min_df = min_df
        self.n_rare = n_rare
        self.fams = sorted({len(s) for s in self.seeds})

    def _raw(self, M, seeds):
        N, L = M.shape
        per = self.H // len(seeds)
        rows, cols = ([], [])
        for sid, off in enumerate(seeds):
            npos = L - off[-1]
            v = np.full((N, npos), (sid + 1) * 2654435761, dtype=np.int64)
            for o in off:
                v = v * 131 + M[:, o:o + npos] & 140737488355327
            cols.append((sid * per + (v * 2654435761 >> 13) % per).ravel())
            rows.append(np.repeat(np.arange(N), npos))
        X = sp.coo_matrix((np.ones(sum((len(c) for c in cols)), dtype=np.float32), (np.concatenate(rows), np.concatenate(cols))), shape=(N, self.H)).tocsr()
        X.sum_duplicates()
        return X

    def fit(self, texts):
        M = codes(texts)
        for key, seeds, Mx in (('L', self.seeds, M), ('S', self.skseeds, skeletonise(M))):
            X = self._raw(Mx, seeds)
            df = np.asarray((X > 0).sum(0)).ravel()
            n = X.shape[0]
            keep = df >= self.min_df
            idf = np.zeros(self.H, dtype=np.float32)
            idf[keep] = np.log((1.0 + n) / (1.0 + df[keep])) + 1.0
            setattr(self, 'idf_' + key, idf)
            if key == 'L':
                per = self.H // len(seeds)
                fam = np.full(self.H, -1, dtype=np.int8)
                fmap = {w: i for i, w in enumerate(self.fams)}
                for sid, off in enumerate(seeds):
                    fam[sid * per:(sid + 1) * per] = fmap[len(off)]
                self.fam = fam
                q = np.quantile(idf[keep], np.linspace(0, 1, self.n_rare + 1)[1:-1])
                self.rare = np.where(keep, np.digitize(idf, q), -1).astype(np.int8)
        return self

    def _vec(self, texts, key, seeds):
        M = codes(texts)
        X = self._raw(skeletonise(M) if key == 'S' else M, seeds).tocsr()
        X.data = 1.0 + np.log(X.data)
        X = X.multiply(getattr(self, 'idf_' + key)[None, :]).tocsr()
        X.eliminate_zeros()
        return X

    def vectors(self, texts):
        XL = self._vec(texts, 'L', self.seeds)
        nf = len(self.fams)
        fn = np.zeros((XL.shape[0], nf + 1), dtype=np.float32)
        for r in range(XL.shape[0]):
            s, e = (XL.indptr[r], XL.indptr[r + 1])
            d2 = XL.data[s:e] ** 2
            fn[r, :nf] = np.bincount(self.fam[XL.indices[s:e]], weights=d2, minlength=nf)[:nf]
            fn[r, nf] = d2.sum()
        fn = np.sqrt(fn)
        fn[fn == 0] = 1.0
        XS = self._vec(texts, 'S', self.skseeds)
        sn = np.sqrt(np.asarray(XS.multiply(XS).sum(1))).ravel()
        sn[sn == 0] = 1.0
        return dict(L=XL, fn=fn, S=XS, sn=sn)
FEATURE_NAMES = None

def profiles(bank, Vq, Vc, pools):
    global FEATURE_NAMES
    nf = len(bank.fams)
    names = ['cos_all'] + ['cos_w%d' % w for w in bank.fams] + ['cos_struct'] + ['rare%d' % b for b in range(bank.n_rare)] + ['n_agree', 'max_agree', 'top5_share']
    FEATURE_NAMES = names
    Q, P = pools.shape
    Fo = np.zeros((Q, P, len(names)), dtype=np.float32)
    A, Bm, An, Bn = (Vq['L'], Vc['L'], Vq['fn'], Vc['fn'])
    buf = np.zeros(bank.H, dtype=np.float32)
    for q in range(Q):
        s, e = (A.indptr[q], A.indptr[q + 1])
        ai = A.indices[s:e]
        buf[ai] = A.data[s:e]
        for p in range(P):
            r = pools[q, p]
            s2, e2 = (Bm.indptr[r], Bm.indptr[r + 1])
            bi = Bm.indices[s2:e2]
            prod = buf[bi] * Bm.data[s2:e2]
            nz = prod > 0
            if not nz.any():
                continue
            pv, idxs = (prod[nz], bi[nz])
            denom = An[q, nf] * Bn[r, nf]
            tot = pv.sum()
            Fo[q, p, 0] = tot / denom
            Fo[q, p, 1:1 + nf] = np.bincount(bank.fam[idxs], weights=pv, minlength=nf)[:nf] / (An[q, :nf] * Bn[r, :nf])
            bk = bank.rare[idxs]
            ok = bk >= 0
            if ok.any():
                Fo[q, p, 2 + nf:2 + nf + bank.n_rare] = np.bincount(bk[ok], weights=pv[ok], minlength=bank.n_rare)[:bank.n_rare] / denom
            k = 2 + nf + bank.n_rare
            Fo[q, p, k] = len(pv)
            Fo[q, p, k + 1] = pv.max() / denom
            Fo[q, p, k + 2] = np.sort(pv)[-5:].sum() / max(tot, 1e-09)
        buf[ai] = 0.0
    XSq, XSc, snq, snc = (Vq['S'], Vc['S'], Vq['sn'], Vc['sn'])
    for q in range(Q):
        s, e = (XSq.indptr[q], XSq.indptr[q + 1])
        ai = XSq.indices[s:e]
        buf[ai] = XSq.data[s:e]
        for p in range(P):
            r = pools[q, p]
            s2, e2 = (XSc.indptr[r], XSc.indptr[r + 1])
            bi = XSc.indices[s2:e2]
            Fo[q, p, 1 + nf] = (buf[bi] * XSc.data[s2:e2]).sum() / (snq[q] * snc[r])
        buf[ai] = 0.0
    return Fo

def featurise(Fp, mu, sd):
    z = np.log1p(np.abs(Fp) * 100.0) * np.sign(Fp)
    pm = z.mean(-2, keepdims=True)
    ps = z.std(-2, keepdims=True) + 1e-06
    a = z[..., :, None, :]
    b = z[..., None, :, :]
    o = ((a < b).sum(-2) + 0.5 * (a == b).sum(-2) - 0.5) / (z.shape[-2] - 1.0) - 0.5
    return np.concatenate([(z - mu) / sd, (z - pm) / ps, o.astype(np.float32)], -1).astype(np.float32)

class ListwiseRanker(nn.Module):

    def __init__(self, n_in, hid=HID, drop=DROP, depth=DEPTH):
        super().__init__()
        layers, d = ([], n_in)
        for _ in range(depth):
            layers += [nn.Linear(d, hid), nn.GELU(), nn.Dropout(drop)]
            d = hid
        self.body = nn.Sequential(*layers)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True, dropout=drop)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        h = self.body(x)
        h = self.norm(h + self.attn(h, h, h, need_weights=False)[0])
        return self.head(h).squeeze(-1)

def subfields_from_train_pools(pools, target_of):
    cands = sorted({c for v in pools.values() for c in v})
    idx = {c: i for i, c in enumerate(cands)}
    par = list(range(len(cands)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for pool in pools.values():
        ids = [idx[c] for c in pool]
        for j in ids[1:]:
            ra, rb = (find(ids[0]), find(j))
            if ra != rb:
                par[ra] = rb
    roots = {}
    for c in cands:
        roots.setdefault(find(idx[c]), len(roots))
    sub_of_cand = {c: roots[find(idx[c])] for c in cands}
    return {q: sub_of_cand[t] for q, t in target_of.items()}

def stratified_folds(sub, n_folds, seed):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(sub), dtype=np.int64)
    for s in np.unique(sub):
        i = np.where(sub == s)[0]
        i = i[rng.permutation(len(i))]
        fold[i] = np.arange(len(i)) % n_folds
    return fold

def sample_pools(idx, sub, seed, n_cand=POOL):
    rng = np.random.default_rng(seed)
    by = {s: idx[sub[idx] == s] for s in np.unique(sub[idx])}
    P = np.empty((len(idx), n_cand), dtype=np.int64)
    for r, i in enumerate(idx):
        pop = by[sub[i]]
        pop = pop[pop != i]
        P[r, 0] = i
        P[r, 1:] = rng.choice(pop, size=n_cand - 1, replace=False)
    return P

def shuffle_pools(F, seed):
    rng = np.random.default_rng(seed)
    out = np.empty_like(F)
    lab = np.empty(F.shape[0], dtype=np.int64)
    for i in range(F.shape[0]):
        p = rng.permutation(F.shape[1])
        out[i] = F[i, p]
        lab[i] = int(np.where(p == 0)[0][0])
    return (out, lab)

def pool_score(S, lab):
    s0 = S[np.arange(len(S)), lab][:, None]
    greater = (S > s0).sum(1)
    equal = (S == s0).sum(1) - 1
    return float((greater + 0.5 * equal).mean() / (S.shape[1] - 1.0))

def validate_submission(sub, sample_path, pools_of):
    import pandas as pd
    sample = pd.read_csv(sample_path, keep_default_na=False)
    assert list(sub.columns) == list(sample.columns), f'columns {list(sub.columns)}'
    assert len(sub) == len(sample), f'rows {len(sub)} != {len(sample)}'
    assert sub.query_id.is_unique, 'duplicate query_id'
    assert set(sub.query_id) == set(sample.query_id), 'query id set mismatch'
    for qid, rank in zip(sub.query_id, sub.ranking):
        ids = rank.split()
        assert len(ids) == POOL, f'{qid}: {len(ids)} ids'
        assert len(set(ids)) == POOL, f'{qid}: duplicate candidate'
        assert set(ids) == set(pools_of[qid]), f"{qid}: ranking is not this query's pool"
    assert (sub.ranking.str.len() > 0).all(), 'empty ranking string'

def main():
    import pandas as pd
    seed_everything()
    log('reading data')
    train = pd.read_csv(PUBLIC_DIR / 'train.csv', keep_default_na=False)
    test = pd.read_csv(PUBLIC_DIR / 'test.csv', keep_default_na=False)
    cand = pd.read_csv(PUBLIC_DIR / 'candidates.csv', keep_default_na=False)
    labels = pd.read_csv(PUBLIC_DIR / 'train_labels.csv', keep_default_na=False)
    sample_path = PUBLIC_DIR / 'sample_submission.csv'
    sample = pd.read_csv(sample_path, keep_default_na=False)
    text_of = dict(zip(cand.candidate_id, cand.snippet_b))
    target_of = dict(zip(labels.query_id, labels.target_id))
    tr_pools = {q: c.split() for q, c in zip(train.query_id, train.candidates)}
    te_pools = {q: c.split() for q, c in zip(test.query_id, test.candidates)}
    A = list(train.snippet_a)
    B = [text_of[target_of[q]] for q in train.query_id]
    sub_of_q = subfields_from_train_pools(tr_pools, target_of)
    sub = np.array([sub_of_q[q] for q in train.query_id])
    n_train = len(A)
    log(f'{n_train} training abstracts, {len(np.unique(sub))} subfields recovered, {len(test)} evaluation queries')
    rng = np.random.default_rng(SEED)
    fit_text = A + B
    fit_text = fit_text + corrupt(fit_text, extra_rate(TRAIN_NOISE, EVAL_NOISE), rng)
    bank = PatternBank(SEED_SPEC, SKEL_SPEC, NBITS, MIN_DF, N_RARE).fit(fit_text)
    log(f'pattern bank fitted ({len(bank.seeds)} spaced seeds, {len(bank.skseeds)} structure seeds)')
    all_idx = np.arange(n_train)
    views = []
    for a in range(N_AUG):
        p = AUG_MIN + (AUG_MAX - AUG_MIN) * a / max(N_AUG - 1, 1)
        q = extra_rate(TRAIN_NOISE, p)
        ra = np.random.default_rng(SEED + 101 * a)
        ta = corrupt(A, q, ra) if q > 0 else list(A)
        tb = corrupt(B, q, ra) if q > 0 else list(B)
        Vq, Vc = (bank.vectors(ta), bank.vectors(tb))
        for s in range(N_POOLSAMP):
            P = sample_pools(all_idx, sub, seed=SEED + 977 * a + 31 * s)
            views.append(profiles(bank, Vq, Vc, P))
        log(f'  training view set {a + 1}/{N_AUG} at ~{p:.2f} corruption')
    V = np.stack(views, 0)
    zz = np.log1p(np.abs(V) * 100.0) * np.sign(V)
    MU = zz.reshape(-1, zz.shape[-1]).mean(0)
    SD = zz.reshape(-1, zz.shape[-1]).std(0) + 1e-06
    X = featurise(V, MU, SD)
    XL = np.empty(X.shape[:2], dtype=np.int64)
    for v in range(X.shape[0]):
        X[v], XL[v] = shuffle_pools(X[v], SEED + 61 * v)
    log(f'training tensor {X.shape}; features {FEATURE_NAMES}')
    te_ids = list(test.query_id)
    te_cand_ids = sorted({c for v in te_pools.values() for c in v})
    cpos = {c: i for i, c in enumerate(te_cand_ids)}
    te_pool_idx = np.array([[cpos[c] for c in te_pools[q]] for q in te_ids], dtype=np.int64)
    Vq_te = bank.vectors(list(test.snippet_a))
    Vc_te = bank.vectors([text_of[c] for c in te_cand_ids])
    X_te = torch.from_numpy(featurise(profiles(bank, Vq_te, Vc_te, te_pool_idx), MU, SD))
    log(f'evaluation tensor {tuple(X_te.shape)}')
    fold = stratified_folds(sub, N_FOLDS, SEED)
    Xt = torch.from_numpy(X)
    XLt = torch.from_numpy(XL)
    test_ranks = np.zeros((len(te_ids), POOL), dtype=np.float64)
    cv_scores = []
    for f in range(N_FOLDS):
        tr_i = np.where(fold != f)[0]
        va_i = np.where(fold == f)[0]
        vp = sample_pools(va_i, sub, seed=SEED + 7919 + f)
        loc = {g: k for k, g in enumerate(va_i)}
        vpl = np.vectorize(loc.get)(vp)
        rv = np.random.default_rng(SEED + 555 + f)
        qq = extra_rate(TRAIN_NOISE, EVAL_NOISE)
        Vq_v = bank.vectors(corrupt([A[i] for i in va_i], qq, rv))
        Vc_v = bank.vectors(corrupt([B[i] for i in va_i], qq, rv))
        Fva, lab_va = shuffle_pools(featurise(profiles(bank, Vq_v, Vc_v, vpl), MU, SD), SEED + f)
        Xva = torch.from_numpy(Fva)
        for s in range(N_SEEDS):
            torch.manual_seed(SEED + 13 * f + s)
            net = ListwiseRanker(X.shape[-1]).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
            steps = max(len(tr_i) // BATCH, 1)
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * steps, pct_start=0.2)
            rg = np.random.default_rng(SEED + 97 * f + s)
            best, best_state = (9.0, None)
            for ep in range(EPOCHS):
                net.train()
                for _ in range(steps):
                    qsel = torch.from_numpy(rg.choice(tr_i, size=BATCH, replace=False))
                    vsel = torch.from_numpy(rg.integers(0, X.shape[0], size=BATCH))
                    sc = net(Xt[vsel, qsel])
                    loss = F.cross_entropy(sc, XLt[vsel, qsel])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sch.step()
                if (ep + 1) % EVAL_EVERY == 0:
                    net.eval()
                    with torch.no_grad():
                        v = pool_score(net(Xva).numpy(), lab_va)
                    if v < best:
                        best = v
                        best_state = {k: t.clone() for k, t in net.state_dict().items()}
            net.load_state_dict(best_state)
            net.eval()
            with torch.no_grad():
                s_te = net(X_te).numpy()
            test_ranks += -np.argsort(np.argsort(-s_te, 1), 1)
            cv_scores.append(best)
            log(f'  fold {f} init {s}: held-out abstracts score {best:.4f}')
    log(f'CV (held-out abstracts, pools rebuilt from held-out abstracts, {EVAL_NOISE:.0%} noise): {np.mean(cv_scores):.4f} +- {np.std(cv_scores):.4f}')
    order = np.argsort(-test_ranks, axis=1, kind='stable')
    rankings = []
    for r, q in enumerate(te_ids):
        pool = te_pools[q]
        rankings.append(' '.join((pool[j] for j in order[r])))
    out = pd.DataFrame({'query_id': te_ids, 'ranking': rankings})
    out = out.set_index('query_id').loc[list(sample.query_id)].reset_index()
    validate_submission(out, sample_path, te_pools)
    out.to_csv(SUBMISSION_OUT, index=False)
    back = pd.read_csv(SUBMISSION_OUT, keep_default_na=False)
    validate_submission(back, sample_path, te_pools)
    log(f'wrote {SUBMISSION_OUT} shape={out.shape}')
if __name__ == '__main__':
    main()
