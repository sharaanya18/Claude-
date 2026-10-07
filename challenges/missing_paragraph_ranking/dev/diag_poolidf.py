"""Every candidate in a pool shares the query's subfield, so patterns that occur across the
whole pool carry no discriminating power. Re-weight each matched pattern by how rare it is
WITHIN THIS QUERY'S OWN POOL (the pool is part of one query's input, so this stays row-local
and never touches another query). Unsupervised probe of the idea before building it in."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from ngram import SeedTfidf, seed_lib

D = Data(sys.argv[1]); fold = make_folds(D.subv, 5, seed=0)
A, B = np.array(D.A, dtype=object), np.array(D.B, dtype=object)
R30 = extra_rate(0.20, 0.30)
seeds = seed_lib("w2:2-8,w3:3-8,w4:4-8")
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 5
acc = {a: [] for a in ALPHAS}
for f in range(NF):
    tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
    pools = build_pools(vai, D.subv, seed=100 + f)
    pos = {g: k for k, g in enumerate(vai)}
    pl = np.vectorize(pos.get)(pools)
    rs = np.random.default_rng(5 + f)
    fit = list(A[tri]) + list(B[tri]); fit = fit + corrupt(fit, R30, rs)
    vec = SeedTfidf(seeds, nbits=22, min_df=2).fit(fit)
    r2 = np.random.default_rng(5000 + f)
    Xa = vec.transform(corrupt(list(A[vai]), R30, r2)).tocsr()
    Xb = vec.transform(corrupt(list(B[vai]), R30, r2)).tocsr()
    H = vec.idf.shape[0]
    buf = np.zeros(H, dtype=np.float32); cnt = np.zeros(H, dtype=np.int16)
    S = {a: np.zeros(pl.shape) for a in ALPHAS}
    for q in range(len(vai)):
        rows = pl[q]
        touched = []
        for r in rows:                                   # pool document frequency, this pool only
            bi = Xb.indices[Xb.indptr[r]:Xb.indptr[r + 1]]
            cnt[bi] += 1; touched.append(bi)
        s, e = Xa.indptr[q], Xa.indptr[q + 1]
        ai = Xa.indices[s:e]; buf[ai] = Xa.data[s:e]
        for p, r in enumerate(rows):
            bi = touched[p]
            prod = buf[bi] * Xb.data[Xb.indptr[r]:Xb.indptr[r + 1]]
            nz = prod > 0
            if not nz.any(): continue
            pv = prod[nz]; pc = cnt[bi[nz]].astype(np.float32)
            for a in ALPHAS:
                S[a][q, p] = (pv / pc ** a).sum()
        buf[ai] = 0.0
        for bi in touched: cnt[bi] -= 1
    for a in ALPHAS: acc[a].append(score_from_matrix(S[a]))
print(f"{'alpha':>6s} | {'@0.30':>7s} {'sd':>6s}   (alpha=0 is the plain cosine)")
for a in ALPHAS:
    print(f"{a:6.2f} | {np.mean(acc[a]):7.4f} {np.std(acc[a]):6.4f}")
