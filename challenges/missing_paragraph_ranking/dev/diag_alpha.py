"""The cosine carries idf on BOTH sides, so a co-occurring pattern is weighted by idf^2.
Under a likelihood-ratio view a shared pattern is worth ~log(1/df), i.e. idf^1. Sweep the
per-side exponent alpha (alpha=0.5 -> idf^1 in total) and the tf transform."""
import sys
from pathlib import Path
import numpy as np, scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from ngram import seed_lib, codes, hash_features

D = Data(sys.argv[1]); fold = make_folds(D.subv, 5, seed=0)
A, B = np.array(D.A, dtype=object), np.array(D.B, dtype=object)
R30 = extra_rate(0.20, 0.30)
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 3
seeds = seed_lib(sys.argv[3] if len(sys.argv) > 3 else "w2:2-8,w3:3-8,w4:4-8")
NB = 22
ALPHAS = [0.25, 0.4, 0.5, 0.65, 0.8, 1.0]
acc = {}
for f in range(NF):
    tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
    pools = build_pools(vai, D.subv, seed=100 + f); pos = {g: k for k, g in enumerate(vai)}
    pl = np.vectorize(pos.get)(pools); I = np.arange(len(vai))[:, None]
    rs = np.random.default_rng(5 + f)
    fit = list(A[tri]) + list(B[tri]); fit = fit + corrupt(fit, R30, rs)
    X = hash_features(codes(fit), seeds, NB)
    df = np.asarray((X > 0).sum(0)).ravel(); n = X.shape[0]
    keep = df >= 2
    idf = np.zeros(X.shape[1], dtype=np.float32)
    idf[keep] = np.log((1.0 + n) / (1.0 + df[keep])) + 1.0
    r2 = np.random.default_rng(5000 + f)
    Ra = hash_features(codes(corrupt(list(A[vai]), R30, r2)), seeds, NB).tocsr()
    Rb = hash_features(codes(corrupt(list(B[vai]), R30, r2)), seeds, NB).tocsr()
    for subl in (True, False):
        for a in ALPHAS:
            w = idf ** a
            def tr(Y):
                Y = Y.copy()
                if subl: Y.data = 1.0 + np.log(Y.data)
                Y = Y.multiply(w[None, :]).tocsr(); Y.eliminate_zeros()
                nr = np.sqrt(np.asarray(Y.multiply(Y).sum(1))).ravel(); nr[nr == 0] = 1.0
                return sp.diags(1.0 / nr) @ Y
            S = np.asarray((tr(Ra) @ tr(Rb).T).todense())[I, pl]
            acc.setdefault((subl, a), []).append(score_from_matrix(S))
    print(f"fold {f} done", flush=True)
print(f"{'sublinear':>9s} {'alpha':>6s} | {'@0.30':>7s} {'sd':>6s}")
for k, v in sorted(acc.items(), key=lambda kv: np.mean(kv[1])):
    print(f"{str(k[0]):>9s} {k[1]:6.2f} | {np.mean(v):7.4f} {np.std(v):6.4f}")
