"""One pooled vector space lets the seed family with the most features dominate the cosine.
Score each family in its OWN normalised space instead, then fuse the per-family scores after
standardising them within the query's pool, so each family contributes on a comparable scale."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from ngram import SeedTfidf, seed_lib

D = Data(sys.argv[1]); fold = make_folds(D.subv, 5, seed=0)
A, B = np.array(D.A, dtype=object), np.array(D.B, dtype=object)
R30 = extra_rate(0.20, 0.30)
FAM = {"w2": "w2:2-8", "w3": "w3:3-8", "w4": "w4:4-8", "w5": "w5:5-8"}
POOLED = "w2:2-8,w3:3-8,w4:4-8"
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 5

def zpool(S):                       # standardise each score within its own pool
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
def rpool(S):                       # within-pool rank, 0..1
    o = np.argsort(np.argsort(S, 1), 1)
    return o / (S.shape[1] - 1.0)

acc = {}
def add(k, v): acc.setdefault(k, []).append(v)
for f in range(NF):
    tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
    pools = build_pools(vai, D.subv, seed=100 + f)
    pos = {g: k for k, g in enumerate(vai)}
    pl = np.vectorize(pos.get)(pools)
    rs = np.random.default_rng(5 + f)
    fit = list(A[tri]) + list(B[tri]); fit = fit + corrupt(fit, R30, rs)
    r2 = np.random.default_rng(5000 + f)
    va = corrupt(list(A[vai]), R30, r2); vb = corrupt(list(B[vai]), R30, r2)
    idx = np.arange(len(vai))[:, None]
    def score(spec):
        vec = SeedTfidf(seed_lib(spec), nbits=22, min_df=2).fit(fit)
        return np.asarray((vec.transform(va) @ vec.transform(vb).T).todense())[idx, pl]
    Sp = score(POOLED); add("pooled space (baseline)", score_from_matrix(Sp))
    fam = {k: score(v) for k, v in FAM.items()}
    for k, v in fam.items(): add(f"  family {k} alone", score_from_matrix(v))
    for name, keys in [("w2+w3+w4", ["w2", "w3", "w4"]), ("w3+w4", ["w3", "w4"]),
                       ("w2+w3+w4+w5", ["w2", "w3", "w4", "w5"])]:
        add(f"mean  {name}", score_from_matrix(np.mean([fam[k] for k in keys], 0)))
        add(f"zmean {name}", score_from_matrix(np.mean([zpool(fam[k]) for k in keys], 0)))
        add(f"rmean {name}", score_from_matrix(np.mean([rpool(fam[k]) for k in keys], 0)))
    add("zmean pooled+fams", score_from_matrix(np.mean([zpool(Sp)] + [zpool(fam[k]) for k in ("w2","w3","w4")], 0)))
print(f"{'method':28s} | {'@0.30':>7s} {'sd':>6s}")
for k, v in acc.items():
    print(f"{k:28s} | {np.mean(v):7.4f} {np.std(v):6.4f}")
