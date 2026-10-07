"""Do spaced seeds recover matching evidence that contiguous n-grams lose to corruption?
Unsupervised cosine only, so this is a representation test, not a candidate solution."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from ngram import SeedTfidf, seed_sets

D = Data(sys.argv[1]); fold = make_folds(D.subv, 5, seed=0)
A, B = np.array(D.A, dtype=object), np.array(D.B, dtype=object)
R30 = extra_rate(0.20, 0.30)
print(f"{'seeds':10s} {'nseed':>5s} {'min_df':>6s} | {'@0.20':>7s} {'@0.30':>7s} {'sd':>6s} {'top1':>6s} {'top5':>6s}")
for name in ["c234", "c2345", "g3w4", "g3w5", "g4w5", "g4w6", "mix_a", "mix_b", "mix_c"]:
    seeds = seed_sets(name)
    for mindf in (3,):
        res = {0.20: [], 0.30: []}
        extra = {}
        for f in range(5):
            tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
            pools = build_pools(vai, D.subv, seed=100 + f)
            pos = {g: k for k, g in enumerate(vai)}
            pl = np.array([[pos[j] for j in row] for row in pools])
            rs = np.random.default_rng(5 + f)
            fit = list(A[tri]) + list(B[tri])
            fit = fit + corrupt(fit, R30, rs)
            vec = SeedTfidf(seeds, min_df=mindf).fit(fit)
            for eff in (0.20, 0.30):
                r2 = np.random.default_rng(5000 + f); q = extra_rate(0.20, eff)
                va = corrupt(list(A[vai]), q, r2) if q > 0 else list(A[vai])
                vb = corrupt(list(B[vai]), q, r2) if q > 0 else list(B[vai])
                Xa, Xb = vec.transform(va), vec.transform(vb)
                S = np.asarray((Xa @ Xb.T).todense())[np.arange(len(vai))[:, None], pl]
                res[eff].append(metrics_from_matrix(S))
                if eff == 0.30: extra = res[eff]
        s30 = [m["score"] for m in res[0.30]]
        print(f"{name:10s} {len(seeds):5d} {mindf:6d} | {np.mean([m['score'] for m in res[0.20]]):7.4f} "
              f"{np.mean(s30):7.4f} {np.std(s30):6.4f} {np.mean([m['top1'] for m in extra]):6.3f} "
              f"{np.mean([m['top5'] for m in extra]):6.3f}")
