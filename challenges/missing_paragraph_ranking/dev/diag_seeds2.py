import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from ngram import SeedTfidf, seed_lib

D = Data(sys.argv[1]); fold = make_folds(D.subv, 5, seed=0)
A, B = np.array(D.A, dtype=object), np.array(D.B, dtype=object)
R30 = extra_rate(0.20, 0.30)
SPECS = [s.strip() for s in sys.argv[2].split("|")]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 20
MD = int(sys.argv[4]) if len(sys.argv) > 4 else 3
print(f"nbits={NB} min_df={MD}")
print(f"{'spec':34s} {'#seed':>5s} | {'@0.20':>7s} {'@0.30':>7s} {'sd':>6s} {'top1':>6s} {'top5':>6s}")
for spec in SPECS:
    seeds = seed_lib(spec)
    res = {0.20: [], 0.30: []}
    for f in range(5):
        tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
        pools = build_pools(vai, D.subv, seed=100 + f)
        pos = {g: k for k, g in enumerate(vai)}
        pl = np.array([[pos[j] for j in row] for row in pools])
        rs = np.random.default_rng(5 + f)
        fit = list(A[tri]) + list(B[tri]); fit = fit + corrupt(fit, R30, rs)
        vec = SeedTfidf(seeds, nbits=NB, min_df=MD).fit(fit)
        for eff in (0.20, 0.30):
            r2 = np.random.default_rng(5000 + f); q = extra_rate(0.20, eff)
            va = corrupt(list(A[vai]), q, r2) if q > 0 else list(A[vai])
            vb = corrupt(list(B[vai]), q, r2) if q > 0 else list(B[vai])
            S = np.asarray((vec.transform(va) @ vec.transform(vb).T).todense())[np.arange(len(vai))[:, None], pl]
            res[eff].append(metrics_from_matrix(S))
    s30 = [m["score"] for m in res[0.30]]
    print(f"{spec:34s} {len(seeds):5d} | {np.mean([m['score'] for m in res[0.20]]):7.4f} "
          f"{np.mean(s30):7.4f} {np.std(s30):6.4f} {np.mean([m['top1'] for m in res[0.30]]):6.3f} "
          f"{np.mean([m['top5'] for m in res[0.30]]):6.3f}")
