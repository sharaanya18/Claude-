"""Cache evidence profiles so model choices can be swept cheaply.
Per outer fold: the train fold is split into an INNER-TRAIN part (the ranker's data and the
only text the idf/bucket statistics ever see) and an INNER-VAL part used for early stopping.
Inner-val and outer-val pools are built only from their own held-out abstracts, exactly the
way the hidden evaluation pools are built."""
import sys, time, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from evidence import Evidence

ap = argparse.ArgumentParser()
ap.add_argument("public"); ap.add_argument("out")
ap.add_argument("--fold", type=int, default=0); ap.add_argument("--naug", type=int, default=5)
ap.add_argument("--npool", type=int, default=4); ap.add_argument("--augmin", type=float, default=0.20)
ap.add_argument("--augmax", type=float, default=0.42)
ap.add_argument("--spec", type=str, default="w2:2-8,w3:3-8,w4:4-8")
ap.add_argument("--skel", type=str, default="w3:3-6,w4:4-7"); ap.add_argument("--nbuck", type=int, default=6)
ap.add_argument("--mindf", type=int, default=2); ap.add_argument("--innerfrac", type=float, default=0.2)
G = ap.parse_args()
D = Data(G.public); fold = make_folds(D.subv, 5, seed=0)
A, B = np.array(D.A, dtype=object), np.array(D.B, dtype=object)
f = G.fold
tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
rng = np.random.default_rng(4242 + f)
inner = make_folds(D.subv[tri], int(round(1 / G.innerfrac)), seed=77 + f)
tr_in, tr_va = tri[inner != 0], tri[inner == 0]
R30 = extra_rate(0.20, 0.30)

rs = np.random.default_rng(5 + f)
fit = list(A[tr_in]) + list(B[tr_in]); fit = fit + corrupt(fit, R30, rs)
t0 = time.time()
ev = Evidence(G.spec, G.skel, nbits=22, min_df=G.mindf, nbuck=G.nbuck).fit(fit)
print(f"fold {f}: idf on {len(tr_in)} inner-train abstracts, ncell={ev.ncell} nfeat={ev.nfeat} ({time.time()-t0:.0f}s)")

def pools_of(idx, seed):
    rr = np.random.default_rng(seed)
    bysub = {s: idx[D.subv[idx] == s] for s in np.unique(D.subv[idx])}
    P = np.empty((len(idx), 20), dtype=np.int64)
    for r, i in enumerate(idx):
        pop = bysub[D.subv[i]]; pop = pop[pop != i]
        P[r, 0] = i; P[r, 1:] = rr.choice(pop, size=19, replace=False)
    return P

def bank(idx, pools_seeds, augs, tag):
    pos = {g: k for k, g in enumerate(idx)}
    out = []
    for a, p in enumerate(augs):
        q = extra_rate(0.20, p)
        rr = np.random.default_rng(hash((tag, a)) % (2**31))
        ta = corrupt(list(A[idx]), q, rr) if q > 0 else list(A[idx])
        tb = corrupt(list(B[idx]), q, rr) if q > 0 else list(B[idx])
        Xq, Xc = ev.vectors(ta), ev.vectors(tb)
        for s in pools_seeds:
            P = np.vectorize(pos.get)(pools_of(idx, s))
            out.append(ev.profiles(Xq, Xc, P))
    return np.stack(out, 0)

augs_tr = [G.augmin + (G.augmax - G.augmin) * k / max(G.naug - 1, 1) for k in range(G.naug)]
t0 = time.time()
Ftr = bank(tr_in, [900 + 37 * s + f for s in range(G.npool)], augs_tr, f"tr{f}")
print(f"  train bank {Ftr.shape} ({time.time()-t0:.0f}s)")
Fiv = bank(tr_va, [31 + f], [0.30, 0.30], f"iv{f}")
Fov = bank(vai, [100 + f], [0.30, 0.30], f"ov{f}")
print(f"  inner-val {Fiv.shape}  outer-val {Fov.shape}")
np.savez_compressed(G.out, Ftr=Ftr.astype(np.float32), Fiv=Fiv.astype(np.float32),
                    Fov=Fov.astype(np.float32), ncellL=ev.cell["L"]["ncell"],
                    ncellS=ev.cell["S"]["ncell"])
print("saved", G.out)
