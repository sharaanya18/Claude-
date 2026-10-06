import numpy as np, pandas as pd, time
from pathlib import Path
import features
P = Path("dataset/public")
z = np.load(P/"waveforms.npz")
ids = list(z.keys())
t0=time.time()
rows = {g: [] for g in features.GROUPS}
for i,k in enumerate(ids):
    d = features.extract(z[k])
    for g in features.GROUPS: rows[g].append(d[g])
X = np.concatenate([np.array(rows[g], dtype=np.float64) for g in features.GROUPS], axis=1)
nm = features.names()
cols, gtag = [], []
for g in features.GROUPS:
    cols += nm[g]; gtag += [g]*len(nm[g])
print("X", X.shape, "s", round(time.time()-t0,1), "finite", np.isfinite(X).all())
np.savez_compressed("feat_cache.npz", X=X, ids=np.array(ids), cols=np.array(cols), gtag=np.array(gtag))
