import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
P = Path("/home/user/Claude-/challenges/seismic_site_profile/dataset/public")

tr = pd.read_csv(P/"train.csv"); te = pd.read_csv(P/"test.csv")
ss = pd.read_csv(P/"sample_submission.csv", keep_default_na=False)
fo = pd.read_csv(P/"folds.csv")
print("shapes", tr.shape, te.shape, ss.shape, fo.shape)
print("cols", list(tr.columns), list(te.columns), list(ss.columns), list(fo.columns))
print("train dup ids", tr.id.duplicated().sum(), "test dup ids", te.id.duplicated().sum())
print("train/test id overlap", len(set(tr.id)&set(te.id)))
print("ss ids == test ids (order)", ss.id.tolist()==te.id.tolist())

# profile parsing
cells = tr.profile.str.split("|", expand=True)
print("cells shape", cells.shape, "any null", cells.isna().any().any())
for c in range(4):
    print("cell", c, Counter(cells[c]).most_common())
print("all tokens in v1..v4:", set(np.ravel(cells.values)) <= {"v1","v2","v3","v4"})

# folds coverage
print("folds ids == train ids set:", set(fo.id)==set(tr.id), "fold dup", fo.id.duplicated().sum())
print("n sites", fo.site.nunique(), "fold counts\n", fo.fold.value_counts().sort_index())
site_fold = fo.groupby("site").fold.nunique()
print("sites spanning >1 fold:", (site_fold>1).sum())
rps = fo.groupby("site").size()
print("records/site: min",rps.min(),"med",rps.median(),"max",rps.max(),"mean",round(rps.mean(),2))
print("sites per fold:\n", fo.groupby("fold").site.nunique())

# label is per-site constant?
m = tr.merge(fo, on="id")
pp = m.groupby("site").profile.nunique()
print("sites with >1 distinct profile:", (pp>1).sum(), "of", len(pp))

# per-fold class presence per cell
m[[f"c{i}" for i in range(4)]] = m.profile.str.split("|", expand=True)
for i in range(4):
    t = pd.crosstab(m.fold, m[f"c{i}"])
    print(f"\ncell{i} by fold (records):\n", t)
    # site-level
    sl = m.drop_duplicates("site")
    print(f"cell{i} by fold (sites):\n", pd.crosstab(sl.fold, sl[f"c{i}"]))

# profile combos
print("\nn distinct profiles", tr.profile.nunique())
print(tr.profile.value_counts().head(15))
# monotonic?
num = cells.replace({"v1":1,"v2":2,"v3":3,"v4":4}).astype(int)
mono = (num.values[:,1:] >= num.values[:,:-1]).all(axis=1)
print("records non-decreasing with depth:", mono.mean().round(4))
for i in range(3):
    inc=(num.values[:,i+1]>num.values[:,i]).mean(); same=(num.values[:,i+1]==num.values[:,i]).mean()
    print(f"cell{i}->cell{i+1}: inc {inc:.3f} same {same:.3f} dec {1-inc-same:.3f}")
print("\ncorr between cells (numeric band):\n", num.corr().round(3))
