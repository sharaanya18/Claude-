"""Which features describe the SITE rather than the earthquake?

Each station has ~5 records of different earthquakes but one answer, so the
between-site share of a feature's variance (an intraclass correlation) measures
how much of it is site signature and how much is source/path noise. Computed on
TRAIN only. High ICC is necessary but not sufficient - it also has to relate to
the label - so both are reported."""
import numpy as np, pandas as pd
import cv

D = cv.load()
X, site, Y = D["Xtr"], D["site"], D["Y"]
cols, gtag = D["cols"], D["gtag"]
Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)

codes, _ = pd.factorize(site)
ns = codes.max() + 1
grand = Xs.mean(0)
mu = np.zeros((ns, Xs.shape[1])); cnt = np.zeros(ns)
for i in range(ns):
    m = codes == i
    mu[i] = Xs[m].mean(0); cnt[i] = m.sum()
within = np.zeros(Xs.shape[1])
for i in range(ns):
    m = codes == i
    within += ((Xs[m] - mu[i]) ** 2).sum(0)
within /= (len(Xs) - ns)
between = (cnt[:, None] * (mu - grand) ** 2).sum(0) / (ns - 1)
icc = np.clip((between - within) / (between + within * (cnt.mean() - 1) + 1e-12), 0, 1)

# site-level |Spearman| with the mean numeric band (overall column stiffness)
from scipy.stats import spearmanr
ybar = Y.mean(1)
site_y = np.array([ybar[codes == i][0] for i in range(ns)])
rho = np.array([abs(spearmanr(mu[:, j], site_y).statistic) for j in range(Xs.shape[1])])

df = pd.DataFrame({"feat": cols, "grp": gtag, "icc": icc.round(3), "rho": rho.round(3)})
print("== median ICC and |rho| by group ==")
print(df.groupby("grp")[["icc", "rho"]].median().join(
      df.groupby("grp").size().rename("n")).round(3).to_string())
print("\n== top 25 features by ICC x |rho| ==")
df["prod"] = (df.icc * df.rho).round(3)
print(df.sort_values("prod", ascending=False).head(25).to_string(index=False))
print("\n== how many features clear thresholds ==")
for t in (0.2, 0.3, 0.4, 0.5):
    print(f"  ICC>{t}: {(icc > t).sum():3d}   ICC>{t} & rho>0.2: {((icc > t) & (rho > 0.2)).sum():3d}")
np.savez("icc.npz", icc=icc, rho=rho, cols=cols, gtag=gtag)
