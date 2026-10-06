"""Baseline family sweep: reproduce the reference tiers, then look for headroom."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

D = cv.load()
print("Xtr", D["Xtr"].shape, "sites", len(set(D["site"])))

def mk_percell(make):
    def fp(Xtr, Ytr, Xva, seed):
        out = np.zeros((len(Xva), 4, 4))
        for c in range(4):
            m = make(seed)
            m.fit(Xtr, Ytr[:, c])
            pr = m.predict_proba(Xva)
            cls = m[-1].classes_ if hasattr(m, "steps") else m.classes_
            for j, cl in enumerate(cls):
                out[:, c, cl] = pr[:, j]
        return out
    return fp

LR  = lambda C: mk_percell(lambda s: make_pipeline(StandardScaler(),
        LogisticRegression(C=C, max_iter=3000)))
KNN = lambda k: mk_percell(lambda s: make_pipeline(StandardScaler(),
        KNeighborsClassifier(n_neighbors=k, weights="distance", metric="cosine")))
HGB = lambda lr,leaf: mk_percell(lambda s: HistGradientBoostingClassifier(
        learning_rate=lr, max_leaf_nodes=leaf, max_iter=200, l2_regularization=1.0,
        random_state=s, early_stopping=False))
ET  = lambda n: mk_percell(lambda s: ExtraTreesClassifier(n_estimators=n, random_state=s,
        min_samples_leaf=3, n_jobs=4))

print("\n== reference tier reproduction ==")
print("two scalars (hv f0 + centroid) + LR:")
# mimic the 0.0678 reference: a couple of hand-built scalars only
import numpy as np
m = np.isin(D["gtag"], ["D","E"])
cols = D["cols"][m]
keep = np.array([c in ("hv_f0","H_cen") for c in D["cols"]])
Dsub = dict(D); Dsub["gtag"] = np.where(keep, "Z", "_")
cv.run_cv(Dsub, "Z", LR(1.0))

print("5NN in log-spectral space (A+B+C):"); cv.run_cv(D, "ABC", KNN(5))
print("LR on A+B+C:");                      cv.run_cv(D, "ABC", LR(1.0))
print("HGB on A+B+C:");                     cv.run_cv(D, "ABC", HGB(0.06, 15))
print("\n== all 215 features ==")
print("LR all:");  cv.run_cv(D, "ABCDEFGHI", LR(1.0))
print("LR all C=0.1:"); cv.run_cv(D, "ABCDEFGHI", LR(0.1))
print("LR all C=0.03:"); cv.run_cv(D, "ABCDEFGHI", LR(0.03))
print("HGB all:"); cv.run_cv(D, "ABCDEFGHI", HGB(0.06, 15))
print("ET all:");  cv.run_cv(D, "ABCDEFGHI", ET(400))
