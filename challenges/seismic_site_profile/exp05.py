"""Stage 1 screening on the supplied split, paired on identical folds.
Reference model is HGB(lr .06, leaf 15, 200 it) - the best measured so far."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import cv, models
D = cv.load(); Vtr, _ = cv.load_crops()
R = cv.run_cv_views
REF = lambda: models.HGB(0.06, 15, 200)
print("features", D["Xtr"].shape)

print("\n== do the new smoothed/low-frequency groups add? ==")
for gs in ["ABCDEFGHIJK", "ABCDEFGHIJKL", "ABCDEFGHIJKM", "ABCDEFGHIJKLM"]:
    R(D, gs, REF(), tag=f"{gs} ({int(np.isin(D['gtag'],list(gs)).sum())}f)")

print("\n== compact sets built from the high-ICC families ==")
for gs in ["ADKLM", "ABDEKLM", "ABDEGHKLM", "ABDEFGHKLM", "ABCDEFGHKLM"]:
    R(D, gs, REF(), tag=f"{gs} ({int(np.isin(D['gtag'],list(gs)).sum())}f)")

F = "ABCDEFGHIJKLM"
print("\n== capacity on the full set ==")
for it, lr, leaf, l2, ff in [(200, 0.06, 15, 1.0, 1.0), (200, 0.06, 8, 1.0, 1.0),
                             (300, 0.04, 8, 1.0, 1.0), (200, 0.06, 8, 5.0, 0.4),
                             (200, 0.06, 15, 1.0, 0.4), (400, 0.03, 15, 1.0, 0.4),
                             (150, 0.09, 31, 1.0, 0.5)]:
    R(D, F, models.HGB(lr, leaf, it, l2, ff=ff),
      tag=f"HGB it{it} lr{lr} lf{leaf} l2{l2} mf{ff}")

print("\n== variance reduction ==")
R(D, F, models.HGB(0.06, 15, 200, ff=0.4), seeds=(0, 1, 2), tag="HGB mf.4 x3seeds")
R(D, F, models.HGB(0.06, 15, 200, ff=0.4), Vtr=Vtr, aug=False, tag="HGB mf.4 cropTTA")
R(D, F, models.HGB(0.06, 15, 200, ff=0.4), Vtr=Vtr, aug=True, tag="HGB mf.4 aug+TTA")

print("\n== other families (ensemble candidates) ==")
for C in (0.003, 0.01, 0.03):
    R(D, F, models.LR(C), tag=f"LR C={C}")
for n in (40, 80):
    R(D, F, models.LR(0.03, pca=n), tag=f"LR C.03 PCA{n}")
R(D, F, models.ET(800, 3), tag="ET800 leaf3")
R(D, F, models.ET(800, 8), tag="ET800 leaf8")
for k in (20, 40):
    R(D, F, models.KNN(k), tag=f"KNN{k} cosine")
for a in (100., 300., 1000.):
    R(D, F, models.RIDGE_LATENT(a), tag=f"Ridge-latent a={a}")
