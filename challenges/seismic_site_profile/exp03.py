"""Capacity, multi-seed averaging, and multi-view (crop) augmentation + TTA."""
import numpy as np, warnings, sys
warnings.filterwarnings("ignore")
import cv, models

D = cv.load()
Vtr, Vte = cv.load_crops()
F = "ABCDEFGHIJK"
print("full feature set", D["Xtr"].shape, "crops", Vtr.shape)

print("\n== capacity sweep, HGB on all groups ==")
for it, lr, leaf in [(200, 0.06, 15), (300, 0.05, 15), (400, 0.04, 15),
                     (300, 0.05, 31), (300, 0.05, 8), (600, 0.03, 15)]:
    cv.run_cv_views(D, F, models.HGB(lr, leaf, it), tag=f"HGB it{it} lr{lr} leaf{leaf}")

print("\n== subsampled features (less overfit on 239 units) ==")
for ff in (0.3, 0.5, 0.7):
    cv.run_cv_views(D, F, models.HGB(0.05, 15, 300, ff=ff), tag=f"HGB maxfeat{ff}")

print("\n== multi-seed averaging ==")
cv.run_cv_views(D, F, models.HGB(0.05, 15, 300), seeds=(0, 1, 2), tag="HGB x3 seeds")

print("\n== multi-view: TTA only, then augmentation + TTA ==")
cv.run_cv_views(D, F, models.HGB(0.05, 15, 300), Vtr=Vtr, aug=False, tag="HGB + crop TTA")
cv.run_cv_views(D, F, models.HGB(0.05, 15, 300), Vtr=Vtr, aug=True, tag="HGB + crop aug+TTA")

print("\n== other families on all groups ==")
cv.run_cv_views(D, F, models.LR(0.01), tag="LR C=0.01")
cv.run_cv_views(D, F, models.LR(0.003), tag="LR C=0.003")
cv.run_cv_views(D, F, models.LR(0.03, pca=60), tag="LR C=0.03 PCA60")
cv.run_cv_views(D, F, models.ET(800, 3), tag="ET 800")
cv.run_cv_views(D, F, models.ET(800, 8), tag="ET 800 leaf8")
cv.run_cv_views(D, F, models.KNN(30), tag="KNN30 cosine")
cv.run_cv_views(D, F, models.RIDGE_LATENT(300.0), tag="Ridge-latent a=300")
cv.run_cv_views(D, F, models.RIDGE_LATENT(1000.0), tag="Ridge-latent a=1000")
