"""Tune the ordinal family (the strongest so far) and test in-fold feature
selection, on three station-held-out splits."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import cv, models, repsplits

D = cv.load()
folds = [D["fold"]] + repsplits.make(2)
F = "ABCDEFGHIJKLM"
print("== ordinal (logistic) regularisation ==")
for C in (0.003, 0.01, 0.03, 0.1, 0.3):
    cv.run_repeated(D, F, models.ORDINAL(C), folds, tag=f"ORD C={C}")
print("\n== ordinal with tree learners ==")
cv.run_repeated(D, F, models.ORDINAL_GB(0.06, 15, 200, 0.4), folds, tag="ORD-GB lf15 it200")
cv.run_repeated(D, F, models.ORDINAL_GB(0.05, 8, 300, 0.3), folds, tag="ORD-GB lf8 it300")
cv.run_repeated(D, F, models.ORDINAL_ET(800, 3), folds, tag="ORD-ET 800")
print("\n== plain multiclass references on the same splits ==")
cv.run_repeated(D, F, models.HGB(0.06, 15, 200, ff=0.4), folds, tag="HGB mf.4")
cv.run_repeated(D, F, models.ET(800, 3), folds, tag="ET800")
