"""Minimal portability check, directly comparable to the exp10 numbers.

`max_features` on HistGradientBoostingClassifier needs scikit-learn >= 1.4. If
the grading machine were older the whole run would crash and score nothing, and
an environment-dependent fallback is itself a rejection pattern, so the
parameter only stays if it is measurably earning its place. Same two splits as
exp10, so 0.4 reproduces 0.2394 and 1.0 is the portable comparison.
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import cv, models, repsplits
D = cv.load(); folds = [D["fold"]] + repsplits.make(1)
F = "ABCDEFGHIJKM"
for ff in (1.0, 0.4):
    note = "portable default" if ff == 1.0 else "needs sklearn>=1.4"
    cv.run_repeated(D, F, models.ORDINAL_GB(0.06, 15, 200, ff, 1.0), folds,
                    tag=f"ORDGB max_features={ff} ({note})")
