"""Tune the lead family (ordinal cumulative-link with boosted binary learners)
on two station-held-out splits. The search is deliberately small: with 239
independent stations, cross-validation cannot resolve fine hyperparameter
differences, so only coarse, well-separated settings are compared and a broad
plateau is preferred to a sharp optimum."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import cv, models, repsplits
D = cv.load(); folds = [D["fold"]] + repsplits.make(1)
F = "ABCDEFGHIJKM"
for lr, leaf, it, ff, l2 in [(0.06, 15, 200, 0.4, 1.0),   # the shipped setting
                             (0.06, 8, 200, 0.4, 1.0),    # shallower trees
                             (0.04, 15, 300, 0.4, 1.0),   # slower, longer
                             (0.06, 15, 200, 0.2, 1.0)]:  # stronger subsampling
    cv.run_repeated(D, F, models.ORDINAL_GB(lr, leaf, it, ff, l2), folds,
                    tag=f"ORDGB lr{lr} lf{leaf} it{it} mf{ff} l2{l2}")
