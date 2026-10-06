"""Does the short-window high-frequency family (N) earn the shallowest cell?
Judged with the best single family on three station-held-out splits, watching
cell 0 specifically - that is the cell the family was designed for."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import cv, models, repsplits
D = cv.load(); folds = [D["fold"]] + repsplits.make(2)
for gs in ["ABCDEFGHIJKM", "ABCDEFGHIJKMN", "ABCDEFGHIJKLMN"]:
    n = int(np.isin(D["gtag"], list(gs)).sum())
    cv.run_repeated(D, gs, models.ORDINAL(0.03), folds, tag=f"ORD {gs} ({n}f)")
    cv.run_repeated(D, gs, models.LR(0.01), folds, tag=f"LR  {gs} ({n}f)")
