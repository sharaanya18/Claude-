"""Last lever: several views of one record. Sub-windows of a record's own valid
span are extra training rows with the same label (augmentation of a real
sample) and, at inference, several looks at the same record whose probabilities
are averaged. Nothing crosses records, so the per-record rule is untouched.
The noise-ceiling diagnostic says this can be worth at most a couple of points,
so it only ships if it actually measures positive on repeated CV."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import cv, models, repsplits
D = cv.load(); Vtr, _ = cv.load_crops()
folds = [D["fold"]] + repsplits.make(1)
F = "ABCDEFGHIJKM"
for name, mk in (("ORD", models.ORDINAL(0.03)), ("LR", models.LR(0.01))):
    cv.run_repeated(D, F, mk, folds, tag=f"{name} full span only")
    cv.run_repeated(D, F, mk, folds, Vtr=Vtr, aug=False, tag=f"{name} crop TTA only")
    cv.run_repeated(D, F, mk, folds, Vtr=Vtr, aug=True, tag=f"{name} crop aug + TTA")
