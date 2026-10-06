"""DIAGNOSTIC ONLY - never part of a submission.

How much of the error is single-record earthquake noise rather than a limit of
the representation? Compare the normal per-record prediction against one where
the validation record's features are replaced by the mean over the ~5 records
of its own station. That averaging pools across evaluation records and is
FORBIDDEN in a submission (and impossible there - test records carry no site
key). It is run here purely to size the headroom: if the site-mean number is
far higher, the bottleneck is source/path noise in one record and the payoff is
in cancellation and variance reduction; if it is close, the representation
itself is the ceiling.
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import cv, models

D = cv.load(); Y = D["Y"]; fold = D["fold"]; site = D["site"]
Xa, _, _ = cv.sel(D, "ABCDEFGHIJK")
mdl = models.HGB(0.05, 15, 300)

for name, use_site_mean in (("per-record (honest)", False), ("site-mean (diagnostic)", True)):
    Prob = np.zeros((len(Y), 4, 4))
    for f in range(5):
        va = fold == f; tr = ~va
        Xv = Xa[va].copy()
        if use_site_mean:
            sv = site[va]
            for s in set(sv):
                m = sv == s
                Xv[m] = Xa[va][m].mean(axis=0)
        Prob[va] = mdl(Xa[tr], Y[tr], Xv, 0)
    pred = Prob.argmax(axis=2)
    s, d = cv.score_oof(Y, pred)
    pf = [cv.score_oof(Y[fold == f], pred[fold == f])[0] for f in range(5)]
    print(f"{name:24s} OOF {s:.4f} | folds {' '.join(f'{x:.3f}' for x in pf)} "
          f"| cell {' '.join(f'{x:.3f}' for x in d['cell_f1'])}")

# also: how consistent are a station's 5 single-record predictions with each other?
Prob = np.zeros((len(Y), 4, 4))
for f in range(5):
    va = fold == f; tr = ~va
    Prob[va] = mdl(Xa[tr], Y[tr], Xa[va], 0)
pred = Prob.argmax(axis=2)
agree = []
for s in set(site):
    m = site == s
    for c in range(4):
        v = pred[m, c]
        agree.append((v == np.bincount(v, minlength=4).argmax()).mean())
print(f"\nwithin-station agreement of per-record predictions: {np.mean(agree):.3f}"
      "\n(1.0 would mean the earthquake does not change the call at all)")
