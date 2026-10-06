"""Decode diagnostics: is the predicted band marginal skewed, and does
prior-matched / structured decoding recover macro-F1?"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import cv, models, decode, metric

D = cv.load(); Y = D["Y"]; fold = D["fold"]
F = "ABCDEFGHIJK"
r = cv.run_cv_views(D, F, models.HGB(0.05, 15, 300), seeds=(0,), tag="base HGB")
prob = r["prob"]
np.save("oof_base.npy", prob)

print("\n== predicted vs gold band marginal, per cell (argmax) ==")
pred = prob.argmax(axis=2)
for c in range(4):
    g = np.bincount(Y[:, c], minlength=4) / len(Y)
    p = np.bincount(pred[:, c], minlength=4) / len(Y)
    f1 = [metric.f1_per_class(Y[:, c], pred[:, c], k) for k in range(4)]
    print(f" cell{c}: gold {np.round(g,3)} pred {np.round(p,3)} "
          f"perclassF1 {np.round(f1,3)}")

print("\n== decode variants, params fitted NESTED inside each training fold ==")
# honest: for each outer fold, an inner site-held-out CV on the training part
# produces the probabilities the decode is fitted on.
def nested_decode(mode):
    Pred = np.zeros((len(Y), 4), dtype=int)
    for f in range(5):
        va = fold == f; tr = ~va
        inner_fold = fold[tr]
        ip = np.zeros((int(tr.sum()), 4, 4))
        Xa, _, m = cv.sel(D, F)
        Xt, Yt = Xa[tr], Y[tr]
        for g in sorted(set(inner_fold)):
            iva = inner_fold == g
            ip[iva] = models.HGB(0.05, 15, 300)(Xt[~iva], Yt[~iva], Xt[iva], 0)
        W = decode.fit_decode(ip, Yt, mode)
        Pred[va] = decode.apply_decode(prob[va], W)
    s, d = cv.score_oof(Y, Pred)
    pf = [cv.score_oof(Y[fold == f], Pred[fold == f])[0] for f in range(5)]
    print(f"  {mode:10s} OOF {s:.4f} | folds {' '.join(f'{x:.3f}' for x in pf)} "
          f"| cell {' '.join(f'{x:.3f}' for x in d['cell_f1'])}")
    return s

s_arg, _ = cv.score_oof(Y, prob.argmax(axis=2))
print(f"  {'argmax':10s} OOF {s_arg:.4f}")
nested_decode("prior")
nested_decode("uniform")

print("\n== structured joint-profile decode (train-only prior, nested lam) ==")
import pandas as pd
for lam in (0.0, 0.3, 0.6, 1.0, 1.5):
    Pred = np.zeros((len(Y), 4), dtype=int)
    for f in range(5):
        va = fold == f; tr = ~va
        prof, cnt = np.unique(Y[tr], axis=0, return_counts=True)
        q = decode.joint_prior_decode(prob[va], prof, lam, cnt.astype(float))
        Pred[va] = q.argmax(axis=2)
    s, d = cv.score_oof(Y, Pred)
    print(f"  lam={lam:<4} OOF {s:.4f} | cell {' '.join(f'{x:.3f}' for x in d['cell_f1'])}")
