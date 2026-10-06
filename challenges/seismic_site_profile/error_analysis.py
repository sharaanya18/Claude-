"""Phase 23/25 stress test on cached OOF predictions: where does the model fail,
and is any apparent gain carried by a single fold or a single band?"""
import numpy as np, warnings, pandas as pd
warnings.filterwarnings("ignore")
import cv, metric

D = cv.load(); Y = D["Y"]; site = D["site"]
import sys
P = np.load(sys.argv[1] if len(sys.argv) > 1 else "oof_ord.npy")
pred = P.argmax(axis=2)

print("== per-cell, per-band F1 and prediction counts ==")
for c in range(4):
    g = np.bincount(Y[:, c], minlength=4)
    p = np.bincount(pred[:, c], minlength=4)
    f1 = [metric.f1_per_class(Y[:, c], pred[:, c], k) for k in range(4)]
    print(f" cell{c}: gold {g} pred {p} F1 {np.round(f1,3)} macro {np.mean(f1):.3f}")

print("\n== confusion, cell by cell (rows gold, cols pred) ==")
for c in range(4):
    M = np.zeros((4, 4), int)
    for a, b in zip(Y[:, c], pred[:, c]):
        M[a, b] += 1
    print(f" cell{c}\n{M}")

print("\n== error vs how extreme the column is ==")
mean_band = Y.mean(1)
for lo, hi, name in [(0, 0.6, "very soft"), (0.6, 1.6, "soft"),
                     (1.6, 2.4, "middle"), (2.4, 3.1, "stiff")]:
    m = (mean_band >= lo) & (mean_band < hi)
    if m.sum() < 10:
        continue
    acc = (pred[m] == Y[m]).mean()
    print(f"  {name:10s} n={m.sum():4d} cellwise accuracy {acc:.3f}")

print("\n== per-station consistency ==")
agree = []
for s in set(site):
    m = site == s
    for c in range(4):
        v = pred[m, c]
        agree.append((v == np.bincount(v, minlength=4).argmax()).mean())
print(f"  within-station agreement of per-record calls: {np.mean(agree):.3f}")

print("\n== is the model just predicting one latent stiffness? ==")
print("  corr between predicted cell bands:")
print(np.round(np.corrcoef(pred.T), 3))
print("  corr between gold cell bands:")
print(np.round(np.corrcoef(Y.T), 3))
