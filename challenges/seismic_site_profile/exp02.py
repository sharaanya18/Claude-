"""Ablate the feature groups (including the new cepstral J and coda-HVSR K)
and compare model families on the full set."""
import numpy as np, warnings, sys
warnings.filterwarnings("ignore")
import cv, models

D = cv.load()
print("features", D["Xtr"].shape)
for g in "ABCDEFGHIJK":
    print(" ", g, int((D["gtag"] == g).sum()), end="")
print()

H = models.HGB(0.08, 15, 120)          # fast surrogate for ablation
print("\n== incremental group ablation (HGB fast) ==")
seq = ["A", "AB", "ABC", "ABCD", "ABCDE", "ABCDEF", "ABCDEFG", "ABCDEFGH",
       "ABCDEFGHI", "ABCDEFGHIJ", "ABCDEFGHIJK"]
for gs in seq:
    print(f"{gs:12s}", end=" "); sys.stdout.flush()
    cv.run_cv(D, gs, H)

print("\n== new groups alone / leave-one-out ==")
for gs in ["J", "K", "JK", "AJ", "ADJK"]:
    print(f"{gs:12s}", end=" "); cv.run_cv(D, gs, H)
full = "ABCDEFGHIJK"
for g in full:
    gs = full.replace(g, "")
    print(f"drop {g:7s}", end=" "); cv.run_cv(D, gs, H)
