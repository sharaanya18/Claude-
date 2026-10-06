"""Phase 6: baselines + information-ceiling diagnostics under CROSS-PERSON CV.

Everything is scored with the exact official metric on held-out PEOPLE.
Each baseline is fitted inside the fold (no statistic crosses the fold line).

Run: python3 baselines.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from kp_data import AXES, T, axis_skill, load_motion, load_targets, score
from kp_feat import DT, G, com_proxy, frame_features, anthro_features, sg_filter, W_ACC, W_VEL, ORDER
from groups import anthro, person_groups

OUT = Path(__file__).resolve().parents[1]
WORK = OUT / "working"
T0 = time.time()
L = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def folds_by_person(lab, k=5, seed=0):
    """Assign whole persons to folds, balancing trial counts. Deterministic."""
    rng = np.random.default_rng(seed)
    people = np.unique(lab)
    cnt = np.array([(lab == c).sum() for c in people])
    order = people[np.argsort(-cnt + rng.random(len(cnt)) * 0.1)]
    load = np.zeros(k)
    pf = {}
    for c in order:
        f = int(np.argmin(load))
        pf[c] = f
        load[f] += (lab == c).sum()
    return np.array([pf[c] for c in lab])


def report(name, P, Y, fold, extra=""):
    """Score a full OOF prediction array, overall + per axis + per fold."""
    tot, per_ax, per = score(P, Y, per_axis=True)
    pf = [per[fold == f].mean() for f in sorted(set(fold.tolist()))]
    p(f"{name:<40} {tot:.4f}   x {per_ax[0]:.4f}  y {per_ax[1]:.4f}  z {per_ax[2]:.4f}"
      f"   folds [{' '.join(f'{v:.3f}' for v in pf)}]  sd {np.std(pf):.4f} {extra}")
    return tot, per_ax, per


# ------------------------------------------------------------------ data ----
p("loading ...")
ids, X = load_motion("train")
_, Y = load_targets(ids)
F = anthro(X)
lab, _, _, _ = person_groups(F, 32)
fold = folds_by_person(lab, 5, 0)
N = len(ids)
p(f"N={N}  people={len(np.unique(lab))}  folds={[int((fold==f).sum()) for f in range(5)]}")
p(f"people per fold={[len(set(lab[fold==f])) for f in range(5)]}")

p("\nbuilding features ...")
FF = frame_features(X)                 # (N,C,T)
AF = anthro_features(X)                # (N,A)
p(f"frame features {FF.shape}  anthro features {AF.shape}  [{time.time()-T0:.0f}s]")
np.savez_compressed(WORK / "feats.npz", FF=FF, AF=AF, Y=Y, lab=lab, fold=fold,
                    ids=np.array(ids))
p(f"cached working/feats.npz  [{time.time()-T0:.0f}s]")

p("\n" + "=" * 110)
p("BASELINES, cross-person 5-fold, exact official metric")
p("=" * 110)
p(f"{'baseline':<40} {'TOTAL':>6}   {'x':>6}  {'y':>6}  {'z':>6}   per-fold")

# ---- B0 zero submission (the metric's definition of 'no skill') -----------
report("B0 all-zero submission", np.zeros_like(Y), Y, fold)

# ---- B1 global mean waveform (fit per fold on training people only) ------
P = np.zeros_like(Y)
for f in range(5):
    tr, va = fold != f, fold == f
    P[va] = Y[tr].mean(axis=0)[None]
report("B1 mean waveform (per-axis)", P, Y, fold)

# ---- B1b global MEDIAN waveform: L1 metric -> median is the optimal const --
P = np.zeros_like(Y)
for f in range(5):
    tr, va = fold != f, fold == f
    P[va] = np.median(Y[tr], axis=0)[None]
report("B1b median waveform (L1-optimal)", P, Y, fold)

# ---- B2 subject-normalised mean: scale the mean waveform per trial --------
# The metric is scale sensitive; a per-trial amplitude is NOT knowable without
# the target, so the honest version predicts the amplitude from motion. Here we
# first measure the ORACLE version to size the headroom.
P = np.zeros_like(Y)
for f in range(5):
    tr, va = fold != f, fold == f
    mw = np.median(Y[tr], axis=0)                      # (3,T)
    for a in range(3):
        num = np.abs(Y[va][:, a]).sum(axis=1)
        den = np.abs(mw[a]).sum()
        P[va, a] = mw[a][None] * (num / den)[:, None]
report("B2* ORACLE-amplitude median waveform", P, Y, fold, "<- uses truth, ceiling only")

# ---- B3 naive Newtonian, zero training -----------------------------------
com = com_proxy(X.astype(np.float64))
for w, nm in [(W_VEL, "sg15"), (W_ACC, "sg31"), (51, "sg51")]:
    acc = sg_filter(com, w, ORDER, 2, DT)
    P = np.stack([acc[:, :, 0] / G, acc[:, :, 1] / G + 1.0, acc[:, :, 2] / G], axis=1)
    report(f"B3 Newtonian COM accel ({nm}), untrained", P, Y, fold)

# ---- B4 ridge: per-trial scalar features -> whole waveform ----------------
def ridge_fit(A, B, lam):
    """Solve min ||A w - B||^2 + lam||w||^2 ; A (n,d) with bias appended."""
    d = A.shape[1]
    G_ = A.T @ A + lam * np.eye(d)
    return np.linalg.solve(G_, A.T @ B)


def standardise(tr, *others):
    mu, sd = tr.mean(0), tr.std(0) + 1e-8
    return [(z - mu) / sd for z in (tr,) + others]


AFb = np.concatenate([AF, np.ones((N, 1), np.float32)], axis=1)
for lam in [1.0, 10.0, 100.0]:
    P = np.zeros_like(Y)
    for f in range(5):
        tr, va = fold != f, fold == f
        a_tr, a_va = standardise(AF[tr], AF[va])
        A_tr = np.concatenate([a_tr, np.ones((a_tr.shape[0], 1))], 1)
        A_va = np.concatenate([a_va, np.ones((a_va.shape[0], 1))], 1)
        for a in range(3):
            W = ridge_fit(A_tr, Y[tr][:, a], lam)
            P[va, a] = A_va @ W
    report(f"B4 ridge(anthro+gait scalars) lam={lam:g}", P, Y, fold)

# ---- B5 local linear: per-FRAME features -> force at that frame ----------
# This is the key diagnostic: how much of the force is an instantaneous,
# person-independent linear function of the kinematics?
Cc = FF.shape[1]
Fflat = FF.transpose(0, 2, 1).reshape(N * T, Cc).astype(np.float64)
Yflat = Y.transpose(0, 2, 1).reshape(N * T, 3)
rowfold = np.repeat(fold, T)
for lam in [1e2, 1e3, 1e4]:
    P = np.zeros_like(Y)
    for f in range(5):
        trr, var = rowfold != f, rowfold == f
        mu = Fflat[trr].mean(0); sd = Fflat[trr].std(0) + 1e-8
        A_tr = np.concatenate([(Fflat[trr] - mu) / sd, np.ones((trr.sum(), 1))], 1)
        A_va = np.concatenate([(Fflat[var] - mu) / sd, np.ones((var.sum(), 1))], 1)
        W = ridge_fit(A_tr, Yflat[trr], lam)
        pred = A_va @ W                                    # (n_va*T, 3)
        P[fold == f] = pred.reshape((fold == f).sum(), T, 3).transpose(0, 2, 1)
    report(f"B5 per-frame ridge(kinematics) lam={lam:g}", P, Y, fold)

# ---- B6 kNN retrieval in anthropometric+gait space -----------------------
for K in [1, 5, 15, 40]:
    P = np.zeros_like(Y)
    for f in range(5):
        tr, va = fold != f, fold == f
        a_tr, a_va = standardise(AF[tr], AF[va])
        D = np.sqrt(((a_va[:, None, :] - a_tr[None, :, :]) ** 2).sum(-1))
        nn = np.argsort(D, axis=1)[:, :K]
        Ytr = Y[tr]
        P[va] = Ytr[nn].mean(axis=1)
    report(f"B6 kNN retrieval (K={K}) on scalars", P, Y, fold)

# ---- B7 PCA / DCT basis: how low-dimensional is the waveform? ------------
p("\n" + "-" * 110)
p("B7 TARGET BASIS: reconstruction ceiling (basis fitted on TRAIN people only)")
p("-" * 110)
for nc in [4, 8, 16, 24, 32, 48, 64, 96, 128]:
    P = np.zeros_like(Y)
    for f in range(5):
        tr, va = fold != f, fold == f
        for a in range(3):
            M = Y[tr][:, a]
            mu = M.mean(0)
            U, S, Vt = np.linalg.svd(M - mu, full_matrices=False)
            B = Vt[:nc]                                     # (nc,T)
            P[va, a] = (Y[va][:, a] - mu) @ B.T @ B + mu    # ORACLE projection
    tot, pa, _ = score(P, Y, per_axis=True)
    p(f"  PCA  {nc:>3} comps  ORACLE-projection score {tot:.4f}  "
      f"x {pa[0]:.4f} y {pa[1]:.4f} z {pa[2]:.4f}")
for nc in [8, 16, 24, 32, 48, 64, 96, 128]:
    k = np.arange(T)
    Bd = np.stack([np.cos(np.pi * (k + 0.5) * j / T) for j in range(nc)])   # DCT-II
    Bd /= np.linalg.norm(Bd, axis=1, keepdims=True)
    P = np.einsum('nat,ct,cs->nas', Y, Bd, Bd)
    tot, pa, _ = score(P, Y, per_axis=True)
    p(f"  DCT  {nc:>3} comps  ORACLE-projection score {tot:.4f}  "
      f"x {pa[0]:.4f} y {pa[1]:.4f} z {pa[2]:.4f}")
p("  -> the number of components where the ORACLE projection stops gaining is")
p("     the most compression the metric tolerates. Below that, a basis target")
p("     caps the achievable score (oversmoothing kills peaks the metric counts).")

# ---- information ceiling: per-person oracle ------------------------------
p("\n" + "-" * 110)
p("INFORMATION CEILING DIAGNOSTICS")
p("-" * 110)
# (a) if we knew the person, how good is THEIR OWN mean waveform?
P = np.zeros_like(Y)
for c in np.unique(lab):
    m = lab == c
    if m.sum() > 1:
        for i in np.where(m)[0]:
            o = m.copy(); o[i] = False
            P[i] = np.median(Y[o], axis=0)
report("C1* ORACLE same-person median waveform", P, Y, fold, "<- unreachable: person unknown at test")

# (b) leave-one-out global median (no person info)
P = np.zeros_like(Y)
med_all = np.median(Y, axis=0)
P[:] = med_all[None]
report("C2 global median (in-sample)", P, Y, fold)

# (c) how much does knowing the TRUE amplitude help on top of B5?
p("")
p(f"[{time.time()-T0:.0f}s] done")

with open(OUT / "reports" / "baselines_raw.txt", "w") as f:
    f.write("\n".join(L) + "\n")
p("wrote reports/baselines_raw.txt")
