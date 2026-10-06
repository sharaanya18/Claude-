"""Targeted improvement attempt: replicate solution.py's EXACT person-disjoint
holdout, then test one hypothesis per config.

Hypothesis H1 (the main one): the 27 trial-level scalars are close to person
IDENTIFIERS (body size and proportions), and the audit showed person identity is
nearly worthless on this task (oracle same-person template 0.285 vs 0.242 for a
global median). With only 25 training people in the holdout fit, conditioning on
them lets the network memorise walkers rather than learn the physics, which is
exactly the wrong inductive bias for 10 UNSEEN walkers. Splitting the scalars
into "body size" (identity-like, cols 0-14) and "gait/dynamics" (cols 15-26)
tests it directly.

Hypothesis H2: dropout 0.05 / wd 1e-4 is too weak for a 0.14 train-holdout gap.

Run: python3 v2.py <configA> [<configB> ...]
Everything is fitted on the holdout's TRAINING people only; the holdout is used
to compare configs and is reported, never used to fit anything.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

import kp_model
from kp_data import score
from kp_model import Standardiser, train_one

torch.set_num_threads(4)

WORK = Path(__file__).resolve().parents[1] / "working"
PRIOR_CH = [3, 4, 5]
SEED = 42
HOLDOUT_PEOPLE = 7
T0 = time.time()

# column split of the 27 trial scalars (see kp_feat.anthro_features order)
SIZE_COLS = list(range(0, 15))     # segment lengths + proportions -> identity-like
GAIT_COLS = list(range(15, 27))    # speed, cadence, accel spreads, excursions

CONFIGS = {
    # name:            dict of overrides
    "A_shipped":      dict(scalars="all", drop=0.05, wd=1e-4),
    "B_no_scalars":   dict(scalars="none", drop=0.05, wd=1e-4),
    "C_gait_only":    dict(scalars="gait", drop=0.05, wd=1e-4),
    "D_reg":          dict(scalars="all", drop=0.15, wd=3e-4),
    "E_gait_reg":     dict(scalars="gait", drop=0.15, wd=3e-4),
}
EPOCHS = 28
BATCH = 32
LR = 3e-3
WIDTH = 72
BLOCKS = 6
HUBER = 0.02


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def pick(AF, mode):
    if mode == "all":
        return AF
    if mode == "gait":
        return AF[:, GAIT_COLS]
    if mode == "none":
        return np.zeros((len(AF), 1), dtype=np.float32)
    raise ValueError(mode)


def main(names):
    FF = np.load(WORK / "cache_FF.npy")
    AF = np.load(WORK / "cache_AF.npy")
    Y = np.load(WORK / "cache_Y.npy").astype(np.float64)
    mFF = np.load(WORK / "cache_mFF.npy")
    mAF = np.load(WORK / "cache_mAF.npy")
    mY = np.load(WORK / "cache_mY.npy").astype(np.float64)
    lab = np.load(WORK / "cache_lab.npy")

    # EXACTLY solution.py's holdout: the 7 largest person clusters
    sizes = np.bincount(lab)
    order = np.argsort(-sizes)
    hold = set(order[:HOLDOUT_PEOPLE].tolist())
    va = np.array([i for i in range(len(lab)) if lab[i] in hold])
    trn = np.array([i for i in range(len(lab)) if lab[i] not in hold])
    log(f"holdout {len(hold)} people / {len(va)} trials; train {len(trn)} trials, "
        f"{len(set(lab[trn]))} people")

    PR = np.ascontiguousarray(FF[:, PRIOR_CH, :])
    mPR = np.ascontiguousarray(mFF[:, PRIOR_CH, :])

    # best constant waveform on this split, the strip-the-ML reference
    med = np.median(Y[trn], axis=0)
    mtot, _, _ = score(np.repeat(med[None], len(va), axis=0), Y[va], per_axis=True)

    results = {}
    for nm in names:
        cfg = CONFIGS[nm]
        A_tr, A_va, A_mr = (pick(AF[trn], cfg["scalars"]), pick(AF[va], cfg["scalars"]),
                            pick(mAF[trn], cfg["scalars"]))
        st = Standardiser(FF, AF, trn)
        amu = A_tr.mean(0, keepdims=True)
        asd = A_tr.std(0, keepdims=True) + 1e-6
        log(f"=== {nm}  scalars={cfg['scalars']} ({A_tr.shape[1]} cols) "
            f"drop={cfg['drop']} wd={cfg['wd']}")
        # patch the module-level dropout used inside ResBlock
        vp, _, hist = train_one(
            np.concatenate([st.f(FF[trn]), st.f(mFF[trn])]),
            np.concatenate([(A_tr - amu) / asd, (A_mr - amu) / asd]).astype(np.float32),
            np.concatenate([Y[trn], mY[trn]]),
            st.f(FF[va]), ((A_va - amu) / asd).astype(np.float32), Y[va],
            np.concatenate([PR[trn], mPR[trn]]), PR[va],
            width=WIDTH, blocks=BLOCKS, k=5, drop=cfg["drop"], epochs=EPOCHS,
            bs=BATCH, lr=LR, wd=cfg["wd"], seed=SEED, huber=HUBER,
            phys_residual=True, log=log)
        tot, pax, per = score(vp, Y[va], per_axis=True)
        pp = sorted(float(per[lab[va] == c].mean()) for c in hold)
        results[nm] = (tot, pax, pp, hist[-1][1])
        log(f"RESULT {nm}: TOTAL {tot:.4f}  x {pax[0]:.4f} y {pax[1]:.4f} z {pax[2]:.4f}"
            f"   train_E {hist[-1][1]:.4f}  gap {(1-hist[-1][1])-tot:+.4f}")
        log(f"  per-unseen-person: {' '.join(f'{v:.3f}' for v in pp)}")

    log("")
    log(f"{'config':<16} {'TOTAL':>7} {'x':>7} {'y':>7} {'z':>7} {'gap':>7}  vs A")
    base = results.get("A_shipped", (None,))[0]
    for nm, (tot, pax, pp, tr) in results.items():
        d = f"{tot-base:+.4f}" if base else "  n/a"
        log(f"{nm:<16} {tot:7.4f} {pax[0]:7.4f} {pax[1]:7.4f} {pax[2]:7.4f} "
            f"{(1-tr)-tot:7.4f}  {d}")
    log(f"best constant waveform on this split: {mtot:.4f}")
    log(f"printed AI baseline to beat: 0.60")


if __name__ == "__main__":
    main(sys.argv[1:] or ["A_shipped", "C_gait_only"])
