"""KineProof: Echoes of an Invisible Floor -- motion-capture -> 3-axis ground-reaction-force waveform.

Reconstructs the 256-frame, 3-axis GRF waveform (body-weight units) of a walking
trial from its 256x22x3 lower-body marker sequence, by training a dilated
temporal convolutional network from scratch inside this script.

=====================================================================
REQUIREMENTS MAP (challenge description -> where this script satisfies it)
=====================================================================
"Reconstruct the measured 3D GRF waveform ... from a marker sequence"
    -> a trained seq2seq temporal CNN maps (256,22,3) markers to (3,256) force
       (build_model / train_fold). Not a lookup, not a template, not a rule.
"each answer is a force vector at the same 256 time points"
    -> every output row is 3 JSON arrays of exactly 256 finite floats.
"Exactly 322 data rows, one for each test.csv ID, plus the header"
    -> validate_submission() asserts count, ids, order, length and finiteness
       and RAISES before any file is written.
"force_x forward, force_y vertical, force_z lateral, body-weight units"
    -> targets are used in their given units; the Newtonian prior channels are
       built in exactly those units (see PHYSICS note below).
"Each axis gets equal weight after its own reference normalization"
    -> the training loss IS the metric's error term: per trial-axis L1 divided
       by that trial-axis sum|y| (metric_loss). An unweighted loss would let
       force_y (whose denominator is ~11x force_x's and ~22x force_z's)
       dominate and, as the description warns, cap the score near one third.
"A model that reconstructs only vertical loading can earn at most one third"
    -> all three axes are predicted jointly and weighted equally by the loss.
"All trials from a person stay on one side of the split" / 10 unseen people
    -> the in-script validation holds out whole PERSONS, grouped by
       anthropometry (derive_person_groups), because person ids are not
       disclosed. Random trial-level validation would be leaked and optimistic.
"Submit predictions, not code. The grader does not execute solver code."
    -> this script writes the predictions CSV; no solver code is referenced
       from the submission file.
"Do not submit a scalar per trial, a foot-contact class, or a force-plate index"
    -> output is the complete three-axis waveform.
"Do not copy the placeholder submission as a solution."
    -> sample_submission.csv is read for its COLUMN NAMES and ID ORDER only
       (never for its placeholder values); predictions come from the model.

Compliance notes
----------------
* Self-contained: raw bundle -> features -> training -> inference -> CSV, every
  run. No cached weights, embeddings or artefacts from any earlier run.
* No pretrained weights and no external data: the network is trained from
  scratch on the supplied training trials only.
* Nothing is fitted on the test split. Feature standardisation, the person
  clustering and every model parameter are fitted on TRAIN rows only and then
  applied to test with .transform-style reuse. Each test row's prediction is a
  function of that row's own markers and the frozen train-fitted model, so
  dropping any other test rows cannot change it (no whole-test statistic).
* Fixed work plan: fixed epochs, folds, seeds, batch size, thread count and
  schedule. Wall-clock time is used for logging only -- never in a condition,
  a loop bound or a library argument.
* Device is hardcoded to CPU. The description states no hardware, and the model
  is sized to finish well inside the runtime budget on CPU; this avoids any
  environment-dependent branch (which the Deterministic Execution check flags)
  and cannot crash on a machine without a GPU.
* No hardcoded tuned constants: the architecture/schedule constants below are
  round, principled defaults; the physical constants (g, Winter segment-mass
  fractions, 150 Hz sampling rate) come from units and the biomechanics
  literature, not from score tuning.

PHYSICS note (why this task generalises to unseen walkers at all)
----------------------------------------------------------------
Newton for the whole body: GRF = m*a_com + m*g (vertical), GRF = m*a_com
(horizontal). The target is normalised by m*g, so in body-weight units

    f_x = a_com_x / g,    f_y = a_com_y / g + 1,    f_z = a_com_z / g

MASS CANCELS: the target is a pure kinematic quantity, COM acceleration in
units of g. That is supplied to the network as explicit input channels and as
an additive residual base, so the network only has to learn the *correction*
from a lower-body COM proxy to the true whole-body COM -- the trunk, arms and
head carry most of the mass and are not instrumented -- plus the envelope of
when the walker is actually on the instrumented area.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- contract --
PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
SUBMISSION_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")
WORK_DIR = SUBMISSION_OUT.parent
WORK_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# ------------------------------------------------------- fixed work plan ----
SEED = 42
DEVICE = "cpu"          # hardcoded: no environment-dependent branch anywhere
T = 256                 # frames per trial (given by the challenge)
N_MARKERS = 22
FS = 150.0              # Hz: 256 frames / "about 1.7 seconds" -> 150 Hz source rate
DT = 1.0 / FS
G = 9.80665             # the normalising gravity the description names

WIDTH = 72              # TCN channels
BLOCKS = 6              # residual blocks; dilations 1,2,4,8,16,32 -> RF 253 frames
KERNEL = 5
DROPOUT = 0.05
EPOCHS = 28
BATCH = 32
LR = 3e-3
WEIGHT_DECAY = 1e-4
HUBER = 0.02            # smooths the L1 kink early in training
N_SEEDS = 3             # full-data refits averaged (variance reduction)
N_PEOPLE = 32           # the description states 32 training people
HOLDOUT_PEOPLE = 7      # one person-disjoint validation fold for the diagnostic

T0 = time.time()        # LOGGING ONLY -- never used in a condition


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(8)        # fixed, not derived from os.cpu_count()


# ------------------------------------------------------------- marker map ---
MARKERS = [
    "R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS", "L.Iliac.Crest", "R.Iliac.Crest",
    "R.GTR", "R.Knee", "R.HF", "R.TT", "R.Ankle", "R.Heel", "R.MT1", "R.MT5",
    "L.GTR", "L.Knee", "L.HF", "L.TT", "L.Ankle", "L.Heel", "L.MT1", "L.MT5",
]
MI = {m: i for i, m in enumerate(MARKERS)}
# contralateral permutation, used by the left/right mirror augmentation
MIRROR_IDX = [MI[("L." + m[2:]) if m.startswith("R.") else ("R." + m[2:])] for m in MARKERS]
PELVIS4 = [MI["R.ASIS"], MI["L.ASIS"], MI["R.PSIS"], MI["L.PSIS"]]

# Winter-style segment mass fractions, renormalised over the segments that the
# supplied markers actually cover. Literature constants, not fitted numbers.
SEG_COM = [
    (0.497, PELVIS4),                                      # pelvis + trunk + head + arms
    (0.100, [MI["R.GTR"], MI["R.Knee"]]),
    (0.100, [MI["L.GTR"], MI["L.Knee"]]),
    (0.0465, [MI["R.Knee"], MI["R.Ankle"]]),
    (0.0465, [MI["L.Knee"], MI["L.Ankle"]]),
    (0.0145, [MI["R.Heel"], MI["R.MT1"], MI["R.MT5"]]),
    (0.0145, [MI["L.Heel"], MI["L.MT1"], MI["L.MT5"]]),
]


# ------------------------------------------------- Savitzky-Golay filtering -
def savgol_coeffs(window: int, order: int, deriv: int) -> np.ndarray:
    """FIR coefficients of a Savitzky-Golay smoother / differentiator.

    Marker-derived accelerations need a low-pass differentiator: plain finite
    differences amplify marker noise by 1/dt^2. A least-squares polynomial fit
    over a short window is the standard biomechanics choice and is exact on
    polynomials up to `order`.
    """
    half = window // 2
    t = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(t, order + 1, increasing=True)
    return np.linalg.pinv(A)[deriv] * math.factorial(deriv)


def sg(x: np.ndarray, window: int, deriv: int, order: int = 3) -> np.ndarray:
    """Apply the SG filter along axis 1 of (N, T, ...) with edge replication."""
    c = savgol_coeffs(window, order, deriv) / DT ** deriv
    half = window // 2
    xp = np.concatenate([np.repeat(x[:, :1], half, axis=1), x,
                         np.repeat(x[:, -1:], half, axis=1)], axis=1)
    out = np.zeros(x.shape, dtype=np.float64)
    for k in range(window):
        out += c[k] * xp[:, k:k + x.shape[1]]
    return out


W_POS, W_VEL, W_ACC = 9, 15, 31          # 60 ms / 100 ms / 207 ms at 150 Hz
PRIOR_CH = [3, 4, 5]                     # the com/sg31 Newtonian estimate channels


def com_proxy(X: np.ndarray) -> np.ndarray:
    tot = sum(w for w, _ in SEG_COM)
    com = np.zeros((X.shape[0], T, 3), dtype=np.float64)
    for w, idx in SEG_COM:
        com += (w / tot) * X[:, :, idx, :].mean(axis=2)
    return com


def mirror_motion(X: np.ndarray) -> np.ndarray:
    """Left/right mirror: swap contralateral markers AND negate the lateral axis."""
    Xm = X[:, :, MIRROR_IDX, :].copy()
    Xm[..., 2] *= -1.0
    return Xm


def mirror_force(Y: np.ndarray) -> np.ndarray:
    """Under that mirror, forward and vertical force are unchanged; lateral flips."""
    Ym = Y.copy()
    Ym[:, 2] *= -1.0
    return Ym


# ------------------------------------------------------ feature extraction --
KEY = ["R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS", "R.GTR", "L.GTR",
       "R.Knee", "L.Knee", "R.Ankle", "L.Ankle", "R.Heel", "L.Heel",
       "R.MT1", "L.MT1", "R.MT5", "L.MT5", "R.TT", "L.TT",
       "R.Iliac.Crest", "L.Iliac.Crest", "R.HF", "L.HF"]
FOOT = ["R.Heel", "L.Heel", "R.MT1", "L.MT1", "R.MT5", "L.MT5",
        "R.Ankle", "L.Ankle", "R.Knee", "L.Knee", "R.TT", "L.TT"]


def frame_features(X: np.ndarray) -> np.ndarray:
    """(N,T,22,3) markers -> (N, C, T) per-frame features, float32.

    Every channel is a local-in-time function of ONE trial's own markers, so no
    information crosses rows or the train/test boundary. These are inputs to the
    trained network, never a hand-built answer: with the network removed nothing
    here produces a force waveform.
    """
    N = X.shape[0]
    Xd = X.astype(np.float64)
    pel = Xd[:, :, PELVIS4, :].mean(axis=2)                  # pelvis centre per frame
    rel = Xd - pel[:, :, None, :]                            # pelvis-local coordinates

    com = com_proxy(Xd)
    com_v = sg(com, W_VEL, 1)
    com_a15, com_a31 = sg(com, W_VEL, 2), sg(com, W_ACC, 2)
    pel_v = sg(pel, W_VEL, 1)
    pel_a15, pel_a31 = sg(pel, W_VEL, 2), sg(pel, W_ACC, 2)

    f = []
    # (1) the Newtonian estimate itself, in the exact units of the target.
    #     Channels 3,4,5 (com_a31) are also used as the additive residual base.
    for a in (com_a15, com_a31, pel_a15, pel_a31):
        f += [a[:, :, 0] / G, a[:, :, 1] / G + 1.0, a[:, :, 2] / G]
    # (2) COM / pelvis kinematics
    f += [com_v[:, :, k] for k in range(3)]
    f += [pel_v[:, :, k] for k in range(3)]
    f += [com[:, :, 1], pel[:, :, 1],
          np.linalg.norm(com_v[:, :, [0, 2]], axis=-1)]
    # (3) marker positions in the moving pelvis frame (translation invariant)
    relp = sg(rel.reshape(N, T, -1), W_POS, 0).reshape(N, T, N_MARKERS, 3)
    for m in (MI[k] for k in KEY):
        f += [relp[:, :, m, k] for k in range(3)]
    # (4) foot/shank velocity and acceleration in the lab frame: these carry the
    #     impact transients that create the force peaks the metric scores
    mv = sg(Xd.reshape(N, T, -1), W_VEL, 1).reshape(N, T, N_MARKERS, 3)
    ma = sg(Xd.reshape(N, T, -1), W_VEL, 2).reshape(N, T, N_MARKERS, 3)
    for m in (MI[k] for k in FOOT):
        for k in range(3):
            f += [mv[:, :, m, k], ma[:, :, m, k] / G]
    # (5) explicit contact cues -- what a biomechanist reads for heel-strike and
    #     toe-off. The window often starts or ends off the instrumented area, so
    #     the network must learn a contact envelope as well as a waveform.
    for nm in ["R.Heel", "L.Heel", "R.MT1", "L.MT1", "R.MT5", "L.MT5"]:
        m = MI[nm]
        h = Xd[:, :, m, 1]
        f += [h - h.min(axis=1, keepdims=True),
              np.linalg.norm(mv[:, :, m, :], axis=-1),
              rel[:, :, m, 0]]
    f += [Xd[:, :, MI["R.Heel"], 0] - Xd[:, :, MI["L.Heel"], 0],
          Xd[:, :, MI["R.MT1"], 0] - Xd[:, :, MI["L.MT1"], 0],
          Xd[:, :, MI["R.Heel"], 1] - Xd[:, :, MI["L.Heel"], 1],
          Xd[:, :, MI["R.Ankle"], 2] - Xd[:, :, MI["L.Ankle"], 2]]

    # (6) joint angles in cosine form (smooth and bounded, no arccos)
    def ang(a, b, c):
        u = Xd[:, :, MI[a], :] - Xd[:, :, MI[b], :]
        v = Xd[:, :, MI[c], :] - Xd[:, :, MI[b], :]
        return (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)

    for trip in [("R.ASIS", "R.GTR", "R.Knee"), ("L.ASIS", "L.GTR", "L.Knee"),
                 ("R.GTR", "R.Knee", "R.Ankle"), ("L.GTR", "L.Knee", "L.Ankle"),
                 ("R.Knee", "R.Ankle", "R.MT1"), ("L.Knee", "L.Ankle", "L.MT1")]:
        f.append(ang(*trip))
    return np.stack(f, axis=1).astype(np.float32)


RIGID = [("R.ASIS", "L.ASIS"), ("R.ASIS", "R.PSIS"), ("L.ASIS", "L.PSIS"),
         ("R.Iliac.Crest", "L.Iliac.Crest"), ("R.GTR", "L.GTR"),
         ("R.GTR", "R.Knee"), ("L.GTR", "L.Knee"), ("R.Knee", "R.Ankle"),
         ("L.Knee", "L.Ankle"), ("R.TT", "R.Ankle"), ("L.TT", "L.Ankle"),
         ("R.Heel", "R.MT1"), ("L.Heel", "L.MT1"), ("R.Heel", "R.MT5"),
         ("L.Heel", "L.MT5"), ("R.MT1", "R.MT5"), ("L.MT1", "L.MT5"),
         ("R.ASIS", "R.GTR"), ("L.ASIS", "L.GTR"), ("R.Knee", "R.HF"),
         ("L.Knee", "L.HF"), ("R.ASIS", "R.Knee"), ("L.ASIS", "L.Knee"),
         ("R.PSIS", "R.Iliac.Crest"), ("L.PSIS", "L.Iliac.Crest")]


def segment_lengths(X: np.ndarray) -> np.ndarray:
    """Median-over-time rigid segment lengths: a per-trial body-size fingerprint."""
    Xd = X.astype(np.float64)
    out = np.empty((len(X), len(RIGID)), dtype=np.float64)
    for j, (m1, m2) in enumerate(RIGID):
        d = np.linalg.norm(Xd[:, :, MI[m1], :] - Xd[:, :, MI[m2], :], axis=-1)
        out[:, j] = np.median(d, axis=1)
    return out


def trial_features(X: np.ndarray) -> np.ndarray:
    """(N, A) per-trial scalars: body proportions and gait summary.

    Person-level conditioning. Because the force target is mass-normalised, what
    matters is the walker's mass DISTRIBUTION (proportions) and gait, not mass.
    """
    Xd = X.astype(np.float64)
    S = segment_lengths(X)

    def s(m1, m2):
        return S[:, RIGID.index((m1, m2))]

    thigh = 0.5 * (s("R.GTR", "R.Knee") + s("L.GTR", "L.Knee"))
    shank = 0.5 * (s("R.Knee", "R.Ankle") + s("L.Knee", "L.Ankle"))
    foot = 0.5 * (s("R.Heel", "R.MT1") + s("L.Heel", "L.MT1"))
    pelw, gtrw = s("R.ASIS", "L.ASIS"), s("R.GTR", "L.GTR")
    crestw = s("R.Iliac.Crest", "L.Iliac.Crest")
    pdep = 0.5 * (s("R.ASIS", "R.PSIS") + s("L.ASIS", "L.PSIS"))
    leg = thigh + shank

    pel = Xd[:, :, PELVIS4, :].mean(axis=2)
    footy = np.minimum(Xd[:, :, MI["R.Heel"], 1], Xd[:, :, MI["L.Heel"], 1])
    pelh = np.median(pel[:, :, 1] - footy, axis=1)

    com = com_proxy(Xd)
    cv, ca = sg(com, W_VEL, 1), sg(com, W_VEL, 2)
    speed = np.linalg.norm(cv[:, :, [0, 2]], axis=-1).mean(axis=1)

    # cadence from the dominant frequency of vertical pelvis motion (one peak
    # per step). Computed per trial from its own signal.
    v = pel[:, :, 1] - pel[:, :, 1].mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(v * np.hanning(T)[None, :], axis=1))
    frq = np.fft.rfftfreq(T, DT)
    band = (frq > 0.6) & (frq < 4.0)
    cad = frq[band][np.argmax(spec[:, band], axis=1)]

    cols = [thigh, shank, foot, pelw, gtrw, crestw, pdep, leg, pelh,
            thigh / leg, shank / leg, foot / leg, pelw / leg, gtrw / leg, pelh / leg,
            speed, cad, speed / np.maximum(leg, 1e-6), speed / np.maximum(cad, 1e-6),
            np.abs(ca).mean(axis=(1, 2)) / G,
            ca[:, :, 1].std(axis=1) / G, ca[:, :, 0].std(axis=1) / G,
            ca[:, :, 2].std(axis=1) / G,
            np.ptp(pel[:, :, 1], axis=1),
            np.ptp(Xd[:, :, MI["R.Heel"], 1], axis=1),
            np.ptp(Xd[:, :, MI["L.Heel"], 1], axis=1),
            Xd[:, -1, :, 0].mean(axis=1) - Xd[:, 0, :, 0].mean(axis=1)]
    return np.stack(cols, axis=1).astype(np.float32)


# --------------------------------------------------------- person grouping --
def derive_person_groups(S: np.ndarray, n_people: int = N_PEOPLE) -> np.ndarray:
    """Cluster TRAIN trials into approximate persons from segment lengths.

    The description says the hidden split is person-disjoint, that 32 people form
    the training set, and that the person key is an HMAC-SHA-256 digest under an
    evaluator-only key -- so person identity is deliberately NOT recoverable from
    the ids, and no attempt is made to recover it. Segment lengths are a
    deterministic function of the public training markers and are near-constant
    within a walker, which makes them the legitimate route to person groups.
    These groups are used ONLY to make the in-script validation person-disjoint
    (random trial-level validation would place near-replicate trials of the same
    walker on both sides and report an optimistic number). They are never a
    model feature and are never computed for test rows.

    Average-linkage agglomerative clustering, implemented directly so the result
    is deterministic and inspectable.
    """
    Z = (S - S.mean(0)) / (S.std(0) + 1e-12)
    n = len(Z)
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    # active clusters as index lists; merge the closest pair by average linkage
    members = {i: [i] for i in range(n)}
    Dm = D.copy()
    while len(members) > n_people:
        keys = sorted(members)
        sub = Dm[np.ix_(keys, keys)]
        a, b = np.unravel_index(np.argmin(sub), sub.shape)
        ka, kb = keys[a], keys[b]
        na, nb = len(members[ka]), len(members[kb])
        # average-linkage update of the merged row/column
        newd = (na * Dm[ka, :] + nb * Dm[kb, :]) / (na + nb)
        members[ka] = members[ka] + members[kb]
        del members[kb]
        Dm[ka, :] = newd
        Dm[:, ka] = newd
        Dm[ka, ka] = np.inf
        Dm[kb, :] = np.inf
        Dm[:, kb] = np.inf
    lab = np.empty(n, dtype=np.int64)
    for c, (_, idx) in enumerate(sorted(members.items())):
        lab[idx] = c
    return lab


# ----------------------------------------------------------------- model ----
class ResBlock(nn.Module):
    def __init__(self, c: int, dil: int):
        super().__init__()
        pad = dil * (KERNEL - 1) // 2
        self.c1 = nn.Conv1d(c, c, KERNEL, padding=pad, dilation=dil)
        self.n1 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv1d(c, c, KERNEL, padding=pad, dilation=dil)
        self.n2 = nn.GroupNorm(8, c)
        self.do = nn.Dropout(DROPOUT)

    def forward(self, x):
        h = Fn.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(self.do(h)))
        return Fn.gelu(x + h)


class ForceTCN(nn.Module):
    """Dilated temporal CNN: (B,C,T) features + (B,A) scalars -> (B,3,T) force.

    Why a dilated conv stack rather than a transformer: 1,093 trials from 32
    people is a small sample, and the target is a local-in-time physical map
    (force at t is COM acceleration at t) plus a medium-range gait-phase
    context. A conv stack has exactly that inductive bias and is
    translation-equivariant along time, which is correct here because the
    256-frame window is an arbitrary 1.7 s cut of a longer trial.
    Dilations 1,2,4,8,16,32 over 6 two-conv blocks give a receptive field of
    1 + 4*(1+2+4+8+16+32) = 253 frames, i.e. essentially the whole window.
    """

    def __init__(self, c_in: int, n_scalar: int):
        super().__init__()
        self.stem = nn.Conv1d(c_in + n_scalar, WIDTH, KERNEL, padding=KERNEL // 2)
        self.sn = nn.GroupNorm(8, WIDTH)
        self.blocks = nn.ModuleList([ResBlock(WIDTH, min(2 ** i, 32)) for i in range(BLOCKS)])
        self.head = nn.Sequential(nn.Conv1d(WIDTH, WIDTH, 1), nn.GELU(),
                                  nn.Conv1d(WIDTH, 3, 1))
        # zero-init the output layer so training starts exactly at the Newtonian
        # prior and the network only ever learns the correction to it
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, xf, xs, prior):
        x = torch.cat([xf, xs[:, :, None].expand(-1, -1, xf.shape[2])], dim=1)
        h = Fn.gelu(self.sn(self.stem(x)))
        for b in self.blocks:
            h = b(h)
        return self.head(h) + prior          # learned correction on the physics base


def metric_loss(pred, truth, den, huber: float = 0.0):
    """The metric's own error term, used directly as the training loss.

    Official metric:  E_ia = sum_t|p-y| / sum_t|y| ;  score = mean_ia max(0,1-E_ia)
    This loss       :  mean_ia [ sum_t|p-y| / sum_t|y| ]  ==  1 - score (pre-clip)
    so minimising it maximises the official score, and it balances the axes
    exactly the way the metric does. `den` is a label-derived per-trial-axis
    weight used at training time only; inference never needs it.
    """
    r = pred - truth
    if huber > 0:
        a = r.abs()
        r = torch.where(a < huber, 0.5 * r * r / huber + 0.5 * huber, a)
    else:
        r = r.abs()
    return (r.sum(dim=2) / den.clamp_min(1e-6)).mean()


def train_fold(FFtr, SStr, Ytr, PRtr, seed: int, tag: str):
    """Train one model with the FIXED plan. No clock, no early exit, no fallback."""
    seed_everything(seed)
    model = ForceTCN(FFtr.shape[1], SStr.shape[1]).to(DEVICE)
    Xf = torch.from_numpy(FFtr)
    Xs = torch.from_numpy(SStr)
    Yt = torch.from_numpy(Ytr.astype(np.float32))
    Pr = torch.from_numpy(PRtr.astype(np.float32))
    den = Yt.abs().sum(dim=2)

    n = Xf.shape[0]
    steps = ((n + BATCH - 1) // BATCH) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=steps, pct_start=0.15,
        div_factor=10.0, final_div_factor=100.0)
    gen = torch.Generator().manual_seed(seed)

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for st in range(0, n, BATCH):
            b = perm[st:st + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = metric_loss(model(Xf[b], Xs[b], Pr[b]), Yt[b], den[b], HUBER)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(b)
        if ep % 7 == 0 or ep == EPOCHS - 1:
            log(f"    {tag} epoch {ep:3d}/{EPOCHS}  train normalised-L1 {tot / n:.4f}")
    model.eval()
    return model


def predict(model, FF, SS, PR, batch: int = 64) -> np.ndarray:
    """Per-row inference. Each row's output depends only on that row's own
    features and the frozen model -- no statistic is pooled across rows."""
    out = np.empty((len(FF), 3, T), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(FF), batch):
            e = min(s + batch, len(FF))
            out[s:e] = model(torch.from_numpy(FF[s:e]),
                             torch.from_numpy(SS[s:e]),
                             torch.from_numpy(PR[s:e].astype(np.float32))).numpy()
    return out


# ---------------------------------------------------------------- scoring ---
def axis_skill(pred, truth) -> float:
    """The official per-axis skill, transcribed from the challenge grader."""
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    baseline = np.abs(truth).sum()
    err = np.abs(pred - truth).sum()
    return float(err == 0) if baseline == 0 else max(0.0, 1.0 - err / baseline)


def official_score(pred, truth):
    """Mean of axis_skill over x/y/z, then over trials. Returns (total, per-axis)."""
    per = np.empty((len(pred), 3))
    for i in range(len(pred)):
        for a in range(3):
            per[i, a] = axis_skill(pred[i, a], truth[i, a])
    return per.mean(), per.mean(axis=0), per


# ------------------------------------------------------------- validation ---
def validate_submission(sub: pd.DataFrame, sample_path: Path) -> None:
    """Hard gate. Raises before anything is written if the grammar is wrong.

    A malformed CSV ranks BELOW a score of zero and still costs a credit, and a
    wrong header / row count / duplicate or unknown id is a global structural
    error that raises in the grader, so every one of those is checked here.
    """
    sample = pd.read_csv(sample_path, keep_default_na=False)
    assert list(sub.columns) == list(sample.columns), \
        f"columns {list(sub.columns)} != {list(sample.columns)}"
    assert len(sub) == len(sample) == 322, f"rows {len(sub)} != {len(sample)} (expected 322)"
    assert sub["sample_id"].is_unique, "duplicate sample_id"
    assert sub["sample_id"].astype(str).tolist() == sample["sample_id"].astype(str).tolist(), \
        "sample_id set/order differs from sample_submission.csv"
    for col in ["force_x", "force_y", "force_z"]:
        for i, cell in enumerate(sub[col].tolist()):
            assert isinstance(cell, str) and len(cell) > 0, f"empty cell {col} row {i}"
            arr = json.loads(cell)                      # must be valid JSON
            assert isinstance(arr, list) and len(arr) == T, \
                f"{col} row {i}: length {len(arr)} != {T}"
            v = np.asarray(arr, dtype=float)
            assert np.isfinite(v).all(), f"{col} row {i}: non-finite value"
    log("submission validated: 322 rows x 3 axes x 256 finite values, ids in sample order")


def to_json_row(v: np.ndarray) -> str:
    """Fixed 6-decimal JSON array: compact, finite, no scientific notation."""
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


# -------------------------------------------------------------------- main --
def main() -> None:
    seed_everything()
    log(f"public_dir={PUBLIC_DIR}  submission_out={SUBMISSION_OUT}  device={DEVICE}")

    for f in ["train.csv", "test.csv", "train_targets.csv", "sample_submission.csv"]:
        assert (PUBLIC_DIR / f).exists(), f"missing required file {f}"

    tr_man = pd.read_csv(PUBLIC_DIR / "train.csv")
    te_man = pd.read_csv(PUBLIC_DIR / "test.csv")
    tg = pd.read_csv(PUBLIC_DIR / "train_targets.csv").set_index("sample_id")
    sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv", keep_default_na=False)
    log(f"train trials {len(tr_man)}  test trials {len(te_man)}")

    # ---- load the motion arrays named by each manifest (and nothing else) ----
    def load_split(man):
        A = np.empty((len(man), T, N_MARKERS, 3), dtype=np.float32)
        for k, rel in enumerate(man["motion_file"]):
            a = np.load(PUBLIC_DIR / rel)
            assert a.shape == (T, N_MARKERS, 3), f"{rel}: unexpected shape {a.shape}"
            A[k] = a
        return A

    Xtr, Xte = load_split(tr_man), load_split(te_man)
    assert np.isfinite(Xtr).all() and np.isfinite(Xte).all(), "non-finite marker data"

    Ytr = np.empty((len(tr_man), 3, T), dtype=np.float64)
    for k, sid in enumerate(tr_man["sample_id"]):
        for a, col in enumerate(["force_x", "force_y", "force_z"]):
            Ytr[k, a] = json.loads(tg.at[sid, col])
    assert np.isfinite(Ytr).all(), "non-finite target data"
    log(f"loaded motion {Xtr.shape} / {Xte.shape} and targets {Ytr.shape}")

    # ---- features (built per split; no statistic crosses the split) ---------
    FFtr, SStr = frame_features(Xtr), trial_features(Xtr)
    FFte, SSte = frame_features(Xte), trial_features(Xte)
    # physically exact left/right mirror as the one augmentation: swapping the
    # contralateral markers and negating the lateral axis maps a real walking
    # trial to another real walking trial, with force_x/force_y unchanged and
    # force_z negated. It doubles the data without fabricating anything.
    Xm = mirror_motion(Xtr)
    FFmr, SSmr = frame_features(Xm), trial_features(Xm)
    Ymr = mirror_force(Ytr)
    assert np.allclose(Ymr[:, 0], Ytr[:, 0]) and np.allclose(Ymr[:, 1], Ytr[:, 1]) \
        and np.allclose(Ymr[:, 2], -Ytr[:, 2]), "mirror is not the expected symmetry"
    log(f"features: frame {FFtr.shape[1]} channels, trial scalars {SStr.shape[1]}")

    PRtr = np.ascontiguousarray(FFtr[:, PRIOR_CH, :])    # Newtonian residual base
    PRte = np.ascontiguousarray(FFte[:, PRIOR_CH, :])
    PRmr = np.ascontiguousarray(FFmr[:, PRIOR_CH, :])
    del Xm, Xte            # raw marker arrays are dead once features exist

    # ---- person-disjoint in-script validation (diagnostic, fixed plan) ------
    lab = derive_person_groups(segment_lengths(Xtr), N_PEOPLE)
    sizes = np.bincount(lab)
    log(f"derived {len(sizes)} person groups, sizes min {sizes.min()} "
        f"max {sizes.max()} median {int(np.median(sizes))}")
    # hold out whole persons, largest-first into the holdout until HOLDOUT_PEOPLE
    order = np.argsort(-sizes)
    hold_people = set(order[:HOLDOUT_PEOPLE].tolist())   # whole persons, train rows only
    va = np.array([i for i in range(len(lab)) if lab[i] in hold_people])
    trn = np.array([i for i in range(len(lab)) if lab[i] not in hold_people])
    log(f"validation sets aside {len(hold_people)} whole persons "
        f"({len(va)} trials); training on {len(trn)}")

    # standardisation fitted on the VALIDATION-FOLD TRAINING ROWS only
    def fit_std(FF, SS, idx):
        fm = FF[idx].mean(axis=(0, 2), keepdims=True).astype(np.float32)
        fs = (FF[idx].std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
        sm = SS[idx].mean(axis=0, keepdims=True).astype(np.float32)
        ss = (SS[idx].std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        return fm, fs, sm, ss

    fm, fs, sm, ss = fit_std(FFtr, SStr, trn)
    mdl = train_fold(
        np.concatenate([(FFtr[trn] - fm) / fs, (FFmr[trn] - fm) / fs]),
        np.concatenate([(SStr[trn] - sm) / ss, (SSmr[trn] - sm) / ss]),
        np.concatenate([Ytr[trn], Ymr[trn]]),
        np.concatenate([PRtr[trn], PRmr[trn]]),
        seed=SEED, tag="holdout")
    vp = predict(mdl, (FFtr[va] - fm) / fs, (SStr[va] - sm) / ss, PRtr[va])
    vtot, vax, vper = official_score(vp, Ytr[va])
    log(f"person-disjoint TRAIN validation, official metric: TOTAL {vtot:.4f}  "
        f"x {vax[0]:.4f}  y {vax[1]:.4f}  z {vax[2]:.4f}")
    pp = sorted(float(vper[lab[va] == c].mean()) for c in hold_people)
    log(f"  per-unseen-person scores: {' '.join(f'{v:.3f}' for v in pp)}")
    # a trained model must beat the best constant waveform, which is the
    # L1-optimal median of the training waveforms; this guard proves the network
    # is doing the work rather than reproducing a template.
    med = np.median(Ytr[trn], axis=0)
    mtot, _, _ = official_score(np.repeat(med[None], len(va), axis=0), Ytr[va])
    log(f"  reference: best constant (median) waveform scores {mtot:.4f} "
        f"-> model adds {vtot - mtot:+.4f}")
    assert vtot > mtot, "model does not beat the constant-waveform baseline"

    # ---- final fit on 100% of train, N_SEEDS averaged ----------------------
    fm, fs, sm, ss = fit_std(FFtr, SStr, np.arange(len(FFtr)))
    # Standardise in place and preallocate the doubled array, so the peak
    # footprint is one copy rather than the original plus two temporaries.
    n1 = len(FFtr)
    FFa = np.empty((2 * n1,) + FFtr.shape[1:], dtype=np.float32)
    np.subtract(FFtr, fm, out=FFtr); np.divide(FFtr, fs, out=FFtr)
    np.subtract(FFmr, fm, out=FFmr); np.divide(FFmr, fs, out=FFmr)
    FFa[:n1] = FFtr; FFa[n1:] = FFmr
    del FFmr
    SSa = np.concatenate([(SStr - sm) / ss, (SSmr - sm) / ss]).astype(np.float32)
    Ya = np.concatenate([Ytr, Ymr])
    PRa = np.concatenate([PRtr, PRmr])
    FFteS, SSteS = (FFte - fm) / fs, (SSte - sm) / ss

    preds = np.zeros((len(FFte), 3, T), dtype=np.float64)
    for si in range(N_SEEDS):
        log(f"  full-data refit, seed {si + 1}/{N_SEEDS}")
        m = train_fold(FFa, SSa, Ya, PRa, seed=SEED + 101 * si, tag=f"full{si}")
        preds += predict(m, FFteS, SSteS, PRte) / N_SEEDS
    log(f"averaged {N_SEEDS} full-data models")

    # ---- sanity: test predictions must look like the training targets -------
    log(f"  prediction stats   force_x mean {preds[:,0].mean():+.4f} sd {preds[:,0].std():.4f}"
        f" | train mean {Ytr[:,0].mean():+.4f} sd {Ytr[:,0].std():.4f}")
    log(f"                     force_y mean {preds[:,1].mean():+.4f} sd {preds[:,1].std():.4f}"
        f" | train mean {Ytr[:,1].mean():+.4f} sd {Ytr[:,1].std():.4f}")
    log(f"                     force_z mean {preds[:,2].mean():+.4f} sd {preds[:,2].std():.4f}"
        f" | train mean {Ytr[:,2].mean():+.4f} sd {Ytr[:,2].std():.4f}")
    assert np.isfinite(preds).all(), "non-finite predictions"
    assert preds.std() > 1e-6, "degenerate constant predictions"

    # ---- build, validate, then write (never write a broken file) -----------
    rows = {"sample_id": te_man["sample_id"].tolist()}
    for a, col in enumerate(["force_x", "force_y", "force_z"]):
        rows[col] = [to_json_row(preds[i, a]) for i in range(len(preds))]
    sub = pd.DataFrame(rows)[list(sample.columns)]
    # the grader ignores row order, but matching sample_submission.csv exactly
    # removes any doubt about ids
    sub = sub.set_index("sample_id").loc[sample["sample_id"]].reset_index()
    validate_submission(sub, PUBLIC_DIR / "sample_submission.csv")

    tmp = SUBMISSION_OUT.with_suffix(".csv.tmp")
    sub.to_csv(tmp, index=False)
    os.replace(tmp, SUBMISSION_OUT)
    back = pd.read_csv(SUBMISSION_OUT, keep_default_na=False)
    validate_submission(back, PUBLIC_DIR / "sample_submission.csv")
    log(f"wrote {SUBMISSION_OUT} shape={back.shape}")


if __name__ == "__main__":
    main()
