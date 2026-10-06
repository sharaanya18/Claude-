"""Per-frame kinematic feature construction + the sampling-rate / physics check.

Physics that drives the design
------------------------------
Newton on the whole body:  sum(F_ext) = m * a_com, with F_ext = GRF + gravity.
Vertically:   GRF_y = m*a_com_y + m*g
Horizontally: GRF_x = m*a_com_x,   GRF_z = m*a_com_z
The target is normalised by m*g, so in body-weight units

    f_y = a_com_y/g + 1,    f_x = a_com_x/g,    f_z = a_com_z/g

MASS CANCELS. That is why the task can generalise to unseen walkers at all:
the target is a pure kinematic quantity (COM acceleration in units of g).
The difficulty is that only the LOWER body is instrumented, so a_com of the
whole body (trunk/arms/head carry most of the mass) must be inferred from
pelvis + leg kinematics -- that inference is what the model has to learn.

Two further things the model must learn, both visible in the audit:
 * the walker is not always on the instrumented area (9.8% of frames read
   ~0 N), so there is a contact/no-contact envelope on top of the waveform;
 * the absolute lab position is removed (origin = frame-0 ASIS midpoint), so
   where the plates are is not directly observable -> irreducible uncertainty.
"""
from __future__ import annotations

import math

import numpy as np

from kp_data import MI, MIRROR, N_MARKERS, T

FS = 150.0                 # Hz; 256 frames / 1.7 s = 150.6 -> 150 Hz source rate
DT = 1.0 / FS
G = 9.80665

PELVIS4 = [MI["R.ASIS"], MI["L.ASIS"], MI["R.PSIS"], MI["L.PSIS"]]
# crude segment-mass proxies for a lower-body COM estimate (Winter-style
# fractions, renormalised over the segments we can see). These are PHYSICAL
# constants from the biomechanics literature, not tuned numbers.
SEG_COM = [
    # (weight, [markers whose centroid proxies that segment's COM])
    (0.497, PELVIS4),                                   # pelvis+trunk+head+arms
    (0.100, [MI["R.GTR"], MI["R.Knee"]]),               # right thigh
    (0.100, [MI["L.GTR"], MI["L.Knee"]]),               # left thigh
    (0.0465, [MI["R.Knee"], MI["R.Ankle"]]),            # right shank
    (0.0465, [MI["L.Knee"], MI["L.Ankle"]]),            # left shank
    (0.0145, [MI["R.Heel"], MI["R.MT1"], MI["R.MT5"]]), # right foot
    (0.0145, [MI["L.Heel"], MI["L.MT1"], MI["L.MT5"]]), # left foot
]


def savgol_coeffs(window: int, order: int, deriv: int) -> np.ndarray:
    """Savitzky-Golay FIR coefficients (deriv-th derivative, per sample)."""
    half = window // 2
    t = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(t, order + 1, increasing=True)
    # pseudo-inverse row `deriv` gives the deriv-th Taylor coefficient
    c = np.linalg.pinv(A)[deriv] * math.factorial(deriv)
    return c


def sg_filter(x: np.ndarray, window: int, order: int, deriv: int, dt: float) -> np.ndarray:
    """Apply Savitzky-Golay along axis 1 of (N, T, ...) with edge replication.

    Deterministic FIR; no library edge-mode surprises. Derivative in units of
    (unit of x) / s**deriv.
    """
    c = savgol_coeffs(window, order, deriv) / dt ** deriv
    half = window // 2
    xp = np.concatenate([np.repeat(x[:, :1], half, axis=1), x,
                         np.repeat(x[:, -1:], half, axis=1)], axis=1)
    out = np.zeros_like(x, dtype=np.float64)
    for k in range(window):
        out += c[k] * xp[:, k:k + x.shape[1]]
    return out


def com_proxy(X: np.ndarray) -> np.ndarray:
    """Weighted lower-body COM estimate, (N,T,3) metres."""
    tot = sum(w for w, _ in SEG_COM)
    com = np.zeros((X.shape[0], T, 3), dtype=np.float64)
    for w, idx in SEG_COM:
        com += (w / tot) * X[:, :, idx, :].mean(axis=2)
    return com


def mirror(X: np.ndarray) -> np.ndarray:
    """Left/right mirror: swap contralateral markers AND negate the lateral (z) axis."""
    Xm = X[:, :, MIRROR, :].copy()
    Xm[..., 2] *= -1.0
    return Xm


def mirror_targets(Y: np.ndarray) -> np.ndarray:
    """Under a left/right mirror, force_x and force_y are unchanged, force_z flips."""
    Ym = Y.copy()
    Ym[:, 2] *= -1.0
    return Ym


# --------------------------------------------------------------- features ---
# Smoothing windows: 15 frames = 100 ms at 150 Hz, the standard window for
# marker-derived accelerations; 31 frames = 207 ms gives a second, smoother view.
W_POS, W_VEL, W_ACC = 9, 15, 31
ORDER = 3

KEY = ["R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS", "R.GTR", "L.GTR",
       "R.Knee", "L.Knee", "R.Ankle", "L.Ankle", "R.Heel", "L.Heel",
       "R.MT1", "L.MT1", "R.MT5", "L.MT5", "R.TT", "L.TT",
       "R.Iliac.Crest", "L.Iliac.Crest", "R.HF", "L.HF"]
KEYI = [MI[k] for k in KEY]


def frame_features(X: np.ndarray) -> np.ndarray:
    """(N,T,22,3) -> (N, C, T) float32 per-frame features.

    Everything here is a local-in-time function of one trial's own markers.
    No cross-trial and no cross-split statistic is involved.
    """
    N = X.shape[0]
    Xd = X.astype(np.float64)

    # pelvis frame: origin at the pelvis centroid of EACH frame, so features are
    # invariant to where in the lab the walker is (absolute position is already
    # partly removed by the challenge, this removes the rest).
    pel = Xd[:, :, PELVIS4, :].mean(axis=2)                      # (N,T,3)
    rel = Xd - pel[:, :, None, :]                                # (N,T,22,3)

    com = com_proxy(Xd)
    com_v = sg_filter(com, W_VEL, ORDER, 1, DT)                  # m/s
    com_a15 = sg_filter(com, W_VEL, ORDER, 2, DT)                # m/s^2
    com_a31 = sg_filter(com, W_ACC, ORDER, 2, DT)                # smoother view
    pel_v = sg_filter(pel, W_VEL, ORDER, 1, DT)
    pel_a15 = sg_filter(pel, W_VEL, ORDER, 2, DT)
    pel_a31 = sg_filter(pel, W_ACC, ORDER, 2, DT)

    feats = []

    # (1) the direct Newtonian estimate, in the exact units of the target.
    #     This is the physics prior handed to the network as a channel.
    for a in (com_a15, com_a31, pel_a15, pel_a31):
        feats.append(a[:, :, 0] / G)             # -> f_x
        feats.append(a[:, :, 1] / G + 1.0)       # -> f_y
        feats.append(a[:, :, 2] / G)             # -> f_z

    # (2) COM / pelvis kinematics
    feats += [com_v[:, :, k] for k in range(3)]
    feats += [pel_v[:, :, k] for k in range(3)]
    feats += [com[:, :, 1], pel[:, :, 1]]                        # heights
    feats.append(np.linalg.norm(com_v[:, :, [0, 2]], axis=-1))   # horizontal speed

    # (3) per-marker position in the moving pelvis frame (smoothed)
    relp = sg_filter(rel.reshape(N, T, -1), W_POS, ORDER, 0, DT).reshape(N, T, N_MARKERS, 3)
    for m in KEYI:
        for k in range(3):
            feats.append(relp[:, :, m, k])

    # (4) per-marker velocity and acceleration in the LAB frame for the feet and
    #     shanks (these carry the impact transients that drive the force peaks)
    FOOT = [MI[k] for k in ["R.Heel", "L.Heel", "R.MT1", "L.MT1", "R.MT5", "L.MT5",
                            "R.Ankle", "L.Ankle", "R.Knee", "L.Knee", "R.TT", "L.TT"]]
    mv = sg_filter(Xd.reshape(N, T, -1), W_VEL, ORDER, 1, DT).reshape(N, T, N_MARKERS, 3)
    ma = sg_filter(Xd.reshape(N, T, -1), W_VEL, ORDER, 2, DT).reshape(N, T, N_MARKERS, 3)
    for m in FOOT:
        for k in range(3):
            feats.append(mv[:, :, m, k])
            feats.append(ma[:, :, m, k] / G)

    # (5) explicit contact cues: foot height above its own trial minimum, foot
    #     speed, and distance from the pelvis along travel. These are what a
    #     biomechanist reads to find heel-strike / toe-off.
    for nm in ["R.Heel", "L.Heel", "R.MT1", "L.MT1", "R.MT5", "L.MT5"]:
        m = MI[nm]
        h = Xd[:, :, m, 1]
        feats.append(h - h.min(axis=1, keepdims=True))            # height over own min
        feats.append(np.linalg.norm(mv[:, :, m, :], axis=-1))     # speed
        feats.append(rel[:, :, m, 0])                             # fore-aft vs pelvis
    # feet together/apart and left-right load-sharing geometry
    feats.append(Xd[:, :, MI["R.Heel"], 0] - Xd[:, :, MI["L.Heel"], 0])
    feats.append(Xd[:, :, MI["R.MT1"], 0] - Xd[:, :, MI["L.MT1"], 0])
    feats.append(Xd[:, :, MI["R.Heel"], 1] - Xd[:, :, MI["L.Heel"], 1])
    feats.append(Xd[:, :, MI["R.Ankle"], 2] - Xd[:, :, MI["L.Ankle"], 2])

    # (6) joint angles (cosine form, no arccos -> smooth and bounded)
    def ang(a, b, c):
        u = Xd[:, :, MI[a], :] - Xd[:, :, MI[b], :]
        v = Xd[:, :, MI[c], :] - Xd[:, :, MI[b], :]
        num = (u * v).sum(-1)
        den = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9
        return num / den

    for (a, b, c) in [("R.ASIS", "R.GTR", "R.Knee"), ("L.ASIS", "L.GTR", "L.Knee"),
                      ("R.GTR", "R.Knee", "R.Ankle"), ("L.GTR", "L.Knee", "L.Ankle"),
                      ("R.Knee", "R.Ankle", "R.MT1"), ("L.Knee", "L.Ankle", "L.MT1")]:
        feats.append(ang(a, b, c))

    F = np.stack(feats, axis=1).astype(np.float32)               # (N, C, T)
    return F


def anthro_features(X: np.ndarray) -> np.ndarray:
    """(N, A) per-trial scalar descriptors: body proportions + gait summary.

    Person-level conditioning. Legitimate: derived from each trial's own markers.
    """
    Xd = X.astype(np.float64)

    def seg(a, b):
        return np.median(np.linalg.norm(Xd[:, :, MI[a], :] - Xd[:, :, MI[b], :], axis=-1), axis=1)

    thigh = 0.5 * (seg("R.GTR", "R.Knee") + seg("L.GTR", "L.Knee"))
    shank = 0.5 * (seg("R.Knee", "R.Ankle") + seg("L.Knee", "L.Ankle"))
    foot = 0.5 * (seg("R.Heel", "R.MT1") + seg("L.Heel", "L.MT1"))
    pelw = seg("R.ASIS", "L.ASIS")
    gtrw = seg("R.GTR", "L.GTR")
    crestw = seg("R.Iliac.Crest", "L.Iliac.Crest")
    pdep = 0.5 * (seg("R.ASIS", "R.PSIS") + seg("L.ASIS", "L.PSIS"))
    leg = thigh + shank
    # pelvis height above the lower foot = standing-ish stature proxy
    pel = Xd[:, :, PELVIS4, :].mean(axis=2)
    footy = np.minimum(Xd[:, :, MI["R.Heel"], 1], Xd[:, :, MI["L.Heel"], 1])
    pelh = np.median(pel[:, :, 1] - footy, axis=1)

    com = com_proxy(Xd)
    cv = sg_filter(com, W_VEL, ORDER, 1, DT)
    ca = sg_filter(com, W_VEL, ORDER, 2, DT)
    speed = np.linalg.norm(cv[:, :, [0, 2]], axis=-1).mean(axis=1)

    # cadence: dominant frequency of vertical pelvis motion (one peak per step)
    v = pel[:, :, 1] - pel[:, :, 1].mean(axis=1, keepdims=True)
    sp = np.abs(np.fft.rfft(v * np.hanning(T)[None, :], axis=1))
    frq = np.fft.rfftfreq(T, DT)
    lo = (frq > 0.6) & (frq < 4.0)
    cad = frq[lo][np.argmax(sp[:, lo], axis=1)]

    cols = [thigh, shank, foot, pelw, gtrw, crestw, pdep, leg, pelh,
            thigh / leg, shank / leg, foot / leg, pelw / leg, gtrw / leg, pelh / leg,
            speed, cad, speed / np.maximum(leg, 1e-6), speed / np.maximum(cad, 1e-6),
            np.abs(ca).mean(axis=(1, 2)) / G,
            ca[:, :, 1].std(axis=1) / G, ca[:, :, 0].std(axis=1) / G, ca[:, :, 2].std(axis=1) / G,
            np.ptp(pel[:, :, 1], axis=1),
            np.ptp(Xd[:, :, MI["R.Heel"], 1], axis=1),
            np.ptp(Xd[:, :, MI["L.Heel"], 1], axis=1),
            (Xd[:, -1, :, 0].mean(axis=1) - Xd[:, 0, :, 0].mean(axis=1)),
            ]
    return np.stack(cols, axis=1).astype(np.float32)
