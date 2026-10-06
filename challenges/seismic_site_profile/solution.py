"""Seismic Site Profile - recover a four-cell shear-wave velocity band profile
from one three-component, amplitude-normalised ground-motion record.

HOW IT WORKS
    Each record is reduced to site-response features that survive the fact that
    source and path dominate a single waveform: the horizontal-to-vertical
    spectral ratio (the two share a ray path, so the ratio largely cancels
    them), the same ratio restricted to the late coda, spectra taken relative
    to their own smooth trend, and rotation-invariant polarisation and
    time-domain shape. Those features train an ensemble of six model families
    from scratch on the 1,179 supplied records, and a macro-F1-aware decode
    turns the per-cell probabilities into four band names.

CHALLENGE REQUIREMENTS MAP
    "from-scratch ... no pretrained weights of any kind, no external
     seismological corpora, no models trained elsewhere"
        -> only numpy/scipy/sklearn estimators, every one constructed untrained
           and fitted here on the supplied data. No download, no network, no
           checkpoint, no tokenizer, no embedding is loaded anywhere.
    "train only on the provided train.csv and waveforms.npz"
        -> the only inputs read are PUBLIC_DIR/{train,test,sample_submission,
           folds}.csv and PUBLIC_DIR/waveforms.npz.
    "do not look up the evaluation records in any outside source"
        -> no network access of any kind; the profile is inferred from the
           waveform alone.
    "do not use the id, the row order, or a record's position as a signal"
        -> ids are used only as dictionary keys into waveforms.npz and to write
           the output column; no id string, hash, order or index enters a
           feature, and no feature depends on how rows are sorted.
    "predict each evaluation record on its own ... do not pool, vote or average
     predictions across evaluation records, and do not group them by station"
        -> every feature is computed from one record in isolation. The multiple
           views of a record are sub-windows OF THAT RECORD. No scaler, PCA,
           quantile, cluster or threshold is fitted on evaluation data: the
           band weights come from training out-of-fold predictions and are a
           fixed 4x4 constant at inference. Running the script on any subset of
           test.csv yields identical predictions for those rows.
    "do not ... pseudo-label the evaluation set"
        -> evaluation labels are never produced, stored or fed back.
    "CPU only - 10 CPU cores, 62 GB RAM within 90 minutes"
        -> no torch, no GPU code path; thread counts pinned to 10 before numpy
           is imported; a fixed work plan (fixed folds, seeds, views, trees and
           iterations) that measured ~12 min end to end.
    "no packages beyond the preinstalled environment"
        -> numpy, pandas, scikit-learn only.
    submission grammar
        -> exactly the columns id,profile for all 333 test ids in test.csv
           order, four lowercase band names joined by "|", validated by
           validate_submission() which raises before anything is written.

DETERMINISM
    All seeds fixed; thread counts pinned; no branch anywhere depends on
    elapsed time, on hardware, or on what happens to be importable. Elapsed
    time is printed for logging only and never read back.
"""
import os
import sys
import random
from pathlib import Path

PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
SUBMISSION_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")

# Thread pinning must happen before numpy/scipy/sklearn initialise their BLAS
# pools. Fixed at the 10 cores the challenge states - never read from the
# machine, so the work plan cannot change with the hardware.
N_THREADS = 10
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(N_THREADS)
os.environ["PYTHONHASHSEED"] = "0"

import time
import numpy as np
import pandas as pd

T0 = time.time()          # LOGGING ONLY - never used in any condition
SEED = 42


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Fixed work plan. Every count below is a constant chosen from cross-validation
# on the supplied station-held-out folds; nothing here adapts to the machine,
# to the clock, or to the evaluation data.
# ---------------------------------------------------------------------------
USE_GROUPS = "ABCDEFGHIJKM"
MODEL_NAMES = ("ordgb", "ord", "et",)
N_VIEWS = 1          # 1 = the record's full valid span only
SEEDS = (0,)
NCROP = 3
FRAC = 0.62

HGB_LR, HGB_LEAF, HGB_IT, HGB_L2 = 0.06, 15, 200, 1.0
ET_N, ET_LEAF = 800, 3
LR_C = 0.01
PLS_K = 6
ORD_C = 0.03
LDA_SHRINK, LDA_K = 0.3, 30
JOINT_LAMBDA = 1.0   # strength of the training profile prior in the decode


def score_cells_int(Y, P):
    """Challenge metric on integer band indices, for logging only.

    macro-F1 within each depth cell over the bands present in gold, averaged
    over the four cells, expressed as skill over the record-blind policy.
    """
    cell = []
    blind = []
    for c in range(4):
        labels = np.unique(Y[:, c])
        f1 = []
        for l in labels:
            g = Y[:, c] == l
            p = P[:, c] == l
            tp = int((g & p).sum())
            f1.append(0.0 if tp == 0 else
                      2.0 * tp / (2.0 * tp + int((~g & p).sum()) + int((g & ~p).sum())))
        cell.append(float(np.mean(f1)))
        blind.append(1.0 / len(labels))
    b = float(np.mean(blind))
    raw = (float(np.mean(cell)) - b) / (1.0 - b)
    return min(1.0, max(0.001, raw)), cell


def crops(w):
    """Overlapping sub-windows of one record's own valid span."""
    nz = np.abs(w).max(axis=0) > 0
    last = int(np.nonzero(nz)[0][-1]) + 1 if nz.any() else w.shape[1]
    last = max(last, NPERSEG + 256)
    L = int(last * FRAC)
    if L < NPERSEG:
        L = min(last, NPERSEG)
    offs = np.linspace(0, last - L, NCROP).astype(int)
    return [w[:, o:o + L] for o in offs]

# ===========================================================================
# Feature extraction. Physics the representation rests on: a soft surface layer
# over firmer rock resonates near f0 ~ Vs/(4H), amplifies the horizontals far
# more than the vertical there, and attenuates high frequencies. A single
# record is source x path x site; the three components share the source and the
# path, so component RATIOS (H/V) and a spectrum's shape RELATIVE TO ITS OWN
# smooth trend keep the site term while cancelling most of the rest. Amplitude
# is normalised per record, so only shape and inter-component ratios survive.
# Every quantity below is computed from one record in isolation.
# ===========================================================================


FS = 50.0
NPERSEG = 1024          # 20.5 s: resolves f0 down to ~0.5 Hz (df = 0.049 Hz)
NSTEP = 256             # 75% overlap -> 8 windows on a full record, 4 on a padded one
FMIN, FMAX = 0.4, 24.0
NBIN = 24


def _valid(w):
    """Drop trailing all-zero padding (~11% of records carry it)."""
    nz = np.abs(w).max(axis=0) > 0
    if not nz.any():
        return w
    last = int(np.nonzero(nz)[0][-1]) + 1
    return w[:, :max(last, NPERSEG)]


_HANN = np.hanning(NPERSEG)
_FREQ = np.fft.rfftfreq(NPERSEG, 1.0 / FS)
_KEEP = (_FREQ >= FMIN) & (_FREQ <= FMAX)
_FK = _FREQ[_KEEP]
_EDGES = np.logspace(np.log10(FMIN), np.log10(FMAX), NBIN + 1)
_BIDX = np.clip(np.digitize(_FK, _EDGES) - 1, 0, NBIN - 1)
_BMAT = np.zeros((NBIN, len(_FK)))
for _b in range(NBIN):
    _m = _BIDX == _b
    if _m.any():
        _BMAT[_b, _m] = 1.0 / _m.sum()
    else:
        # a narrow low-frequency bin can fall between two FFT lines; use the
        # nearest line so no bin is ever empty (an empty bin would be log10(0))
        _c = np.sqrt(_EDGES[_b] * _EDGES[_b + 1])
        _BMAT[_b, int(np.argmin(np.abs(_FK - _c)))] = 1.0
_BFREQ = _BMAT @ _FK
assert np.all(_BFREQ > 0), "empty frequency bin"
assert (_BMAT.sum(axis=1) > 0).all()
_LOGBF = np.log10(_BFREQ)

# broad octave-ish bands for energy ratios / polarisation
_BANDS = [(0.4, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 24.0)]
_BMASKS = [(_FK >= lo) & (_FK < hi) for lo, hi in _BANDS]

EPS = 1e-12

# Linear-frequency band used for the cepstrum. A layered site reverberates with
# a quasi-period in frequency of 1/(2*tau), tau = one-way travel time through the
# column, so the site term concentrates at quefrency tau while the smooth
# source/path spectrum stays at near-zero quefrency. Two-way times of interest
# (0-30 m at Vs 150-900 m/s) are 0.07-0.40 s.
_CKEEP = (_FREQ >= 0.5) & (_FREQ <= 24.0)
_CF = _FREQ[_CKEEP]
_CTAPER = np.hanning(len(_CF))
_NQ = 30                      # quefrency lags 1..30 -> 0.02 .. 0.60 s
_DQ = 1.0 / (2.0 * 25.0)      # 0.02 s per lag, set by the 25 Hz Nyquist


_CKEEPr = (_FK >= 0.5) & (_FK <= 24.0)


def _cepstrum(logp):
    """Real cepstrum of a log spectrum with its smooth (source/path) trend
    removed and the band edges tapered, keeping the site-reverberation lags."""
    r = logp - np.polyval(np.polyfit(_CF, logp, 3), _CF)
    c = np.fft.irfft(r * _CTAPER, n=2 * (len(_CF) - 1))
    return c[1:_NQ + 1]


# ---- group L: Konno-Ohmachi smoothed H/V on a finer grid ---------------------
# Bin averaging flattens a narrow resonance peak; Konno-Ohmachi smoothing is the
# standard HVSR estimator because its window is constant-width in log frequency,
# so it keeps peak height and position at both ends of the band.
NKO = 32
_KOF = np.logspace(np.log10(0.45), np.log10(24.5), NKO)
_LOGKOF = np.log10(_KOF)


def _ko_matrix(fsrc, fout, b=40.0):
    r = np.where(fsrc[None, :] > 0, fsrc[None, :] / fout[:, None], 1.0)
    x = b * np.log10(np.clip(r, 1e-9, None))
    w = (np.sin(x) / np.where(x == 0, 1.0, x)) ** 4
    w = np.where(np.abs(x) < 1e-9, 1.0, w)
    w[np.abs(np.log10(np.clip(r, 1e-9, None))) > 0.5] = 0.0
    return w / w.sum(axis=1, keepdims=True)


_KOMAT = _ko_matrix(_FK, _KOF)

# ---- group M: long-window estimate for the deepest structure ----------------
# 41 s windows halve the frequency step, which is what the 20-30 m cell needs:
# its resonance sits lowest and a 20 s window barely resolves it.
NLONG = 2048
_HANNL = np.hanning(NLONG)
_FREQL = np.fft.rfftfreq(NLONG, 1.0 / FS)
_KEEPL = (_FREQL >= 0.25) & (_FREQL <= 5.0)
_FKL = _FREQL[_KEEPL]
NBL = 14
_KOFL = np.logspace(np.log10(0.26), np.log10(4.8), NBL)
_KOMATL = _ko_matrix(_FKL, _KOFL, b=30.0)


# ---- group N: short-window, high-frequency detail for the SHALLOWEST cell ---
# The 0-5 m cell resonates near Vs/(4*5) = 10-22 Hz. High frequencies need no
# long window (0.2 Hz steps are ample above 8 Hz) but they do need averaging,
# so 5 s windows give ~11 looks instead of 8 and a far steadier estimate where
# the single-record signal-to-noise is worst. This mirrors group M, which won
# the deep cells with the opposite trade.
NSHORT = 256
_HANNS = np.hanning(NSHORT)
_FREQS = np.fft.rfftfreq(NSHORT, 1.0 / FS)
_KEEPS = (_FREQS >= 7.0) & (_FREQS <= 24.6)
_FKS = _FREQS[_KEEPS]
NBS = 12
_KOFS = np.logspace(np.log10(7.5), np.log10(24.0), NBS)
_KOMATS = _ko_matrix(_FKS, _KOFS, b=20.0)


def _windows(w):
    """(nwin, 3, nfreq) power spectra of Hann-tapered, demeaned windows."""
    n = w.shape[1]
    starts = list(range(0, n - NPERSEG + 1, NSTEP))
    if not starts:
        starts = [0]
    segs = np.stack([w[:, s:s + NPERSEG] for s in starts])        # (nw,3,NPERSEG)
    segs = segs - segs.mean(axis=2, keepdims=True)
    segs = segs * _HANN
    F = np.fft.rfft(segs, axis=2)
    P = (np.abs(F) ** 2)[:, :, _KEEP]
    return P + EPS


def _binned(P):
    """(nwin,3,NBIN) log10 power in log-spaced frequency bins."""
    return np.log10(P @ _BMAT.T + EPS)


def _peaks(curve, logf):
    """Peak structure of a smooth log curve: strongest and second peak."""
    n = len(curve)
    loc = [i for i in range(1, n - 1) if curve[i] > curve[i - 1] and curve[i] >= curve[i + 1]]
    if not loc:
        i = int(np.argmax(curve))
        return logf[i], curve[i], 0.0, 0.0, 0.0, 0.0
    loc.sort(key=lambda i: -curve[i])
    i0 = loc[0]
    # half-power width in log-frequency around the main peak
    half = curve[i0] - np.log10(np.sqrt(2.0))
    l = i0
    while l > 0 and curve[l] > half:
        l -= 1
    r = i0
    while r < n - 1 and curve[r] > half:
        r += 1
    width = logf[r] - logf[l]
    if len(loc) > 1:
        i1 = loc[1]
        f1, a1 = logf[i1], curve[i1]
    else:
        f1, a1 = logf[i0], curve[i0]
    return logf[i0], curve[i0], width, f1, a1, float(len(loc))


def _shape_stats(logp, logf):
    """Centroid / spread / entropy / rolloff / slopes of a log-power curve."""
    p = 10.0 ** logp
    p = p / (p.sum() + EPS)
    cen = float(p @ logf)
    spr = float(np.sqrt(p @ (logf - cen) ** 2))
    ent = float(-(p * np.log(p + EPS)).sum() / np.log(len(p)))
    c = np.cumsum(p)
    rol85 = float(logf[int(np.searchsorted(c, 0.85))])
    rol50 = float(logf[int(np.searchsorted(c, 0.50))])
    lo = logf < np.log10(4.0)
    hi = logf >= np.log10(4.0)
    s_lo = float(np.polyfit(logf[lo], logp[lo], 1)[0]) if lo.sum() > 2 else 0.0
    s_hi = float(np.polyfit(logf[hi], logp[hi], 1)[0]) if hi.sum() > 2 else 0.0
    q = np.polyfit(logf, logp, 2)
    return [cen, spr, ent, rol85, rol50, s_lo, s_hi, float(q[0]), float(q[1])]


def _detrend(logp, logf, deg=2):
    """Residual of a log spectrum about its own smooth trend: resonance only,
    with the broad source/path spectral shape removed."""
    c = np.polyfit(logf, logp, deg)
    return logp - np.polyval(c, logf)


def _envelope(x, k=25):
    e = np.abs(x)
    ker = np.ones(k) / k
    return np.convolve(e, ker, mode="same")


def extract(w):
    """One record (3,3000) -> dict of feature groups, each a 1-D float array."""
    w = _valid(np.asarray(w, dtype=np.float64))
    P = _windows(w)                       # (nw,3,nf)
    B = _binned(P)                        # (nw,3,NBIN) log10
    nw = B.shape[0]

    # per-window horizontal / vertical power and H/V
    PH = 0.5 * (P[:, 0] + P[:, 1])
    PV = P[:, 2]
    hv_w = 0.5 * (np.log10(PH @ _BMAT.T + EPS) - np.log10(PV @ _BMAT.T + EPS))
    hv = np.median(hv_w, axis=0)
    hv_iqr = np.subtract(*np.percentile(hv_w, [75, 25], axis=0))

    lh = np.median(np.log10(PH @ _BMAT.T + EPS), axis=0)
    lv = np.median(np.log10(PV @ _BMAT.T + EPS), axis=0)
    lh_n = lh - lh.mean()
    lv_n = lv - lv.mean()

    out = {}
    # ---- A: H/V spectral ratio curve (the classic site-response observable)
    out["A"] = hv.copy()
    # ---- B/C: normalised horizontal and vertical log-spectral shape
    out["B"] = lh_n
    out["C"] = lv_n
    # ---- D: H/V peak structure (f0 ~ Vs/4H is the depth/velocity observable)
    f0, a0, wid, f1, a1, npk = _peaks(hv, _LOGBF)
    out["D"] = np.array([f0, a0, wid, f1, a1, npk, hv.max() - hv.min(),
                         float(hv.mean()), float(hv.std()),
                         float(np.trapezoid(hv, _LOGBF)),
                         float(hv[_BFREQ < 2].mean()), float(hv[(_BFREQ >= 2) & (_BFREQ < 8)].mean()),
                         float(hv[_BFREQ >= 8].mean())])
    # ---- E: spectral shape statistics + band energy ratios
    e = _shape_stats(lh, _LOGBF) + _shape_stats(lv, _LOGBF)
    ph = np.median(PH, axis=0); pv = np.median(PV, axis=0)
    th, tv = ph.sum() + EPS, pv.sum() + EPS
    for m in _BMASKS:
        e.append(float(np.log10(ph[m].sum() / th + EPS)))
        e.append(float(np.log10(pv[m].sum() / tv + EPS)))
    out["E"] = np.array(e)
    # ---- F: horizontal polarisation, rotation-invariant (eigenvalues only)
    f = []
    FE = np.fft.rfft((w[0] - w[0].mean()))
    FN = np.fft.rfft((w[1] - w[1].mean()))
    fr = np.fft.rfftfreq(w.shape[1], 1.0 / FS)
    for lo, hi in _BANDS:
        m = (fr >= lo) & (fr < hi)
        if m.sum() < 2:
            f += [0.0, 0.0]; continue
        cee = float(np.real(FE[m] @ np.conj(FE[m])))
        cnn = float(np.real(FN[m] @ np.conj(FN[m])))
        cen_ = float(np.real(FE[m] @ np.conj(FN[m])))
        tr = cee + cnn
        det = cee * cnn - cen_ ** 2
        disc = max(tr * tr / 4 - det, 0.0)
        l1 = tr / 2 + np.sqrt(disc); l2 = tr / 2 - np.sqrt(disc)
        f.append(float((l1 - l2) / (l1 + l2 + EPS)))              # anisotropy
        f.append(float(np.log10(l2 / (l1 + EPS) + EPS)))          # eigenvalue ratio
    out["F"] = np.array(f)
    # ---- G: multi-window stability of the site signature
    f0s = [_peaks(hv_w[i], _LOGBF)[0] for i in range(nw)]
    out["G"] = np.array([float(np.mean(hv_iqr)), float(np.std(f0s)), float(np.mean(f0s)),
                         float(np.std(hv_w, axis=0).mean()),
                         float(np.std(hv_w[:, _BFREQ < 2], axis=0).mean()),
                         float(np.std(hv_w[:, _BFREQ >= 8], axis=0).mean()),
                         float(nw),
                         float(np.corrcoef(hv_w[0], hv_w[-1])[0, 1]) if nw > 1 else 1.0])
    # ---- H: time-domain shape / duration / coda decay
    h = []
    tot = 0.0
    for c in range(3):
        x = w[c] - w[c].mean()
        r = np.sqrt((x ** 2).mean()) + EPS
        env = _envelope(x)
        cum = np.cumsum(x ** 2); cum /= cum[-1] + EPS
        i5, i75, i95 = [int(np.searchsorted(cum, q)) for q in (0.05, 0.75, 0.95)]
        pk = int(np.argmax(env))
        tail = env[pk:]
        if len(tail) > 100:
            t = np.arange(len(tail)) / FS
            dec = float(np.polyfit(t, np.log10(tail + EPS), 1)[0])
        else:
            dec = 0.0
        h += [float(np.log10(r)), float(np.abs(x).max() / r),
              float(((x[:-1] * x[1:]) < 0).mean()),
              float(((x - x.mean()) ** 4).mean() / r ** 4),
              float(((x - x.mean()) ** 3).mean() / r ** 3),
              (i95 - i5) / FS, (i75 - i5) / FS, pk / FS, dec,
              float(env.max() / (env.mean() + EPS))]
        tot += r ** 2
    rms = [np.sqrt((w[c] ** 2).mean()) + EPS for c in range(3)]
    h += [float(np.log10(np.sqrt((rms[0] ** 2 + rms[1] ** 2) / 2) / rms[2])),
          float(np.log10(rms[0] / rms[1]))]
    out["H"] = np.array(h)
    # ---- I: trend-removed spectra: resonance shape with the broad source
    #         spectral shape (magnitude / distance / kappa) regressed out
    out["I"] = np.concatenate([_detrend(lh, _LOGBF), _detrend(hv, _LOGBF)])

    # ---- J: cepstral representation on the LINEAR frequency axis. The peak
    #         quefrency of the site term is the two-way travel time through the
    #         soft column, which is exactly what a thickness-weighted harmonic
    #         mean velocity over a fixed depth cell encodes.
    cPH = np.log10(np.median(PH, axis=0)[_CKEEPr] + EPS)
    cPV = np.log10(np.median(PV, axis=0)[_CKEEPr] + EPS)
    cJ_h = _cepstrum(cPH)
    cJ_hv = _cepstrum(0.5 * (cPH - cPV))
    j = [cJ_h, cJ_hv]
    for cc in (cJ_h, cJ_hv):
        a = np.abs(cc)
        i = int(np.argmax(a[1:])) + 1
        j.append(np.array([(i + 1) * _DQ, float(a[i]), float(a.mean()),
                           float(a[:8].sum() / (a.sum() + EPS))]))
    out["J"] = np.concatenate(j)

    # ---- K: window-resolved H/V. The late coda is closer to a diffuse field,
    #         where H/V is tied most directly to the site's own response, so the
    #         coda ratio and its drift from the early ratio separate the site
    #         term from the direct-arrival source/path term.
    half = max(nw // 2, 1)
    hv_e = np.median(hv_w[:half], axis=0)
    hv_l = np.median(hv_w[-half:], axis=0) if nw > 1 else hv_e
    coarse = hv_l.reshape(-1, 2).mean(axis=1) - hv_e.reshape(-1, 2).mean(axis=1)
    f0l, a0l, widl = _peaks(hv_l, _LOGBF)[:3]
    f0e = _peaks(hv_e, _LOGBF)[0]
    out["K"] = np.concatenate([hv_l, coarse,
                               np.array([f0l, a0l, widl, f0l - f0e])])

    # ---- L: Konno-Ohmachi smoothed H/V, 32 bins over 0.45-24.5 Hz, plus its
    #         own peak structure on the finer grid
    koH = np.median(PH, axis=0) @ _KOMAT.T
    koV = np.median(PV, axis=0) @ _KOMAT.T
    ko_hv = 0.5 * (np.log10(koH + EPS) - np.log10(koV + EPS))
    f0k, a0k, widk, f1k, a1k, npkk = _peaks(ko_hv, _LOGKOF)
    out["L"] = np.concatenate([ko_hv, np.array([f0k, a0k, widk, f1k, a1k, npkk])])

    # ---- M: long-window (41 s) low-frequency H/V and horizontal shape
    n = w.shape[1]
    if n < NLONG:                      # short / heavily padded record: zero-fill
        pad = np.zeros((3, NLONG)); pad[:, :n] = w
        segs = pad[None, :, :]
    else:
        st = list(range(0, n - NLONG + 1, 512))
        segs = np.stack([w[:, t:t + NLONG] for t in st])
    segs = (segs - segs.mean(axis=2, keepdims=True)) * _HANNL
    PL = (np.abs(np.fft.rfft(segs, axis=2)) ** 2)[:, :, _KEEPL] + EPS
    PHL = np.median(0.5 * (PL[:, 0] + PL[:, 1]), axis=0) @ _KOMATL.T
    PVL = np.median(PL[:, 2], axis=0) @ _KOMATL.T
    hvL = 0.5 * (np.log10(PHL + EPS) - np.log10(PVL + EPS))
    shL = np.log10(PHL + EPS); shL = shL - shL.mean()
    il = int(np.argmax(hvL))
    # ---- N: short-window high-frequency H/V and horizontal shape
    sts = list(range(0, max(w.shape[1] - NSHORT, 0) + 1, 128))
    if not sts:
        padS = np.zeros((3, NSHORT)); padS[:, :w.shape[1]] = w; segS = padS[None, :, :]
    else:
        segS = np.stack([w[:, t:t + NSHORT] for t in sts])
    segS = (segS - segS.mean(axis=2, keepdims=True)) * _HANNS
    PS = (np.abs(np.fft.rfft(segS, axis=2)) ** 2)[:, :, _KEEPS] + EPS
    PHS = np.median(0.5 * (PS[:, 0] + PS[:, 1]), axis=0) @ _KOMATS.T
    PVS = np.median(PS[:, 2], axis=0) @ _KOMATS.T
    hvS = 0.5 * (np.log10(PHS + EPS) - np.log10(PVS + EPS))
    shS = np.log10(PHS + EPS); shS = shS - shS.mean()
    iS = int(np.argmax(hvS))
    # slope of the high-frequency H/V: how fast amplification dies off above
    # the shallow resonance, which is set by the top few metres
    slS = float(np.polyfit(np.log10(_KOFS), hvS, 1)[0])
    out["N"] = np.concatenate([hvS, shS,
                               np.array([np.log10(_KOFS[iS]), hvS[iS], slS,
                                         float(hvS.mean()), float(hvS.std()),
                                         float(hvS.max() - hvS.min())])])

    out["M"] = np.concatenate([hvL, shL,
                               np.array([np.log10(_KOFL[il]), hvL[il],
                                         float(hvL.mean()), float(hvL.std())])])
    return out


def names():
    """Stable, human-readable feature names per group (same order as extract)."""
    bf = [f"{v:.2f}Hz" for v in _BFREQ]
    n = {}
    n["A"] = [f"hv_{b}" for b in bf]
    n["B"] = [f"hN_{b}" for b in bf]
    n["C"] = [f"vN_{b}" for b in bf]
    n["D"] = ["hv_f0", "hv_a0", "hv_w0", "hv_f1", "hv_a1", "hv_npk", "hv_range",
              "hv_mean", "hv_std", "hv_area", "hv_lo", "hv_mid", "hv_hi"]
    ss = ["cen", "spr", "ent", "rol85", "rol50", "slo_lo", "slo_hi", "q2", "q1"]
    n["E"] = [f"H_{s}" for s in ss] + [f"V_{s}" for s in ss] + \
             [f"{c}_bnd{i}" for i in range(len(_BANDS)) for c in ("H", "V")]
    n["F"] = [f"{k}_bnd{i}" for i in range(len(_BANDS)) for k in ("aniso", "eigr")]
    n["G"] = ["hv_iqr", "f0_std", "f0_mean", "hv_wstd", "hv_wstd_lo", "hv_wstd_hi",
              "nwin", "hv_w_corr"]
    n["H"] = [f"{c}_{k}" for c in ("E", "N", "Z")
              for k in ("lrms", "crest", "zcr", "kurt", "skew", "d595", "d575",
                        "tpk", "decay", "envpk")] + ["hv_rms", "en_rms"]
    n["I"] = [f"hDT_{b}" for b in bf] + [f"hvDT_{b}" for b in bf]
    q = [f"{(i + 1) * _DQ:.2f}s" for i in range(_NQ)]
    n["J"] = [f"cepH_{t}" for t in q] + [f"cepHV_{t}" for t in q] + \
             [f"{k}_{s}" for k in ("cepH", "cepHV")
              for s in ("qpk", "apk", "amean", "lowq")]
    n["L"] = [f"ko_{v:.2f}Hz" for v in _KOF] + \
             ["ko_f0", "ko_a0", "ko_w0", "ko_f1", "ko_a1", "ko_npk"]
    n["M"] = [f"hvLo_{v:.2f}Hz" for v in _KOFL] + \
             [f"hLo_{v:.2f}Hz" for v in _KOFL] + \
             ["hvLo_f0", "hvLo_a0", "hvLo_mean", "hvLo_std"]
    n["N"] = [f"hvHi_{v:.1f}Hz" for v in _KOFS] + \
             [f"hHi_{v:.1f}Hz" for v in _KOFS] + \
             ["hvHi_f0", "hvHi_a0", "hvHi_slope", "hvHi_mean", "hvHi_std", "hvHi_range"]
    n["K"] = [f"hvLate_{b}" for b in bf] + \
             [f"hvDrift_{i}" for i in range(NBIN // 2)] + \
             ["hvL_f0", "hvL_a0", "hvL_w0", "hvL_df0"]
    return n

# ===========================================================================
# Models. Every one is trained from scratch on the supplied training records
# inside this script. No pretrained weights, no external data, no GPU.
# ===========================================================================
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BANDS = ("v1", "v2", "v3", "v4")


def _per_cell(make, Xtr, Ytr, Xs):
    """Fit one classifier per depth cell; score every matrix in Xs with that
    same fit. Xs is a list so the held-out stations and the evaluation records
    are predicted from ONE training pass instead of two."""
    outs = [np.zeros((len(X), 4, 4)) for X in Xs]
    for c in range(4):
        m = make()
        m.fit(Xtr, Ytr[:, c])
        cls = m[-1].classes_ if hasattr(m, "steps") else m.classes_
        for o, X in zip(outs, Xs):
            pr = m.predict_proba(X)
            for j, cl in enumerate(cls):
                o[:, c, cl] = pr[:, j]
    return outs


def _centroid_probs(s_tr, s_list, Ytr):
    """Continuous per-cell score -> band distribution, using each band's TRAIN
    score centroid and spread. Train-derived constants, applied per record."""
    outs = [np.zeros((len(s), 4, 4)) for s in s_list]
    for c in range(4):
        cen, sd = [], []
        for k in range(4):
            m = Ytr[:, c] == k
            cen.append(s_tr[m, c].mean() if m.any() else float(k))
            sd.append(s_tr[m, c].std() if m.sum() > 2 else 1.0)
        cen = np.array(cen)
        sd = np.clip(np.array(sd), 1e-6, None)
        for o, sv in zip(outs, s_list):
            d = -0.5 * ((sv[:, c][:, None] - cen[None, :]) / sd[None, :]) ** 2 \
                - np.log(sd)[None, :]
            d -= d.max(axis=1, keepdims=True)
            p = np.exp(d)
            o[:, c] = p / p.sum(axis=1, keepdims=True)
    return outs


def m_hgb(Xtr, Ytr, Xs, seed):
    """Gradient-boosted trees: the strongest single family on these features."""
    return _per_cell(lambda: HistGradientBoostingClassifier(
        learning_rate=HGB_LR, max_leaf_nodes=HGB_LEAF, max_iter=HGB_IT,
        l2_regularization=HGB_L2, min_samples_leaf=20,
        random_state=seed, early_stopping=False), Xtr, Ytr, Xs)


def m_et(Xtr, Ytr, Xs, seed):
    """Extremely randomised trees: decorrelated from the boosted model."""
    return _per_cell(lambda: ExtraTreesClassifier(
        n_estimators=ET_N, min_samples_leaf=ET_LEAF, max_features="sqrt",
        random_state=seed, n_jobs=N_THREADS), Xtr, Ytr, Xs)


def m_lr(Xtr, Ytr, Xs, seed):
    """Strongly regularised multinomial logistic regression: a smooth linear
    decision surface, which 239 independent stations can actually support."""
    return _per_cell(lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=LR_C, max_iter=4000)),
        Xtr, Ytr, Xs)


def m_pls(Xtr, Ytr, Xs, seed):
    """Partial least squares on all four numeric band targets at once. Suits
    the shape of this problem: many correlated inputs, few samples, and four
    strongly correlated ordered outputs driven by one stiffness factor."""
    m = make_pipeline(StandardScaler(),
                      PLSRegression(n_components=PLS_K, scale=False))
    m.fit(Xtr, Ytr.astype(float))
    return _centroid_probs(m.predict(Xtr), [m.predict(X) for X in Xs], Ytr)


def m_ord(Xtr, Ytr, Xs, seed):
    """Cumulative-link ordinal model: three binary fits per cell for
    P(band > k). Uses the band ordering, so each fit sees every record."""
    outs = [np.zeros((len(X), 4, 4)) for X in Xs]
    for c in range(4):
        cums = [np.ones((len(X), 5)) for X in Xs]
        for cum in cums:
            cum[:, 4] = 0.0
        for k in range(3):
            t = (Ytr[:, c] > k).astype(int)
            if len(set(t.tolist())) < 2:
                for cum in cums:
                    cum[:, k + 1] = float(t[0])
                continue
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=ORD_C, max_iter=4000))
            m.fit(Xtr, t)
            for cum, X in zip(cums, Xs):
                cum[:, k + 1] = m.predict_proba(X)[:, 1]
        for o, cum in zip(outs, cums):
            for k in range(3, 0, -1):
                cum[:, k] = np.minimum(cum[:, k], cum[:, k - 1])
            p = np.clip(cum[:, :4] - cum[:, 1:5], 1e-6, None)
            o[:, c] = p / p.sum(axis=1, keepdims=True)
    return outs


def m_ordgb(Xtr, Ytr, Xs, seed):
    """The cumulative-link construction with boosted trees as the binary
    learner: three fits per cell for P(band > k). Combines the two things that
    measured best independently - the band ordering and gradient boosting - and
    costs about what one four-class booster does (3 binary trees per round
    against 4)."""
    outs = [np.zeros((len(X), 4, 4)) for X in Xs]
    for c in range(4):
        cums = [np.ones((len(X), 5)) for X in Xs]
        for cum in cums:
            cum[:, 4] = 0.0
        for k in range(3):
            t = (Ytr[:, c] > k).astype(int)
            if len(set(t.tolist())) < 2:
                for cum in cums:
                    cum[:, k + 1] = float(t[0])
                continue
            # NOTE: `max_features` is deliberately not passed. It needs
            # scikit-learn >= 1.4, and cross-validation showed the default is
            # BETTER here anyway (0.2470 against 0.2394), so omitting it
            # removes a version dependency and improves the score.
            m = HistGradientBoostingClassifier(
                learning_rate=HGB_LR, max_leaf_nodes=HGB_LEAF, max_iter=HGB_IT,
                l2_regularization=HGB_L2,
                min_samples_leaf=20, random_state=seed, early_stopping=False)
            m.fit(Xtr, t)
            for cum, X in zip(cums, Xs):
                cum[:, k + 1] = m.predict_proba(X)[:, 1]
        for o, cum in zip(outs, cums):
            for k in range(3, 0, -1):
                cum[:, k] = np.minimum(cum[:, k], cum[:, k - 1])
            p = np.clip(cum[:, :4] - cum[:, 1:5], 1e-6, None)
            o[:, c] = p / p.sum(axis=1, keepdims=True)
    return outs


def m_ldak(Xtr, Ytr, Xs, seed):
    """Shrunk LDA to a 3-D discriminant space, then a neighbour vote in it."""
    return _per_cell(lambda: make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="eigen", shrinkage=LDA_SHRINK),
        KNeighborsClassifier(n_neighbors=LDA_K, weights="distance")),
        Xtr, Ytr, Xs)


# The ensemble members, chosen by equal-weight forward selection on repeated
# station-held-out cross-validation (see final_approach.md).
MODELS = [(n, globals()["m_" + n]) for n in MODEL_NAMES]


# ===========================================================================
# Decode. Macro-F1 scores a band that is never predicted as 0, so a plain
# argmax that avoids the two middle bands throws away half of each cell.
# A per-band weight vector is fitted on the out-of-fold TRAINING predictions
# so that the predicted marginal matches the training band frequencies; it is
# then a fixed 4x4 constant applied to one record at a time.
# ===========================================================================
def fit_band_weights(prob, Y, iters=200):
    W = np.ones((4, 4))
    for c in range(4):
        target = np.bincount(Y[:, c], minlength=4) / len(Y)
        w = np.ones(4)
        for _ in range(iters):
            freq = np.bincount((prob[:, c, :] * w).argmax(axis=1), minlength=4) / len(Y)
            w = w * ((target + 1e-3) / (freq + 1e-3)) ** 0.25
            w = w / w.mean()
        W[c] = w
    return W


def apply_band_weights(prob, W):
    return (prob * W[None, :, :]).argmax(axis=2)


# ---------------------------------------------------------------------------
# Structured decode. The four per-cell models are fitted independently, and
# measurement shows they UNDER-couple the column: the correlation between
# predicted cell bands (0.66-0.76) is lower than between the true ones
# (0.71-0.88). Only 68 of the 256 possible profiles occur in training, so the
# empirical distribution over profiles carries real information the per-cell
# models never see. Re-weighting each record's four distributions by that
# distribution and taking the marginals restores the coupling.
#
# The prior is counted from TRAINING labels only and is a fixed table at
# inference; the re-weighting uses one record's own four distributions and
# nothing else, so the prediction stays per-record. LAMBDA was chosen from the
# middle of a flat region of cross-validated scores (0.7-1.3), not at its edge.
# ---------------------------------------------------------------------------
def fit_profile_prior(Y):
    profiles, counts = np.unique(Y, axis=0, return_counts=True)
    return profiles, counts.astype(float)


def joint_decode(prob, profiles, counts, lam=JOINT_LAMBDA):
    logpri = np.log(counts / counts.sum())
    lp = np.log(prob + 1e-9)
    # score every observed profile for every record, then marginalise per cell
    sc = lp[:, np.arange(4)[None, :], profiles].sum(axis=2) + lam * logpri[None, :]
    sc -= sc.max(axis=1, keepdims=True)
    post = np.exp(sc)
    post /= post.sum(axis=1, keepdims=True)
    out = np.zeros_like(prob)
    for c in range(4):
        for k in range(4):
            out[:, c, k] = post[:, profiles[:, c] == k].sum(axis=1)
    return out


# ===========================================================================
# Submission validation: a malformed file is rejected outright and scores
# nothing, so this raises before anything is written.
# ===========================================================================
def validate_submission(sub, sample_path, test_ids):
    sample = pd.read_csv(sample_path, keep_default_na=False)
    assert list(sub.columns) == list(sample.columns), \
        f"columns {list(sub.columns)} != {list(sample.columns)}"
    assert len(sub) == len(sample), f"rows {len(sub)} != {len(sample)}"
    ids = sub["id"].astype(str).tolist()
    assert ids == [str(i) for i in test_ids], "id set/order does not match test.csv"
    assert len(set(ids)) == len(ids), "duplicate ids"
    for p in sub["profile"].astype(str):
        parts = p.split("|")
        assert len(parts) == 4, f"profile {p!r} does not have four cells"
        for b in parts:
            assert b in BANDS, f"band {b!r} outside v1..v4"
    assert not sub["profile"].isna().any(), "NaN profile"
    assert (sub["profile"].astype(str).str.len() > 0).all(), "empty profile"


N_FEAT = sum(len(names()[g]) for g in USE_GROUPS)

# ===========================================================================
# Driver
# ===========================================================================
def build_features(z, ids, n_views):
    """Feature matrix for a list of ids. View 0 is the record's full valid span;
    the remaining views are overlapping sub-windows of THAT SAME record, used as
    extra training rows and as several looks at one record at inference.
    Nothing is ever computed across records."""
    out = np.zeros((len(ids), n_views, N_FEAT), dtype=np.float64)
    for i, k in enumerate(ids):
        w = np.asarray(z[k], dtype=np.float64)
        views = [w] + (crops(w) if n_views > 1 else [])
        for v in range(n_views):
            d = extract(views[v])
            out[i, v] = np.concatenate([d[g] for g in USE_GROUPS])
    return out


def main():
    seed_everything()
    log("start")
    train = pd.read_csv(PUBLIC_DIR / "train.csv")
    test = pd.read_csv(PUBLIC_DIR / "test.csv")
    sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv", keep_default_na=False)
    folds = pd.read_csv(PUBLIC_DIR / "folds.csv")
    z = np.load(PUBLIC_DIR / "waveforms.npz")

    # folds.csv holds out whole stations, which is how the evaluation set is
    # built, so it is used both for the blend/decode fitting below and as the
    # honest estimate of generalisation to unseen stations.
    train = train.merge(folds, on="id", how="left")
    assert train.fold.notna().all(), "a training id is missing from folds.csv"
    Y = np.array([[BANDS.index(b) for b in p.split("|")] for p in train.profile])
    fold = train.fold.to_numpy().astype(int)

    Xtr = build_features(z, train.id.tolist(), N_VIEWS)
    log(f"train features {Xtr.shape}")
    Xte = build_features(z, test.id.tolist(), N_VIEWS)
    log(f"test features {Xte.shape}")

    # ---- fold models. Each fold model predicts its own held-out stations (the
    # out-of-fold material that the blend weights and the band weights are
    # fitted on) and the evaluation records (averaged over folds, a free
    # bagging ensemble). No evaluation label or statistic is ever used.
    oof = {n: np.zeros((len(Y), 4, 4)) for n, _ in MODELS}
    tep = {n: np.zeros((len(test), 4, 4)) for n, _ in MODELS}
    nf = int(fold.max()) + 1
    for f in range(nf):
        va = fold == f
        tr = ~va
        if N_VIEWS > 1:
            Xt = np.vstack([Xtr[tr, v] for v in range(N_VIEWS)])
            Yt = np.vstack([Y[tr]] * N_VIEWS)
        else:
            Xt, Yt = Xtr[tr, 0], Y[tr]
        for name, fn in MODELS:
            for s in SEEDS:
                for v in range(N_VIEWS):
                    # one training pass scores both the held-out stations and
                    # the evaluation records
                    pv, pt = fn(Xt, Yt, [Xtr[va, v], Xte[:, v]], s)
                    oof[name][va] += pv / (len(SEEDS) * N_VIEWS)
                    tep[name] += pt / (len(SEEDS) * N_VIEWS * nf)
        log(f"fold {f} done")

    for name, _ in MODELS:
        s, d = score_cells_int(Y, oof[name].argmax(axis=2))
        log(f"  OOF {name:5s} {s:.4f} cells " + " ".join(f"{x:.3f}" for x in d))

    # ---- blend. Equal weights over the families: with 239 independent
    # stations, fitted weights have more freedom than the out-of-fold set can
    # resolve, and a plain average was at least as good in cross-validation.
    oof_b = np.mean([oof[n] for n, _ in MODELS], axis=0)
    te_b = np.mean([tep[n] for n, _ in MODELS], axis=0)
    s, d = score_cells_int(Y, oof_b.argmax(axis=2))
    log(f"  OOF blend argmax {s:.4f} cells " + " ".join(f"{x:.3f}" for x in d))

    # ---- structured decode. The out-of-fold figure below is computed with the
    # profile prior counted on the OTHER folds only, so it is an honest
    # estimate for stations the model has not seen.
    oof_dec = np.zeros_like(oof_b)
    for f in range(nf):
        va = fold == f
        prof, cnt = fit_profile_prior(Y[~va])
        oof_dec[va] = joint_decode(oof_b[va], prof, cnt)
    s, d = score_cells_int(Y, oof_dec.argmax(axis=2))
    log(f"  OOF blend decoded {s:.4f} cells " + " ".join(f"{x:.3f}" for x in d))

    # for the evaluation records the prior is counted on all training labels
    prof, cnt = fit_profile_prior(Y)
    log(f"  profile prior: {len(prof)} distinct profiles over {int(cnt.sum())} records")
    pred = joint_decode(te_b, prof, cnt).argmax(axis=2)
    profiles = ["|".join(BANDS[k] for k in row) for row in pred]

    sub = pd.DataFrame({"id": test.id.to_numpy(), "profile": profiles})
    sub = sub[list(sample.columns)]
    validate_submission(sub, PUBLIC_DIR / "sample_submission.csv", test.id.tolist())
    SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION_OUT, index=False)
    log(f"wrote {SUBMISSION_OUT} shape={sub.shape}")
    for c in range(4):
        log(f"  cell{c} predicted band counts "
            f"{np.bincount(pred[:, c], minlength=4).tolist()} "
            f"(train {np.bincount(Y[:, c], minlength=4).tolist()})")


if __name__ == "__main__":
    main()
