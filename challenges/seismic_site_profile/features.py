"""Waveform -> site-response feature families for the Seismic Site Profile task.

Physics the representation is built on. A soft surface layer over firmer rock
resonates at f0 ~ Vs/(4H), amplifies the horizontal components near f0 far more
than the vertical, and attenuates high frequencies. Source and path dominate the
raw spectrum, but they are shared by all three components of one record, so
ratios between components (H/V) and the SHAPE of a spectrum relative to its own
smooth trend largely cancel them while keeping the site term. Amplitude is
normalised per record, so only shape and inter-component ratios are available.

Everything here is computed from ONE record in isolation: no cross-record
pooling, no file order, no id. Groups are separable so they can be ablated.
"""
import numpy as np

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


GROUPS = "ABCDEFGHIJKLMN"


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
