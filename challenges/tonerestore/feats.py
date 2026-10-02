"""Acoustic front-end shared by dev experiments and (later) inlined into solution.py.
All per-clip, deterministic, no cross-row statistics."""
import numpy as np, soundfile as sf, torch, torchaudio

HOP_MS = 10


def load_wav(path):
    x, sr = sf.read(path, dtype="float32")
    return x, sr


def yin_f0(x, sr, fmin=70.0, fmax=450.0, hop_ms=HOP_MS, thresh=0.15):
    """Batched YIN on one clip. Returns per-frame (log2 f0 [nan if unvoiced], voicing = 1 - min CMND, log energy)."""
    hop = int(sr * hop_ms / 1000)
    tau_min, tau_max = int(sr / fmax), int(sr / fmin)
    W = 2 * tau_max  # analysis window
    xt = torch.from_numpy(x).double()
    pad = W // 2
    xt = torch.nn.functional.pad(xt, (pad, pad + W))
    fr = xt.unfold(0, W, hop)  # (N, W)
    n_fr = (len(x) // hop) + 1
    fr = fr[:n_fr]
    J = W - tau_max
    a = fr[:, :J]
    # d(tau) = sum_j (a_j - x_{j+tau})^2 = E0 + E_tau - 2 r(tau)
    nfft = 1 << int(np.ceil(np.log2(W + J)))
    Fa = torch.fft.rfft(a, nfft)
    Fb = torch.fft.rfft(fr, nfft)
    r = torch.fft.irfft(Fb * torch.conj(Fa), nfft)[:, : tau_max + 1]
    cs = torch.cumsum(fr**2, dim=1)
    cs = torch.cat([torch.zeros(len(fr), 1, dtype=cs.dtype), cs], 1)
    e0 = cs[:, J : J + 1] - cs[:, 0:1]
    taus = torch.arange(tau_max + 1)
    et = cs[:, J + taus] - cs[:, taus]
    d = e0 + et - 2 * r
    d[:, 0] = 0
    cm = torch.cumsum(d, dim=1)
    cmnd = d[:, 1:] * torch.arange(1, tau_max + 1) / (cm[:, 1:] + 1e-12)  # taus 1..tau_max
    c = cmnd[:, tau_min - 1 :]  # taus tau_min..tau_max
    # first dip below threshold, else global min
    below = c < thresh
    first = torch.where(below.any(1), below.double().argmax(1), c.argmin(1))
    # slide to local minimum after the first dip
    idx = first.clone()
    for _ in range(40):
        nxt = torch.clamp(idx + 1, max=c.shape[1] - 1)
        mv = (c.gather(1, nxt[:, None])[:, 0] < c.gather(1, idx[:, None])[:, 0]) & below.any(1)
        idx = torch.where(mv, nxt, idx)
    i0 = idx.clamp(1, c.shape[1] - 2)
    y0, y1, y2 = c.gather(1, (i0 - 1)[:, None])[:, 0], c.gather(1, i0[:, None])[:, 0], c.gather(1, (i0 + 1)[:, None])[:, 0]
    den = (y0 - 2 * y1 + y2)
    shift = torch.where(den.abs() > 1e-9, 0.5 * (y0 - y2) / den, torch.zeros_like(den)).clamp(-1, 1)
    tau = (i0 + tau_min).double() + shift
    f0 = sr / tau
    vo = (1 - c.gather(1, i0[:, None])[:, 0]).clamp(0, 1)
    en = torch.log(fr[:, tau_max - hop : tau_max + hop].pow(2).mean(1) + 1e-8)  # centered 20 ms
    lf = torch.log2(f0)
    lf = torch.where(vo > 0.7, lf, torch.full_like(lf, float("nan")))
    return lf.float().numpy(), vo.float().numpy(), en.float().numpy()


def f0_features(x, sr):
    """Utterance-normalised frame features (T, 6): relative log-f0 (semitones, 0 when unvoiced), voiced flag,
    voicing strength, delta f0, log-energy (z-scored within clip), voiced-energy-gated."""
    lf, vo, en = yin_f0(x, sr)
    v = np.isfinite(lf)
    if v.sum() >= 3:
        # light median smoothing over voiced runs to kill octave blips
        s = lf.copy()
        for t in range(len(lf)):
            lo, hi = max(0, t - 2), min(len(lf), t + 3)
            w = lf[lo:hi][np.isfinite(lf[lo:hi])]
            s[t] = np.median(w) if (v[t] and len(w)) else np.nan
        lf = s
        v = np.isfinite(lf)
        med = np.median(lf[v])
        rel = np.where(v, (lf - med) * 12.0, 0.0)  # semitones
    else:
        rel = np.zeros_like(lf)
    d = np.zeros_like(rel)
    d[1:] = np.where(v[1:] & v[:-1], rel[1:] - rel[:-1], 0.0)
    ez = (en - en.mean()) / (en.std() + 1e-6)
    return np.stack([rel / 6.0, v.astype(np.float32), vo, d, ez, ez * v], 1).astype(np.float32)
