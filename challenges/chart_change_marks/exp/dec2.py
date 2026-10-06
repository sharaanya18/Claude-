import numpy as np
import solution as S

def candidates(h, b, lo=0.08, win=6):
    cands = S.decode_chart(h, b, lo)
    # mass-based match probability: heat summed within +-win px of the peak, capped at 1
    out = []
    for x, s, p in cands:
        c = int(round(x)); m = float(np.clip(h[max(0, c - win):c + win + 1].max(), 0, 1))
        out.append((x, s, p))
    return out

def decode_ev(cands, gamma=1.0, nscale=1.0, cbar=0.9, prior0=1.0):
    """Pick the top-k marks (by heat) maximising the expected chart score 2*sum(s*c)/(k + n_hat)."""
    if not cands:
        return []
    s = np.array([c[1] for c in cands]) ** gamma
    order = np.argsort(-s)
    s = s[order]
    n_hat = nscale * s.sum()
    p_empty = float(np.prod(1 - np.clip(s, 0, 0.999)))
    best_k, best = 0, p_empty * prior0
    cs = np.cumsum(s * cbar)
    for k in range(1, len(s) + 1):
        e = 2 * cs[k - 1] / (k + n_hat)
        if e > best:
            best, best_k = e, k
    return [cands[i] for i in order[:best_k]]

def to_marks(sel):
    return [{"x": round(x, 2), "p": [float(v) for v in p]} for x, s, p in sel]
