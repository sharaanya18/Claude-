"""DIAGNOSTIC ONLY (not a solution): spaces/punctuation/digits are never corrupted, so token
boundaries and token LENGTHS survive intact. Test whether position-wise fuzzy matching of
equal-length tokens recovers signal that exact n-gram TF-IDF destroys."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import *

D = Data(sys.argv[1])
fold = make_folds(D.subv, 5, seed=0)
A, B = np.array(D.A, dtype=object), np.array(D.B, dtype=object)
TMAX, LMAX = 32, 20

def tokmat(texts):
    """(N, TMAX, LMAX) uint8 codes, 0 = pad, plus (N, TMAX) token lengths."""
    M = np.zeros((len(texts), TMAX, LMAX), dtype=np.uint8)
    L = np.zeros((len(texts), TMAX), dtype=np.int16)
    for n, t in enumerate(texts):
        for k, w in enumerate(t.split()[:TMAX]):
            w = w[:LMAX]
            M[n, k, :len(w)] = np.frombuffer(w.encode(), dtype=np.uint8)
            L[n, k] = len(w)
    return M, L

def fuzzy_scores(qa, cb, pools_local, minlen, mode):
    Mq, Lq = tokmat(qa); Mc, Lc = tokmat(cb)
    isal_q = ((Mq >= 97) & (Mq <= 122))
    out = np.zeros((len(qa), pools_local.shape[1]))
    for r in range(len(qa)):
        idx = pools_local[r]
        mq, lq = Mq[r], Lq[r]                                   # (T,L), (T,)
        mc, lc = Mc[idx], Lc[idx]                               # (P,T,L), (P,T)
        eq = (mq[None, :, None, :] == mc[:, None, :, :])        # (P,Tq,Tc,L)
        eq &= isal_q[r][None, :, None, :]                       # letters only
        match = eq.sum(-1).astype(np.float32)                   # (P,Tq,Tc)
        same = (lq[None, :, None] == lc[:, None, :]) & (lq[None, :, None] >= minlen)
        agree = np.where(same, match / np.maximum(lq[None, :, None], 1), -1.0)
        best = agree.max(2)                                     # (P,Tq) best partner per token
        valid = lq >= minlen
        if mode == "sum":
            out[r] = np.where(best > 0, best, 0).sum(1)
        elif mode == "lenw":            # weight by token length (longer = rarer = more telling)
            w = np.where(valid, lq, 0).astype(np.float32)
            out[r] = (np.where(best > 0, best, 0) * w[None, :]).sum(1) / max(w.sum(), 1)
        elif mode == "strict":          # count tokens matched above 60% of their characters
            out[r] = ((best >= 0.6) & valid[None, :]).sum(1).astype(np.float32)
    return out

f = 0
vai = np.where(fold == f)[0]
pools = build_pools(vai, D.subv, seed=100 + f)
pos = {g: k for k, g in enumerate(vai)}
pl = np.array([[pos[j] for j in row] for row in pools])
print(f"fold 0: {len(vai)} validation queries, pools from held-out same-subfield abstracts\n")
for eff, rate in [(0.20, 0.0), (0.30, extra_rate(0.20, 0.30))]:
    rng = np.random.default_rng(7)
    qa = list(A[vai]); cb = list(B[vai])
    if rate > 0:
        qa = corrupt(qa, rate, rng); cb = corrupt(cb, rate, rng)
    for minlen in (4, 5, 6, 7):
        for mode in ("sum", "lenw", "strict"):
            S = fuzzy_scores(qa, cb, pl, minlen, mode)
            m = metrics_from_matrix(S)
            print(f"noise={eff:.2f} minlen={minlen} mode={mode:7s} -> SCORE {m['score']:.4f} "
                  f"top1 {m['top1']:.3f} top5 {m['top5']:.3f}")
    print()
