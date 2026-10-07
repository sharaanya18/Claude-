"""Spaced-seed ("gapped") character n-gram featuriser.
A contiguous 4-gram needs 4 intact characters and survives 30% corruption with p=0.24; the
same window read through a weight-3 mask needs only 3 and survives with p=0.34, so two
independently corrupted copies of the same text keep far more shared evidence. Spaces, digits
and punctuation are never corrupted, so masks that land on them are free anchors.
Features are hashed; vocabulary/weights are always fitted on TRAIN text only."""
import numpy as np, scipy.sparse as sp

def seed_sets(name):
    S = {
        "c234":   [(0,1),(0,1,2),(0,1,2,3)],
        "c2345":  [(0,1),(0,1,2),(0,1,2,3),(0,1,2,3,4)],
        "g3w5":   [(0,1,2),(0,1,3),(0,2,3),(0,1,4),(0,2,4),(0,3,4)],
        "g3w4":   [(0,1,2),(0,1,3),(0,2,3)],
        "g4w6":   [(0,1,2,3),(0,1,2,4),(0,1,3,4),(0,2,3,4),(0,1,2,5),(0,1,3,5),(0,2,3,5),
                   (0,1,4,5),(0,2,4,5),(0,3,4,5)],
        "g4w5":   [(0,1,2,3),(0,1,2,4),(0,1,3,4),(0,2,3,4)],
        "mix_a":  [(0,1),(0,1,2),(0,1,3),(0,2,3),(0,1,2,3),(0,1,2,4),(0,1,3,4),(0,2,3,4)],
        "mix_b":  [(0,1),(0,2),(0,1,2),(0,1,3),(0,2,3),(0,1,4),(0,2,4),(0,3,4),
                   (0,1,2,3),(0,1,2,4),(0,1,3,4),(0,2,3,4),(0,1,2,3,4)],
        "mix_c":  [(0,1),(0,2),(0,1,2),(0,1,3),(0,2,3),(0,1,4),(0,2,4),(0,3,4),
                   (0,1,2,3),(0,1,2,4),(0,1,3,4),(0,2,3,4),(0,1,2,5),(0,1,3,5),(0,2,3,5),
                   (0,1,4,5),(0,2,4,5),(0,3,4,5),(0,1,2,3,4)],
    }
    return S[name]


def masks(w, W):
    """All weight-w masks spanning exactly a window of W (first and last position always on)."""
    from itertools import combinations
    if w == W: return [tuple(range(W))]
    if w < 2 or w > W: return []
    return [(0,) + tuple(m) + (W - 1,) for m in combinations(range(1, W - 1), w - 2)]

def seed_lib(spec):
    """spec like 'w2:2-6,w3:3-6,w4:4-7' -> list of masks."""
    out = []
    for part in spec.split(","):
        w, rng = part.split(":")
        w = int(w[1:]); lo, hi = (int(x) for x in rng.split("-"))
        for W in range(lo, hi + 1): out += masks(w, W)
    seen, uniq = set(), []
    for m in out:
        if m not in seen: seen.add(m); uniq.append(m)
    return uniq

def codes(texts, L=100):
    M = np.zeros((len(texts), L), dtype=np.int64)
    for n, t in enumerate(texts):
        b = np.frombuffer(t[:L].encode("ascii", "ignore"), dtype=np.uint8)
        M[n, :len(b)] = b
    return M

def hash_features(M, seeds, nbits=20):
    """(N,L) codes -> (N, 2**nbits) sparse counts over hashed spaced-seed patterns."""
    N, L = M.shape
    H = 1 << nbits
    rows, cols = [], []
    for sid, off in enumerate(seeds):
        span = off[-1]
        npos = L - span
        v = np.full((N, npos), sid * 2654435761 + 1, dtype=np.int64)
        for o in off:
            v = (v * 131 + M[:, o:o + npos]) & 0x7FFFFFFFFFFF
        v = ((v * 0x9E3779B1) >> 13) & (H - 1)
        rows.append(np.repeat(np.arange(N), npos)); cols.append(v.ravel())
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    X = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, H)).tocsr()
    X.sum_duplicates()
    return X

class SeedTfidf:
    """TF-IDF over spaced-seed features. fit() sees TRAIN text only."""
    def __init__(self, seeds, nbits=20, min_df=3, sublinear=True):
        self.seeds, self.nbits, self.min_df, self.sublinear = seeds, nbits, min_df, sublinear
    def fit(self, texts):
        X = hash_features(codes(texts), self.seeds, self.nbits)
        df = np.asarray((X > 0).sum(0)).ravel()
        n = X.shape[0]
        if self.min_df <= 1:
            # Keep EVERY pattern. A rare n-gram shared by a held-out opening and its own
            # continuation never appears in training, so any df cut deletes exactly the
            # evidence that discriminates at evaluation time; smoothed idf gives those
            # unseen patterns the maximum weight instead of dropping them.
            self.keep = np.ones(X.shape[1], dtype=bool)
            self.idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
        else:
            self.keep = df >= self.min_df
            self.idf = np.zeros(X.shape[1], dtype=np.float32)
            self.idf[self.keep] = np.log((1.0 + n) / (1.0 + df[self.keep])) + 1.0
        return self
    def transform(self, texts):
        X = hash_features(codes(texts), self.seeds, self.nbits).tocsr()
        if self.sublinear: X.data = 1.0 + np.log(X.data)
        X = X.multiply(self.idf[None, :]).tocsr()
        X.eliminate_zeros()
        nrm = np.sqrt(np.asarray(X.multiply(X).sum(1))).ravel(); nrm[nrm == 0] = 1.0
        return sp.diags(1.0 / nrm) @ X
