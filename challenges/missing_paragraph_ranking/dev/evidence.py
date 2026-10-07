"""Pair EVIDENCE PROFILE.
A single cosine collapses all lexical agreement into one number. Instead we decompose the
agreement between an opening and a candidate into cells -- (seed weight) x (rarity bucket),
plus a noise-free STRUCTURE channel where every letter is masked out so only word lengths,
punctuation and digits remain -- and hand the whole profile to a trained ranker, which learns
how much each kind of evidence is worth. All vocabulary/idf/bucket edges fit on TRAIN only."""
import numpy as np, scipy.sparse as sp
from ngram import seed_lib, codes

LETTER_LO, LETTER_HI = 97, 122

def skeleton(M):
    """Replace every lowercase letter with one placeholder code; the corruption never touches
    spaces, digits or punctuation, so this channel is completely noise-free."""
    out = M.copy()
    out[(M >= LETTER_LO) & (M <= LETTER_HI)] = 1
    return out

class Evidence:
    def __init__(self, spec="w2:2-8,w3:3-8,w4:4-8", skel_spec="w3:3-6,w4:4-7",
                 nbits=22, min_df=3, nbuck=6):
        self.seeds = seed_lib(spec); self.skel = seed_lib(skel_spec)
        self.nbits, self.min_df, self.nbuck = nbits, min_df, nbuck
        self.H = 1 << nbits

    def _raw(self, M, seeds, chan):
        """(N,L) codes -> csr counts; each seed owns a slice of the hash space so the seed
        (and therefore its weight) is recoverable from the bin index."""
        N, L = M.shape
        per = self.H // len(seeds)
        rows, cols = [], []
        for sid, off in enumerate(seeds):
            span = off[-1]; npos = L - span
            v = np.full((N, npos), (sid + 1) * 2654435761, dtype=np.int64)
            for o in off:
                v = (v * 131 + M[:, o:o + npos]) & 0x7FFFFFFFFFFF
            v = sid * per + (((v * 0x9E3779B1) >> 13) % per)
            rows.append(np.repeat(np.arange(N), npos)); cols.append(v.ravel())
        X = sp.coo_matrix((np.ones(sum(len(c) for c in cols), dtype=np.float32),
                           (np.concatenate(rows), np.concatenate(cols))), shape=(N, self.H)).tocsr()
        X.sum_duplicates()
        return X

    def fit(self, texts):
        """idf and rarity buckets from TRAIN text only."""
        self.cell = {}
        for chan, seeds in (("L", self.seeds), ("S", self.skel)):
            M = codes(texts)
            if chan == "S": M = skeleton(M)
            X = self._raw(M, seeds, chan)
            df = np.asarray((X > 0).sum(0)).ravel()
            n = X.shape[0]
            idf = np.zeros(self.H, dtype=np.float32)
            keep = df >= self.min_df
            idf[keep] = np.log((1.0 + n) / (1.0 + df[keep])) + 1.0
            per = self.H // len(seeds)
            wt = np.zeros(self.H, dtype=np.int8)
            for sid, off in enumerate(seeds):
                wt[sid * per:(sid + 1) * per] = min(len(off), 5)
            # rarity buckets on the observed idf range of kept features
            q = np.quantile(idf[keep], np.linspace(0, 1, self.nbuck + 1)[1:-1]) if keep.any() else []
            buck = np.digitize(idf, q).astype(np.int8)
            ws = sorted(set(wt[keep].tolist())) if keep.any() else [3]
            wmap = {w: i for i, w in enumerate(ws)}
            cid = np.full(self.H, -1, dtype=np.int16)
            for w in ws:
                m = keep & (wt == w)
                cid[m] = wmap[w] * self.nbuck + buck[m]
            self.cell[chan] = dict(idf=idf, cid=cid, ncell=len(ws) * self.nbuck, seeds=seeds)
        self.ncell = sum(self.cell[c]["ncell"] for c in ("L", "S"))
        self.nfeat = self.ncell + 2 * 3           # + per-channel aggregates
        return self

    def vectors(self, texts):
        out = {}
        for chan in ("L", "S"):
            M = codes(texts)
            if chan == "S": M = skeleton(M)
            X = self._raw(M, self.cell[chan]["seeds"], chan).tocsr()
            X.data = 1.0 + np.log(X.data)
            X = X.multiply(self.cell[chan]["idf"][None, :]).tocsr()
            X.eliminate_zeros()
            nrm = np.sqrt(np.asarray(X.multiply(X).sum(1))).ravel(); nrm[nrm == 0] = 1.0
            out[chan] = sp.diags(1.0 / nrm) @ X.tocsr()
        return out

    def profiles(self, Xq, Xc, pools):
        """(Q, P, nfeat) evidence profile. pools[q] holds row indices into Xc."""
        Q, P = pools.shape
        F = np.zeros((Q, P, self.nfeat), dtype=np.float32)
        base = 0
        for chan in ("L", "S"):
            A, Bm = Xq[chan].tocsr(), Xc[chan].tocsr()
            cid, K = self.cell[chan]["cid"], self.cell[chan]["ncell"]
            buf = np.zeros(self.H, dtype=np.float32)
            for q in range(Q):
                s, e = A.indptr[q], A.indptr[q + 1]
                ai, av = A.indices[s:e], A.data[s:e]
                buf[ai] = av
                for p in range(P):
                    r = pools[q, p]
                    s2, e2 = Bm.indptr[r], Bm.indptr[r + 1]
                    bi = Bm.indices[s2:e2]
                    prod = buf[bi] * Bm.data[s2:e2]
                    nz = prod > 0
                    if not nz.any():
                        continue
                    pv, pc = prod[nz], cid[bi[nz]]
                    ok = pc >= 0
                    pv, pc = pv[ok], pc[ok]
                    if len(pv) == 0:
                        continue
                    F[q, p, base:base + K] = np.bincount(pc, weights=pv, minlength=K)[:K]
                    F[q, p, base + K] = pv.sum()                       # plain cosine
                    F[q, p, base + K + 1] = len(pv)                    # how many features agree
                    F[q, p, base + K + 2] = pv.max()                   # strongest single match
                buf[ai] = 0.0
            base += K + 3
        return F
