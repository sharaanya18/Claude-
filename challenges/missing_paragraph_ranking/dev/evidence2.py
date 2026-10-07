"""Pair features v2: a compact set of COMPLEMENTARY match scorers, computed in one shared
hash space. Each spaced-seed family is scored in its own normalised subspace (so the family
with the most patterns cannot dominate), alongside rarity strata, match-shape statistics and
a completely noise-free structure channel. The trained ranker learns how to weigh them.
Every statistic (idf, rarity edges, normalisation) is fitted on TRAIN text only."""
import numpy as np, scipy.sparse as sp
from ngram import seed_lib, codes

def skeleton(M):
    out = M.copy(); out[(M >= 97) & (M <= 122)] = 1; return out

class Evidence2:
    def __init__(self, spec="w2:2-8,w3:3-8,w4:4-8", skel="w3:3-6,w4:4-7",
                 nbits=22, min_df=2, nbuck=3):
        self.seeds = seed_lib(spec); self.skseeds = seed_lib(skel)
        self.nbits, self.min_df, self.nbuck = nbits, min_df, nbuck
        self.H = 1 << nbits
        self.fams = sorted({len(s) for s in self.seeds})           # seed weight = family
        self.names = (["cos_all"] + [f"cos_w{w}" for w in self.fams] + ["cos_skel"]
                      + [f"rare{b}" for b in range(nbuck)] + ["nmatch", "maxmatch", "top5share"])

    def _raw(self, M, seeds):
        N, L = M.shape; per = self.H // len(seeds)
        rows, cols = [], []
        for sid, off in enumerate(seeds):
            npos = L - off[-1]
            v = np.full((N, npos), (sid + 1) * 2654435761, dtype=np.int64)
            for o in off:
                v = (v * 131 + M[:, o:o + npos]) & 0x7FFFFFFFFFFF
            cols.append((sid * per + (((v * 0x9E3779B1) >> 13) % per)).ravel())
            rows.append(np.repeat(np.arange(N), npos))
        X = sp.coo_matrix((np.ones(sum(len(c) for c in cols), dtype=np.float32),
                           (np.concatenate(rows), np.concatenate(cols))), shape=(N, self.H)).tocsr()
        X.sum_duplicates(); return X

    def fit(self, texts):
        M = codes(texts)
        for key, seeds, Mx in (("L", self.seeds, M), ("S", self.skseeds, skeleton(M))):
            X = self._raw(Mx, seeds)
            df = np.asarray((X > 0).sum(0)).ravel(); n = X.shape[0]
            keep = df >= self.min_df
            idf = np.zeros(self.H, dtype=np.float32)
            idf[keep] = np.log((1.0 + n) / (1.0 + df[keep])) + 1.0
            setattr(self, f"idf_{key}", idf)
            if key == "L":
                per = self.H // len(seeds)
                fam = np.full(self.H, -1, dtype=np.int8)
                fmap = {w: i for i, w in enumerate(self.fams)}
                for sid, off in enumerate(seeds):
                    fam[sid * per:(sid + 1) * per] = fmap[len(off)]
                self.fam = fam
                q = np.quantile(idf[keep], np.linspace(0, 1, self.nbuck + 1)[1:-1])
                self.buck = np.where(keep, np.digitize(idf, q), -1).astype(np.int8)
        self.nfeat = len(self.names)
        return self

    def _vec(self, texts, key, seeds):
        M = codes(texts)
        X = self._raw(skeleton(M) if key == "S" else M, seeds).tocsr()
        X.data = 1.0 + np.log(X.data)
        X = X.multiply(getattr(self, f"idf_{key}")[None, :]).tocsr(); X.eliminate_zeros()
        return X

    def vectors(self, texts):
        """Unnormalised tf-idf rows plus the per-family norms needed for family cosines."""
        XL = self._vec(texts, "L", self.seeds)
        nf = len(self.fams)
        fn = np.zeros((XL.shape[0], nf + 1), dtype=np.float32)
        for r in range(XL.shape[0]):
            s, e = XL.indptr[r], XL.indptr[r + 1]
            d2 = XL.data[s:e] ** 2
            fn[r, :nf] = np.bincount(self.fam[XL.indices[s:e]], weights=d2, minlength=nf)[:nf]
            fn[r, nf] = d2.sum()
        fn = np.sqrt(fn); fn[fn == 0] = 1.0
        XS = self._vec(texts, "S", self.skseeds)
        sn = np.sqrt(np.asarray(XS.multiply(XS).sum(1))).ravel(); sn[sn == 0] = 1.0
        return dict(L=XL, fn=fn, S=XS, sn=sn)

    def profiles(self, Vq, Vc, pools):
        Q, P = pools.shape
        nf = len(self.fams); F = np.zeros((Q, P, self.nfeat), dtype=np.float32)
        A, Bm, An, Bn = Vq["L"], Vc["L"], Vq["fn"], Vc["fn"]
        buf = np.zeros(self.H, dtype=np.float32)
        for q in range(Q):
            s, e = A.indptr[q], A.indptr[q + 1]
            ai = A.indices[s:e]; buf[ai] = A.data[s:e]
            for p in range(P):
                r = pools[q, p]
                s2, e2 = Bm.indptr[r], Bm.indptr[r + 1]
                bi = Bm.indices[s2:e2]
                prod = buf[bi] * Bm.data[s2:e2]
                nz = prod > 0
                if not nz.any(): continue
                pv, idxs = prod[nz], bi[nz]
                dotf = np.bincount(self.fam[idxs], weights=pv, minlength=nf)[:nf]
                tot = pv.sum()
                F[q, p, 0] = tot / (An[q, nf] * Bn[r, nf])
                F[q, p, 1:1 + nf] = dotf / (An[q, :nf] * Bn[r, :nf])
                bk = self.buck[idxs]; ok = bk >= 0
                if ok.any():
                    F[q, p, 2 + nf:2 + nf + self.nbuck] = \
                        np.bincount(bk[ok], weights=pv[ok], minlength=self.nbuck)[:self.nbuck] / \
                        (An[q, nf] * Bn[r, nf])
                k = 2 + nf + self.nbuck
                F[q, p, k] = len(pv)
                F[q, p, k + 1] = pv.max() / (An[q, nf] * Bn[r, nf])
                F[q, p, k + 2] = np.sort(pv)[-5:].sum() / max(tot, 1e-9)
            buf[ai] = 0.0
        XSq, XSc, snq, snc = Vq["S"], Vc["S"], Vq["sn"], Vc["sn"]
        for q in range(Q):
            s, e = XSq.indptr[q], XSq.indptr[q + 1]
            ai = XSq.indices[s:e]; buf[ai] = XSq.data[s:e]
            for p in range(P):
                r = pools[q, p]
                s2, e2 = XSc.indptr[r], XSc.indptr[r + 1]
                bi = XSc.indices[s2:e2]
                F[q, p, 1 + nf] = (buf[bi] * XSc.data[s2:e2]).sum() / (snq[q] * snc[r])
            buf[ai] = 0.0
        return F
