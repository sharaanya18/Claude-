"""Whose Second Half Is It -- match six listening-day prefixes to six shuffled continuations.

Requirements map (what the challenge asks -> where this script does it)
- CPU only, one command `python3 solution.py <public_dir> <submission_out>`   -> argv contract, fixed thread counts
- Structure must be LEARNED from training material only                        -> collaborative-filtering affinities are
  fitted on non-test sessions (train-row days in full + every other published full day); a LightGBM pair scorer is
  trained on the 1,081 labelled train rows; the 6x6 assignment is decoded jointly per row.
- No fitting on test first halves / other test rows / test-wide statistics      -> test sessions are only ever used as
  one row's six query prefixes. Every normaliser (idf, background score statistics, popularity counts) is computed on
  fit sessions, never on test sessions or test candidates.
- ids, token spellings, file/row order are not used as signals                  -> only plays, releases, recordings.
- Fixed work plan, no wall-clock branching, seeded, no hardware branches.

Pipeline
1. Affinity scores between a prefix and a candidate artist from user-kNN collaborative filtering: cosine similarity of
   the prefix bag of artists (or releases) to every fit day, then the similarity-weighted count of the candidate artist
   in those days.  Three kNN variants, a sequential artist->artist transition score and listener-trait features
   (how obscure a listener's recordings are, missing-release habit).
2. Every score is standardised against a background of random fit prefixes (removes candidate popularity), then
   normalised within the row (z-scores and ranks along both axes).
3. LightGBM binary pair scorer on those features, out-of-fold features built with the held-out rows' sessions removed
   from the fit set.
4. Joint decoding: posterior over the 720 permutations (temperature fitted on OOF), then the assignment that
   maximises expected correct matches.
"""
import os, sys, random, time, itertools
from pathlib import Path

PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
SUBMISSION_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")
SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)

N_THREADS = 10                      # fixed thread cap (never derived from the machine)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS)
os.environ["PYTHONHASHSEED"] = "0"

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize
import lightgbm as lgb
import torch

SEED = 42
N_FOLDS = 5                 # row folds for out-of-fold features / honest CV
N_BG = 2000                 # background prefixes per fit set (score standardisation)
LGB_ROUNDS = 400
LGB_SEEDS = (0, 1, 2)
TRANS_WINDOW, TRANS_DECAY = 10, 0.8
T0 = time.time()


def log(msg):               # elapsed time is telemetry only, never used in a condition
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(N_THREADS)


# --------------------------------------------------------------------------------------------- data
class Data:
    """Index plays by session/artist/release/recording and build the sparse session x token matrices."""

    def __init__(self, L, C, test_sessions):
        self.L = L
        self.aid = {a: i for i, a in enumerate(L.artist.unique())}
        self.NA = len(self.aid)
        self.sid = {x: i for i, x in enumerate(L.session.unique())}
        self.NS = len(self.sid)
        self.a = L.artist.map(self.aid).values
        self.s = L.session.map(self.sid).values
        self.h = L.half.values
        self.rel = pd.factorize(L.release.fillna("NA_" + L.artist))[0]
        self.rec_codes, self.rec_uni = pd.factorize(L.recording)
        self.rel_codes, self.rel_uni = pd.factorize(L.release.fillna("NA_" + L.artist))
        self.is_test = np.zeros(self.NS, bool)
        for x in test_sessions:
            self.is_test[self.sid[x]] = True
        cnt = pd.DataFrame({"s": self.s, "a": self.a, "h": self.h})
        self.Ca = self._mat(cnt.groupby(["s", "a"]).size().reset_index(name="c"), self.NA)
        self.C1 = self._mat(cnt[cnt.h == 1].groupby(["s", "a"]).size().reset_index(name="c"), self.NA)
        cr = pd.DataFrame({"s": self.s, "a": self.rel_codes, "h": self.h})
        self.Ra = self._mat(cr.groupby(["s", "a"]).size().reset_index(name="c"), self.rel_codes.max() + 1)
        self.R1 = self._mat(cr[cr.h == 1].groupby(["s", "a"]).size().reset_index(name="c"), self.rel_codes.max() + 1)
        cq = pd.DataFrame({"s": self.s, "a": self.rec_codes, "h": self.h})
        self.Qa = self._mat(cq.groupby(["s", "a"]).size().reset_index(name="c"), len(self.rec_uni))
        self.Q1 = self._mat(cq[cq.h == 1].groupby(["s", "a"]).size().reset_index(name="c"), len(self.rec_uni))
        self.n1 = np.asarray(self.C1.sum(1)).ravel()
        # continuation -> its two artists in rank order, and first-play recording / release / artist codes
        C = C.sort_values(["continuation", "rank"])
        self.cc = C.groupby("continuation").artist.apply(lambda x: [self.aid[a] for a in x]).to_dict()
        self.C = C
        recmap = {r: i for i, r in enumerate(self.rec_uni)}; relmap = {r: i for i, r in enumerate(self.rel_uni)}
        self.crec = C.groupby("continuation").recording.apply(lambda x: [recmap.get(r, -1) for r in x]).to_dict()
        self.crel = C.groupby("continuation").release.apply(
            lambda x: [relmap.get(r, -1) if isinstance(r, str) else -1 for r in x]).to_dict()
        # per-play session start offsets for fast slicing (L is sorted by session then position)
        starts = np.r_[0, np.where(self.s[1:] != self.s[:-1])[0] + 1, len(self.s)]
        self.span = {self.s[starts[i]]: (starts[i], starts[i + 1]) for i in range(len(starts) - 1)}

    def _mat(self, df, ncol):
        return sp.csr_matrix((df.c.values.astype(np.float32), (df.s.values, df.iloc[:, 1].values)), shape=(self.NS, ncol))

    def prefix_plays(self, sess):
        a, b = self.span[sess]
        k = a + int((self.h[a:b] == 1).sum())
        return a, k


def idf_diag(X, fit_idx, w):
    df = np.asarray((X[fit_idx] > 0).sum(0)).ravel()
    return sp.diags((np.log((len(fit_idx) + 1) / (df + 1)) ** w).astype(np.float32))


def transitions(D, fit_mask):
    """Directed artist->artist counts for plays up to TRANS_WINDOW steps apart inside fit sessions."""
    A = None
    ok = fit_mask[D.s]
    for d in range(1, TRANS_WINDOW + 1):
        m = (D.s[:-d] == D.s[d:]) & ok[:-d]
        M = sp.csr_matrix((np.full(int(m.sum()), TRANS_DECAY ** (d - 1), np.float32), (D.a[:-d][m], D.a[d:][m])),
                          shape=(D.NA, D.NA))
        A = M if A is None else A + M
    return A.tocsr()


# ----------------------------------------------------------------------------------------- features
KNN_CONFIGS = [   # (name, token space, tf transform, idf power, similarity exponent)
    ("art_bin", "art", "bin", 0.5, 1.0),
    ("art_log", "art", "log", 1.0, 0.5),
    ("rel_log", "rel", "log", 0.5, 1.0),
]
TRAITS = ["lrec", "lrel", "lart", "hit", "emp"]


def tf_apply(X, kind):
    if kind == "bin":
        X = X.copy(); X.data[:] = 1.0; return X
    return X.log1p()


def knn_scores(D, fit_idx, query_idx, Tcols, cfg, exclude_self=False, Qmat=None):
    """Similarity-weighted candidate-artist counts for each query prefix (rows) x candidate column."""
    name, space, tf, w, gam = cfg
    P1, Pall = (D.C1, D.Ca) if space == "art" else (D.R1, D.Ra)
    W = idf_diag(Pall, fit_idx, w)
    F = normalize(tf_apply(Pall[fit_idx], tf) @ W)
    Q = normalize(tf_apply(P1[query_idx] if Qmat is None else Qmat, tf) @ W)
    K = (Q @ F.T).toarray()
    if exclude_self:
        pos = {x: i for i, x in enumerate(fit_idx)}
        for r, x in enumerate(query_idx):
            K[r, pos[x]] = 0.0
    K = K ** gam
    return np.asarray(Tcols.T.dot(K.T)).T


def nb_scores(D, fit_idx, query_idx, cols, alpha=20.0, space="art", Qmat=None):
    """Naive-Bayes style item-item score: mean over the prefix artists a of log[P(b|a)/P(b)], with P(b|a) estimated from
    session co-occurrence in fit days and shrunk towards P(b) with pseudo-count alpha."""
    Xall, X1 = {"art": (D.Ca, D.C1), "rel": (D.Ra, D.R1), "rec": (D.Qa, D.Q1)}[space]
    Xf = Xall[fit_idx].copy(); Xf.data[:] = 1.0
    N = len(fit_idx); df = np.asarray(Xf.sum(0)).ravel()
    Pm = (X1[query_idx] if Qmat is None else Qmat).copy(); Pm.data[:] = 1.0
    A = np.unique(Pm.indices); Pm = Pm[:, A].tocsr()
    G = (Xf[:, A].T @ Xf[:, cols]).tocoo()
    pb = np.maximum(df[cols], 0.5) / N; dfa = df[A]
    obs = np.log((G.data + alpha * pb[G.col]) / (dfa[G.row] + alpha) / pb[G.col]) - np.log(alpha / (dfa[G.row] + alpha))
    Lm = sp.csr_matrix((obs, (G.row, G.col)), shape=(len(A), len(cols)))
    S = (Pm @ Lm).toarray() + np.asarray(Pm @ np.log(alpha / (dfa + alpha)))[:, None]
    return S / np.maximum(np.asarray(Pm.sum(1)), 1)


def trait_arrays(D, fit_mask, C):
    """Listener-trait values per play and per continuation first play; counts come from fit sessions only."""
    m = fit_mask[D.s]
    rec_c = np.bincount(D.rec_codes[m], minlength=len(D.rec_uni))
    rel_c = np.bincount(D.rel_codes[m], minlength=len(D.rel_uni))
    art_c = np.bincount(D.a[m], minlength=D.NA)
    pl = {"lrec": np.log1p(rec_c[D.rec_codes]), "lrel": np.log1p(rel_c[D.rel_codes]), "lart": np.log1p(art_c[D.a])}
    pl["hit"] = pl["lrec"] - pl["lart"]
    pl["emp"] = D.L.release.isna().values.astype(np.float32)
    recmap = {r: i for i, r in enumerate(D.rec_uni)}
    relmap = {r: i for i, r in enumerate(D.rel_uni)}
    crec = C.recording.map(recmap).fillna(-1).astype(int).values
    crel = np.array([relmap.get(r, -1) if isinstance(r, str) else -1 for r in C.release.values])
    cart = C.artist.map(D.aid).values
    ca = {"lrec": np.log1p(np.where(crec >= 0, rec_c[np.clip(crec, 0, None)], 0)), "lart": np.log1p(art_c[cart]),
          "lrel": np.log1p(np.where(crel >= 0, rel_c[np.clip(crel, 0, None)], 0))}
    ca["hit"] = ca["lrec"] - ca["lart"]
    ca["emp"] = C.release.isna().values.astype(np.float32)
    cdf = pd.DataFrame(ca); cdf["continuation"] = C.continuation.values
    cvals = {c: g[TRAITS].to_numpy() for c, g in cdf.groupby("continuation", sort=False)}   # rank order preserved
    return pl, cvals


def rowwise(v):
    """Within-row normalisations of a (6 prefixes, 6 candidates) matrix: z-score and rank along both axes."""
    out = []
    for ax in (0, 1):
        m = v.mean(ax, keepdims=True); s = v.std(ax, keepdims=True) + 1e-6
        out += [(v - m) / s, np.argsort(np.argsort(v, ax), ax).astype(np.float32)]
    return out


def exclusive_bags(D, Pidx):
    """Per row: each prefix keeps only the artists that occur in no other prefix of the same row (what makes that
    person different from the five look-alikes).  Uses only the row's own six prefixes."""
    rows, cols_, vals = [], [], []
    for r0 in range(0, len(Pidx), 6):
        bags = [D.C1[x].tocoo() for x in Pidx[r0:r0 + 6]]
        cnt = {}
        for b in bags:
            for a in b.col:
                cnt[a] = cnt.get(a, 0) + 1
        for j, b in enumerate(bags):
            for a, v in zip(b.col, b.data):
                if cnt[a] == 1:
                    rows.append(r0 + j); cols_.append(a); vals.append(v)
    return sp.csr_matrix((vals, (rows, cols_)), shape=(len(Pidx), D.NA), dtype=np.float32)


def rowwise_all(v):
    return rowwise(v)


def build_features(D, R, fit_idx, bg_idx):
    """Pair features (n_rows, 6 prefixes, 6 candidates, F) for rows R using only fit sessions as training material."""
    fit_mask = np.zeros(D.NS, bool); fit_mask[fit_idx] = True
    Pidx = [D.sid[x] for l in R.pre for x in l]
    cols = sorted({a for c in R.cand for ci in c for a in D.cc[ci]}); cm = {a: i for i, a in enumerate(cols)}
    Tcols = D.Ca[fit_idx][:, cols].tocsr(); Tcols.data[:] = 1.0                       # candidate indicator: artist present in that fit day
    # background-standardised log kNN scores (scale-free, popularity-free)
    Z = {}
    for cfg in KNN_CONFIGS:
        S = knn_scores(D, fit_idx, Pidx, Tcols, cfg)
        B = knn_scores(D, fit_idx, bg_idx, Tcols, cfg, exclude_self=True)
        eps = 0.1 * np.median(B[B > 0]) if (B > 0).any() else 1e-6
        LB = np.log(B + eps)
        Z[cfg[0]] = (np.log(S + eps) - LB.mean(0)) / (LB.std(0) + 1e-6)
    # sequential transition affinity (artist->artist within 10 plays), divided by candidate in-degree
    A = transitions(D, fit_mask)
    colsum = np.asarray(A.sum(0)).ravel()[cols] + 1e-9
    rr, cc_ = [], []
    for pi, x in enumerate(Pidx):
        a, k = D.prefix_plays(x); rr += [pi] * (k - a); cc_ += list(D.a[a:k])
    Pm = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc_)), shape=(len(Pidx), D.NA))
    Pm = sp.diags(1 / np.asarray(Pm.sum(1)).ravel()) @ Pm
    TR = np.log((Pm @ A[:, cols]).toarray() / colsum + 1e-3)
    EX = exclusive_bags(D, Pidx)
    NB = {"art20": nb_scores(D, fit_idx, Pidx, cols), "art3": nb_scores(D, fit_idx, Pidx, cols, alpha=3.0),
          "ex20": nb_scores(D, fit_idx, Pidx, cols, Qmat=EX)}
    KX = knn_scores(D, fit_idx, Pidx, Tcols, KNN_CONFIGS[1], Qmat=EX)
    # candidate-centric evidence: fit days that contain BOTH artists of a continuation (consecutive new artists of one
    # person), weighted by how similar the prefix is to those days
    ucand = sorted({c for cl in R.cand for c in cl}); um = {c: i for i, c in enumerate(ucand)}
    Tc = Tcols.tocsc()
    pair_cols = [Tc[:, cm[D.cc[c][0]]].multiply(Tc[:, cm[D.cc[c][1]]]) for c in ucand]
    Tpair = sp.hstack(pair_cols).tocsr()
    KP = knn_scores(D, fit_idx, Pidx, Tpair, KNN_CONFIGS[1])
    pl, cvals = trait_arrays(D, fit_mask, D.C)
    out = []
    for ri, (_, r) in enumerate(R.iterrows()):
        sl = slice(ri * 6, ri * 6 + 6)
        cidx = np.array([[cm[a] for a in D.cc[c]] for c in r.cand])      # (6 candidates, 2 artists)
        F = []
        for nm in Z:
            q = np.stack([Z[nm][sl][:, cidx[:, 0]], Z[nm][sl][:, cidx[:, 1]]], -1)    # (6,6,2)
            mean, mx = q.mean(-1), q.max(-1)
            F += [mean, mx] + rowwise(mean) + rowwise(mx)[:2]
        for key in ("art20", "art3", "ex20"):
            nq = np.stack([NB[key][sl][:, cidx[:, 0]], NB[key][sl][:, cidx[:, 1]]], -1)
            nmean = nq.mean(-1)
            F += [nmean] + rowwise(nmean)
        kxq = np.log(np.stack([KX[sl][:, cidx[:, 0]], KX[sl][:, cidx[:, 1]]], -1).mean(-1) + 1e-3)
        F += [kxq] + rowwise(kxq)
        kpq = np.log(np.stack([KP[sl][:, um[c]] for c in r.cand], -1) + 1e-3)
        F += [kpq] + rowwise(kpq)
        tq = np.stack([TR[sl][:, cidx[:, 0]], TR[sl][:, cidx[:, 1]]], -1).mean(-1)
        F += [tq] + rowwise(tq)
        # listener traits: prefix distribution vs candidate first plays
        cv = np.stack([cvals[c] for c in r.cand])                          # (6 cand, 2 ranks, 5 traits)
        for ti, t in enumerate(TRAITS):
            zd = np.zeros((6, 6)); ad = np.zeros((6, 6)); pc = np.zeros((6, 6)); pmv = np.zeros((6, 6))
            for j, x in enumerate(r.pre):
                a, k = D.prefix_plays(D.sid[x]); v = pl[t][a:k]
                pm = v.mean(); ps = v.std() + 0.3; c = cv[:, :, ti]; cmn = c.mean(1)
                zd[j] = (cmn - pm) / ps; ad[j] = np.abs(cmn - pm)
                pc[j] = ((v[None, :] < c[:, :1]).mean(1) + (v[None, :] < c[:, 1:2]).mean(1)) / 2; pmv[j] = pm
            F += [zd, ad, pc, pmv]
        out.append(np.stack(F, -1))
    return np.stack(out).astype(np.float32)




# ------------------------------------------------------------------------ learned histogram kernel
LK_TYPES = ("art_bin", "art_log", "rel_log")
LK_NB = 14                                      # similarity bins per kernel type
LK_EDGES = np.geomspace(0.01, 0.6, LK_NB - 1)   # log-spaced cosine edges
LK_SIZE_EDGES = np.log([12, 30])                # neighbour-day length buckets (3)


class HistKernel:
    """Counts, per (prefix, candidate artist), the fit days that contain the artist, binned by the day's similarity to the
    prefix (3 similarity types x 14 bins x 3 day-length buckets).  A non-negative weight per bin is then LEARNED by
    maximising the likelihood of the true matching (end-to-end learned neighbour weighting)."""

    def __init__(self, D, fit_mask):
        self.D, self.fit_mask = D, fit_mask
        fi = np.where(fit_mask)[0]
        cfg = {"art_bin": (D.C1, D.Ca, "bin", 0.5), "art_log": (D.C1, D.Ca, "log", 1.0), "rel_log": (D.R1, D.Ra, "log", 0.5)}
        self.Q, self.F = {}, {}
        for t in LK_TYPES:
            P1, Pa, kind, w = cfg[t]
            W = idf_diag(Pa, fi, w)
            self.F[t] = normalize(tf_apply(Pa, kind) @ W).tocsr()
            self.Q[t] = normalize(tf_apply(P1, kind) @ W).tocsr()
        self.size_bucket = np.digitize(np.log(np.maximum(D.Ca.getnnz(1), 1)), LK_SIZE_EDGES)
        self.Tcsc = D.Ca.tocsc()

    def row(self, pre_sessions, cand_artists, own_sessions):
        """Returns (6 prefixes, len(cand_artists), 126) counts; own_sessions are excluded from the neighbours."""
        out = np.zeros((6, len(cand_artists), len(LK_TYPES), LK_NB, 3), np.float32)
        lists = []
        for a in cand_artists:
            ss = self.Tcsc.indices[self.Tcsc.indptr[a]:self.Tcsc.indptr[a + 1]]
            ss = ss[self.fit_mask[ss]]
            lists.append(ss[~np.isin(ss, own_sessions)])
        U = np.unique(np.concatenate(lists))
        if len(U) == 0:
            return out.reshape(6, len(cand_artists), -1)
        pos = {x: i for i, x in enumerate(U)}
        for ti, t in enumerate(LK_TYPES):
            bins = np.digitize((self.Q[t][pre_sessions] @ self.F[t][U].T).toarray(), LK_EDGES)
            for ai, ss in enumerate(lists):
                if len(ss) == 0:
                    continue
                ix = np.array([pos[x] for x in ss]); sb = self.size_bucket[ss]
                for k in range(3):
                    m = sb == k
                    if m.any():
                        for j in range(6):
                            out[j, ai, ti, :, k] = np.bincount(bins[j][ix[m]], minlength=LK_NB)[:LK_NB]
        return out.reshape(6, len(cand_artists), -1)


def lk_features(D, hk, R, exclude_own):
    """(n_rows, 6, 6, 126): bin counts averaged over the two artists of each candidate."""
    out = []
    for _, r in R.iterrows():
        pre = [D.sid[x] for x in r.pre]
        arts = [a for c in r.cand for a in D.cc[c]]
        f = hk.row(pre, arts, np.array(pre) if exclude_own else np.array([], int))
        out.append(f.reshape(6, 6, 2, -1).mean(2))
    return np.stack(out)


def lk_fit(X, idx, l2=0.1, steps=300, lr=0.05):
    """Learn non-negative bin weights by maximising the exact likelihood of the true 6x6 matching (720 permutations)."""
    X = torch.tensor(X).float(); idx = torch.tensor(idx)
    theta = torch.nn.Parameter(torch.randn(X.shape[-1]) * 0.1 - 2.0); leps = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam([theta, leps], lr=lr); P = torch.tensor(PERMS); ar = torch.arange(6)
    for _ in range(steps):
        opt.zero_grad()
        Z = torch.log((X * torch.nn.functional.softplus(theta)).sum(-1) + torch.exp(leps))
        ll = Z[:, ar[None, :], P].sum(-1)
        loss = -(ll.gather(1, idx[:, None]).squeeze(1) - torch.logsumexp(ll, 1)).mean() \
            + l2 * (torch.nn.functional.softplus(theta) ** 2).sum()
        loss.backward(); opt.step()
    return theta.detach(), leps.detach()


def lk_scores(model, X):
    theta, leps = model
    with torch.no_grad():
        return torch.log((torch.tensor(X).float() * torch.nn.functional.softplus(theta)).sum(-1) + torch.exp(leps)).numpy()


# ------------------------------------------------------------------ release-graph links (metadata, not co-listening)
class ReleaseGraph:
    """artist x release incidence from fit days: links an artist credit variant / compilation partner to a prefix even when
    no one listened to both."""

    def __init__(self, D, fit_mask):
        self.D = D
        d = pd.DataFrame({"s": D.s, "a": D.a, "r": D.rel_codes}); d = d[fit_mask[d.s]].drop_duplicates()
        self.AR = sp.csr_matrix((np.ones(len(d), np.float32), (d.a.values, d.r.values)), shape=(D.NA, len(D.rel_uni)))

    def row(self, r, exclude_own):
        D = self.D
        sess = [D.sid[x] for x in r.pre]
        AR = self.AR
        if exclude_own:                      # remove the row's own sessions' contribution (leave-row-out)
            m = np.isin(D.s, sess)
            dd = pd.DataFrame({"a": D.a[m], "r": D.rel_codes[m], "s": D.s[m]}).drop_duplicates()
            OW = sp.csr_matrix((np.ones(len(dd), np.float32), (dd.a.values, dd.r.values)), shape=AR.shape)
            AR = (AR - OW).tocsr(); AR.data = np.maximum(AR.data, 0); AR.eliminate_zeros()
        res = np.zeros((6, 6, 2), np.float32)
        cand_rel = [[set(AR[b].indices) for b in D.cc[c]] for c in r.cand]
        for j, x in enumerate(sess):
            a0, k = D.prefix_plays(x)
            rels = set(D.rel_codes[a0:k]); Rset = set(np.unique(AR[np.unique(D.a[a0:k])].indices))
            for kk in range(6):
                res[j, kk, 0] = sum(len(rels & rb) > 0 for rb in cand_rel[kk])
                res[j, kk, 1] = sum(len(Rset & rb) > 0 for rb in cand_rel[kk])
        return res


def release_features(D, rg, R, exclude_own):
    return np.log1p(np.stack([rg.row(r, exclude_own) for _, r in R.iterrows()]))

# --------------------------------------------------------------------------------------- model/decode
LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=N_THREADS,
                  deterministic=True, force_row_wise=True)


def fit_pair_model(X, y):
    F = X.shape[-1]
    models = []
    for sd in LGB_SEEDS:
        p = dict(LGB_PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd)
        models.append(lgb.train(p, lgb.Dataset(X.reshape(-1, F), y.reshape(-1)), num_boost_round=LGB_ROUNDS))
    return models


def predict_pair(models, X):
    F = X.shape[-1]
    return np.mean([m.predict(X.reshape(-1, F), raw_score=True) for m in models], 0).reshape(-1, 6, 6)


PERMS = np.array(list(itertools.permutations(range(6))))          # 720 x 6, perms[p][i] = candidate of prefix i


def perm_marginals(Z, temp):
    """Posterior over permutations from pair logits Z (6x6): returns P[i, k] = prob(prefix i -> candidate k)."""
    ll = Z[np.arange(6)[None, :], PERMS].sum(1) * temp
    p = np.exp(ll - ll.max()); p /= p.sum()
    M = np.zeros((6, 6))
    for i in range(6):
        np.add.at(M[i], PERMS[:, i], p)
    return M


def decode(Z, temp):
    M = perm_marginals(Z, temp)
    r, c = linear_sum_assignment(-M)
    out = np.zeros(6, int); out[r] = c
    return out


def true_perm_nll(Zs, ys, temp):
    tot = 0.0
    for Z, y in zip(Zs, ys):
        ll = Z[np.arange(6)[None, :], PERMS].sum(1) * temp
        tot += -(ll[np.where((PERMS == y[None, :]).all(1))[0][0]] - (ll.max() + np.log(np.exp(ll - ll.max()).sum())))
    return tot / len(Zs)


def chance_corrected(pred, true):
    A = (np.asarray(pred) == np.asarray(true)).mean()
    return (A - 1 / 6) / (1 - 1 / 6)


# --------------------------------------------------------------------------------------------- main
def validate_submission(sub, sample):
    assert list(sub.columns) == list(sample.columns), "columns"
    assert len(sub) == len(sample) and sub.id.astype(str).tolist() == sample.id.astype(str).tolist(), "ids/order"
    v = sub.iloc[:, 1:].to_numpy()
    assert np.isfinite(v).all() and v.min() >= 1 and v.max() <= 6 and (v == v.astype(int)).all(), "values"
    assert all(sorted(r) == [1, 2, 3, 4, 5, 6] for r in v.astype(int)), "row not a permutation"


def main():
    L = pd.read_csv(PUBLIC_DIR / "listens.csv").sort_values(["session", "relative_position"]).reset_index(drop=True)
    C = pd.read_csv(PUBLIC_DIR / "continuations.csv")
    tr = pd.read_csv(PUBLIC_DIR / "train.csv"); te = pd.read_csv(PUBLIC_DIR / "test.csv")
    sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv")
    for d in (tr, te):
        d["pre"] = d.prefixes.map(json.loads); d["cand"] = d.candidates.map(json.loads)
    Y = tr[[f"match_{i}" for i in range(1, 7)]].to_numpy() - 1
    D = Data(L, C, set(x for l in te.pre for x in l))
    log(f"data: {D.NS} sessions, {D.NA} artists; train rows {len(tr)}, test rows {len(te)}")
    rng = np.random.RandomState(SEED)
    fold = rng.randint(0, N_FOLDS, len(tr))

    # ---- out-of-fold features for the labelled rows (held-out rows' sessions are removed from the fit set)
    Xtr = None
    for f in range(N_FOLDS):
        R = tr[fold == f]
        held = np.zeros(D.NS, bool)
        for l in R.pre:
            for x in l:
                held[D.sid[x]] = True
        fit_idx = np.where(~D.is_test & ~held)[0]
        bg = rng.choice(fit_idx[D.n1[fit_idx] >= 8], N_BG, replace=False)
        F = build_features(D, R, fit_idx, bg)
        if Xtr is None:
            Xtr = np.zeros((len(tr),) + F.shape[1:], np.float32)
        Xtr[fold == f] = F
        log(f"features fold {f}: {F.shape}")
    # ---- learned histogram kernel + release-graph links (leave-row-out: a row's own six days never serve as neighbours)
    fit_all = ~D.is_test
    hk = HistKernel(D, fit_all); rg = ReleaseGraph(D, fit_all)
    LKtr = lk_features(D, hk, tr, True); RLtr = release_features(D, rg, tr, True)
    perm_id = {tuple(p): i for i, p in enumerate(PERMS)}
    ptr = np.array([perm_id[tuple(y)] for y in Y])
    Ztr = np.zeros((len(tr), 6, 6), np.float32)
    for f in range(N_FOLDS):                                  # out-of-fold learned-kernel scores for the labelled rows
        Ztr[fold == f] = lk_scores(lk_fit(LKtr[fold != f], ptr[fold != f]), LKtr[fold == f])
    log(f"learned-kernel OOF (Hungarian only): {chance_corrected(np.array([linear_sum_assignment(-z)[1] for z in Ztr]), Y):.4f}")

    def extra(Z, RL):
        parts = [Z[..., None]] + [v[..., None] for v in rowwise_all(Z)] + [RL]
        return np.concatenate(parts, -1).astype(np.float32)
    Xtr = np.concatenate([Xtr, np.stack([extra(Ztr[i], RLtr[i]) for i in range(len(tr))])], -1)
    ytr = np.zeros((len(tr), 6, 6), np.float32)
    for i in range(len(tr)):
        ytr[i, np.arange(6), Y[i]] = 1

    # ---- honest CV of the pair scorer (features were built out-of-fold), and decode temperature on OOF
    oof = np.zeros((len(tr), 6, 6))
    for f in range(N_FOLDS):
        oof[fold == f] = predict_pair(fit_pair_model(Xtr[fold != f], ytr[fold != f]), Xtr[fold == f])
        log(f"cv fold {f} done")
    temps = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    nll = [true_perm_nll(oof, Y, t) for t in temps]; temp = temps[int(np.argmin(nll))]
    hung = np.array([linear_sum_assignment(-z)[1] for z in oof])
    post = np.array([decode(z, temp) for z in oof])
    per_fold = [chance_corrected(post[fold == f], Y[fold == f]) for f in range(N_FOLDS)]
    log(f"OOF score: argmax {chance_corrected(oof.argmax(2), Y):.4f} hungarian {chance_corrected(hung, Y):.4f} "
        f"posterior-decode {chance_corrected(post, Y):.4f} (temp {temp}); per-fold {np.round(per_fold, 4)} "
        f"mean {np.mean(per_fold):.4f} std {np.std(per_fold):.4f}")

    # ---- final model on all labelled rows; test features use every non-test session as fit material
    models = fit_pair_model(Xtr, ytr)
    fit_idx = np.where(~D.is_test)[0]
    bg = rng.choice(fit_idx[D.n1[fit_idx] >= 8], N_BG, replace=False)
    Xte = build_features(D, te, fit_idx, bg)
    Zte = lk_scores(lk_fit(LKtr, ptr), lk_features(D, hk, te, False))
    RLte = release_features(D, rg, te, False)
    Xte = np.concatenate([Xte, np.stack([extra(Zte[i], RLte[i]) for i in range(len(te))])], -1)
    Zpair = predict_pair(models, Xte)
    pred = np.array([decode(z, temp) for z in Zpair]) + 1
    sub = pd.DataFrame(pred, columns=[f"match_{i}" for i in range(1, 7)])
    sub.insert(0, "id", te.id.values)
    sub = sample[["id"]].merge(sub, on="id", how="left")
    validate_submission(sub, sample)
    sub.to_csv(SUBMISSION_OUT, index=False)
    log(f"wrote {SUBMISSION_OUT} shape={sub.shape}")


if __name__ == "__main__":
    main()
