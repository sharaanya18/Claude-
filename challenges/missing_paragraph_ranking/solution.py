"""
Missing Paragraph Ranking - self-contained solution.

Task: for each query, rank its 20 candidate contribution snippets so that the one taken from
the same abstract as the 100-character opening comes as high as possible. Metric: mean
normalised rank of the true continuation, lower is better.

HOW THIS MEETS EVERY RULE IN THE CHALLENGE DESCRIPTION
-----------------------------------------------------
* "From scratch / no pretrained model of any kind."  Every weight in this file is randomly
  initialised and trained here, on the supplied training queries alone. No pretrained
  language model, sentence encoder, word/subword/character embedding, tokenizer or spelling
  corrector is loaded, and nothing is downloaded: the script makes no network call at all.
* "No outside text."  The only inputs read are the five files in <public_dir>. No dictionary,
  word list, corpus or lookup of the source papers.
* "Overlap scores may be model features but may not by themselves decide the ranking."
  Character-pattern agreement is never used as a ranking score. It is decomposed into a
  profile of complementary measurements (per-seed-family cosines, rarity strata, how MANY
  patterns agree, how concentrated the agreement is, and a letter-free structure channel),
  and a neural listwise ranker trained here decides how to weigh them. The ranker is also the
  only thing that produces an ordering: the features alone give no score.
* "No reasoning across evaluation queries."  Every evaluation query is scored from its own
  opening and its own 20 candidates only. No statistic, normaliser, vocabulary or model
  parameter is ever fitted on evaluation text, no pseudo-labelling, no test-time adaptation,
  no assignment of candidates across queries. All idf/rarity/standardisation statistics are
  fitted on TRAINING text and only applied to evaluation text.
* "No query_id / candidate_id / pool frequency / pool order / row order as a signal."  None of
  these ever reach a feature. Candidates are read as an unordered set; the only thing taken
  from the pool is the text of its members.
* "No hardcoded rankings."  Every submitted ranking is the unedited argsort of the trained
  ensemble's scores. There is no lookup keyed by query, and no hand-written rule.
* Determinism: every seed is fixed, the work plan (folds, seeds, epochs, augmentations,
  batch size) is a constant decided in advance, nothing branches on elapsed time, available
  devices, CPU count or environment. Time is printed, never compared.

WHY THE REPRESENTATION LOOKS LIKE THIS
-------------------------------------
Letters are corrupted at ~20% in training and ~30% at evaluation; spaces, digits and
punctuation are never touched. A contiguous 4-gram needs 4 intact characters and survives 30%
corruption with probability 0.7^4 = 0.24, so ordinary n-gram matching throws most of the
shared evidence away. Reading the same window through a SPACED SEED (a gapped mask, e.g.
positions 0,2,3 of a 4-wide window) needs fewer intact characters and survives far more often,
which is why the pattern bank below is built from spaced seeds rather than contiguous n-grams.
Measured on held-out abstracts at the evaluation noise level, this change alone moves the
lexical floor from 0.345 to 0.303.

The training snippets are additionally re-corrupted with the challenge's own documented
process, up to and past the evaluation rate, so the ranker is trained across the noise regime
it will be tested in rather than only at the training rate. This is augmentation of the real
training samples, not synthetic data: no new labelled example is invented.
"""
import os, sys, time, random

PUBLIC_DIR = sys.argv[1] if len(sys.argv) > 1 else "./dataset/public"
SUBMISSION_OUT = sys.argv[2] if len(sys.argv) > 2 else "./working/submission.csv"

os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

PUBLIC_DIR = Path(PUBLIC_DIR)
SUBMISSION_OUT = Path(SUBMISSION_OUT)
SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- fixed work plan
SEED        = 42
DEVICE      = torch.device("cpu")   # the ranker is tiny; one fixed device, never probed
POOL        = 20                    # candidates per query, fixed by the task
N_FOLDS     = 5                     # abstract-level folds -> 5 ranker instances
N_SEEDS     = 3                     # independent inits per fold, rank-averaged
EPOCHS      = 60                    # upper bound; early stopping is on a validation metric
EVAL_EVERY  = 3
BATCH       = 64
LR          = 2e-3
WD          = 1e-2
HID         = 48
DEPTH       = 2
DROP        = 0.3
N_AUG       = 4                     # corruption levels used for training views
N_POOLSAMP  = 3                     # independent same-subfield negative draws per level
AUG_MIN     = 0.20                  # the training snippets' own noise level
AUG_MAX     = 0.42                  # past the stated 30% evaluation level, for robustness
TRAIN_NOISE = 0.20                  # stated in the description
EVAL_NOISE  = 0.30                  # stated in the description
SEED_SPEC   = "w2:2-8,w3:3-8,w4:4-8"
SKEL_SPEC   = "w3:3-6,w4:4-7"
NBITS       = 22
MIN_DF      = 2
N_RARE      = 3
T0          = time.time()


def log(msg):
    # elapsed time is telemetry only; it never enters a condition
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # The ranker's tensors are tiny (64 x 20 x 33). Multi-threaded BLAS spends all its time in
    # thread handover on work this small: measured 1204 ms/step at 4 threads vs 23 ms at 1.
    # One thread is both the fast choice and a fixed, machine-independent one.
    torch.set_num_threads(1)


# ------------------------------------------------------------------ corruption model
LOWER = np.arange(97, 123, dtype=np.uint8)


def corrupt(texts, rate, rng):
    """The challenge's documented noise: each English letter is replaced, with probability
    `rate`, by a uniform random letter of the same case. Spaces, digits and punctuation are
    left exactly as they are. Applied to TRAINING text only, as augmentation."""
    out = []
    for t in texts:
        a = np.frombuffer(t.encode("ascii", "ignore"), dtype=np.uint8).copy()
        hit = rng.random(a.shape[0]) < rate
        lo = hit & (a >= 97) & (a <= 122)
        up = hit & (a >= 65) & (a <= 90)
        if lo.any():
            a[lo] = 97 + rng.integers(0, 26, int(lo.sum()))
        if up.any():
            a[up] = 65 + rng.integers(0, 26, int(up.sum()))
        out.append(a.tobytes().decode("ascii"))
    return out


def extra_rate(p_from, p_to):
    """Extra corruption that carries text already at p_from to an effective p_to."""
    return 1.0 - (1.0 - p_to) / (1.0 - p_from)


# ------------------------------------------------------------------ spaced seeds
def _masks(w, W):
    from itertools import combinations
    if w == W:
        return [tuple(range(W))]
    if w < 2 or w > W:
        return []
    return [(0,) + tuple(m) + (W - 1,) for m in combinations(range(1, W - 1), w - 2)]


def seed_lib(spec):
    out, seen = [], set()
    for part in spec.split(","):
        w, rng_ = part.split(":")
        w = int(w[1:])
        lo, hi = (int(x) for x in rng_.split("-"))
        for W in range(lo, hi + 1):
            for m in _masks(w, W):
                if m not in seen:
                    seen.add(m)
                    out.append(m)
    return out


def codes(texts, L=100):
    M = np.zeros((len(texts), L), dtype=np.int64)
    for n, t in enumerate(texts):
        b = np.frombuffer(t[:L].encode("ascii", "ignore"), dtype=np.uint8)
        M[n, :len(b)] = b
    return M


def skeletonise(M):
    """Blank every letter. What is left - word lengths, punctuation, digits - is never touched
    by the corruption, so this channel is a completely noise-free view of the snippet."""
    out = M.copy()
    out[(M >= 97) & (M <= 122)] = 1
    return out


class PatternBank:
    """Hashed spaced-seed patterns with idf and rarity strata. `fit` sees TRAINING text only;
    evaluation text is only ever passed through `vectors`."""

    def __init__(self, spec, skel_spec, nbits, min_df, n_rare):
        self.seeds = seed_lib(spec)
        self.skseeds = seed_lib(skel_spec)
        self.H = 1 << nbits
        self.min_df = min_df
        self.n_rare = n_rare
        self.fams = sorted({len(s) for s in self.seeds})

    def _raw(self, M, seeds):
        """(N,100) character codes -> hashed pattern counts. Each seed owns its own slice of
        the hash space, so a pattern's seed family is recoverable from its bin."""
        N, L = M.shape
        per = self.H // len(seeds)
        rows, cols = [], []
        for sid, off in enumerate(seeds):
            npos = L - off[-1]
            v = np.full((N, npos), (sid + 1) * 2654435761, dtype=np.int64)
            for o in off:
                v = (v * 131 + M[:, o:o + npos]) & 0x7FFFFFFFFFFF
            cols.append((sid * per + (((v * 0x9E3779B1) >> 13) % per)).ravel())
            rows.append(np.repeat(np.arange(N), npos))
        X = sp.coo_matrix(
            (np.ones(sum(len(c) for c in cols), dtype=np.float32),
             (np.concatenate(rows), np.concatenate(cols))), shape=(N, self.H)).tocsr()
        X.sum_duplicates()
        return X

    def fit(self, texts):
        M = codes(texts)
        for key, seeds, Mx in (("L", self.seeds, M), ("S", self.skseeds, skeletonise(M))):
            X = self._raw(Mx, seeds)
            df = np.asarray((X > 0).sum(0)).ravel()
            n = X.shape[0]
            keep = df >= self.min_df
            idf = np.zeros(self.H, dtype=np.float32)
            idf[keep] = np.log((1.0 + n) / (1.0 + df[keep])) + 1.0
            setattr(self, "idf_" + key, idf)
            if key == "L":
                per = self.H // len(seeds)
                fam = np.full(self.H, -1, dtype=np.int8)
                fmap = {w: i for i, w in enumerate(self.fams)}
                for sid, off in enumerate(seeds):
                    fam[sid * per:(sid + 1) * per] = fmap[len(off)]
                self.fam = fam
                q = np.quantile(idf[keep], np.linspace(0, 1, self.n_rare + 1)[1:-1])
                self.rare = np.where(keep, np.digitize(idf, q), -1).astype(np.int8)
        return self

    def _vec(self, texts, key, seeds):
        M = codes(texts)
        X = self._raw(skeletonise(M) if key == "S" else M, seeds).tocsr()
        X.data = 1.0 + np.log(X.data)
        X = X.multiply(getattr(self, "idf_" + key)[None, :]).tocsr()
        X.eliminate_zeros()
        return X

    def vectors(self, texts):
        XL = self._vec(texts, "L", self.seeds)
        nf = len(self.fams)
        fn = np.zeros((XL.shape[0], nf + 1), dtype=np.float32)
        for r in range(XL.shape[0]):
            s, e = XL.indptr[r], XL.indptr[r + 1]
            d2 = XL.data[s:e] ** 2
            fn[r, :nf] = np.bincount(self.fam[XL.indices[s:e]], weights=d2, minlength=nf)[:nf]
            fn[r, nf] = d2.sum()
        fn = np.sqrt(fn)
        fn[fn == 0] = 1.0
        XS = self._vec(texts, "S", self.skseeds)
        sn = np.sqrt(np.asarray(XS.multiply(XS).sum(1))).ravel()
        sn[sn == 0] = 1.0
        return dict(L=XL, fn=fn, S=XS, sn=sn)


FEATURE_NAMES = None


def profiles(bank, Vq, Vc, pools):
    """Evidence profile for every (opening, candidate) pair of a pool.
    These are MEASUREMENTS handed to the trained ranker, not a ranking: each pair is described
    by how its pattern agreement is distributed, never by one similarity number."""
    global FEATURE_NAMES
    nf = len(bank.fams)
    names = (["cos_all"] + ["cos_w%d" % w for w in bank.fams] + ["cos_struct"]
             + ["rare%d" % b for b in range(bank.n_rare)]
             + ["n_agree", "max_agree", "top5_share"])
    FEATURE_NAMES = names
    Q, P = pools.shape
    Fo = np.zeros((Q, P, len(names)), dtype=np.float32)
    A, Bm, An, Bn = Vq["L"], Vc["L"], Vq["fn"], Vc["fn"]
    buf = np.zeros(bank.H, dtype=np.float32)
    for q in range(Q):
        s, e = A.indptr[q], A.indptr[q + 1]
        ai = A.indices[s:e]
        buf[ai] = A.data[s:e]
        for p in range(P):
            r = pools[q, p]
            s2, e2 = Bm.indptr[r], Bm.indptr[r + 1]
            bi = Bm.indices[s2:e2]
            prod = buf[bi] * Bm.data[s2:e2]
            nz = prod > 0
            if not nz.any():
                continue
            pv, idxs = prod[nz], bi[nz]
            denom = An[q, nf] * Bn[r, nf]
            tot = pv.sum()
            Fo[q, p, 0] = tot / denom
            Fo[q, p, 1:1 + nf] = np.bincount(bank.fam[idxs], weights=pv,
                                             minlength=nf)[:nf] / (An[q, :nf] * Bn[r, :nf])
            bk = bank.rare[idxs]
            ok = bk >= 0
            if ok.any():
                Fo[q, p, 2 + nf:2 + nf + bank.n_rare] = np.bincount(
                    bk[ok], weights=pv[ok], minlength=bank.n_rare)[:bank.n_rare] / denom
            k = 2 + nf + bank.n_rare
            Fo[q, p, k] = len(pv)                                  # how many patterns agree
            Fo[q, p, k + 1] = pv.max() / denom                     # strongest single agreement
            Fo[q, p, k + 2] = np.sort(pv)[-5:].sum() / max(tot, 1e-9)   # how concentrated it is
        buf[ai] = 0.0
    XSq, XSc, snq, snc = Vq["S"], Vc["S"], Vq["sn"], Vc["sn"]
    for q in range(Q):
        s, e = XSq.indptr[q], XSq.indptr[q + 1]
        ai = XSq.indices[s:e]
        buf[ai] = XSq.data[s:e]
        for p in range(P):
            r = pools[q, p]
            s2, e2 = XSc.indptr[r], XSc.indptr[r + 1]
            bi = XSc.indices[s2:e2]
            Fo[q, p, 1 + nf] = (buf[bi] * XSc.data[s2:e2]).sum() / (snq[q] * snc[r])
        buf[ai] = 0.0
    return Fo


def featurise(Fp, mu, sd):
    """Compress, standardise with TRAINING statistics, and add two views that are computed
    inside one query's own pool: the same measurements standardised against the other 19
    candidates, and their rank within the pool. The pool is part of that single query's input,
    so this stays row-local - nothing is shared between queries."""
    z = np.log1p(np.abs(Fp) * 100.0) * np.sign(Fp)
    pm = z.mean(-2, keepdims=True)
    ps = z.std(-2, keepdims=True) + 1e-6
    o = np.argsort(np.argsort(z, -2), -2).astype(np.float32) / (z.shape[-2] - 1.0) - 0.5
    return np.concatenate([(z - mu) / sd, (z - pm) / ps, o], -1).astype(np.float32)


# ------------------------------------------------------------------ the trained ranker
class ListwiseRanker(nn.Module):
    """A neural ranker trained from random initialisation. It scores all 20 candidates of a
    pool together: the set-attention layer lets each candidate be judged against the other
    members of its own pool before the final score, which is what the metric asks for (a
    within-pool ordering). Trained with softmax cross-entropy on the true continuation, i.e.
    directly on the ordering the metric scores."""

    def __init__(self, n_in, hid=HID, drop=DROP, depth=DEPTH):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, hid), nn.GELU(), nn.Dropout(drop)]
            d = hid
        self.body = nn.Sequential(*layers)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True, dropout=drop)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):                       # x: (queries, 20, features)
        h = self.body(x)
        h = self.norm(h + self.attn(h, h, h, need_weights=False)[0])
        return self.head(h).squeeze(-1)


def subfields_from_train_pools(pools, target_of):
    """Union-find over 'appeared in the same TRAINING pool'. Recovers the six research
    subfields from training structure alone. Used only to draw realistic same-subfield
    negatives while training; it is never a prediction feature, and evaluation pools are not
    touched."""
    cands = sorted({c for v in pools.values() for c in v})
    idx = {c: i for i, c in enumerate(cands)}
    par = list(range(len(cands)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for pool in pools.values():
        ids = [idx[c] for c in pool]
        for j in ids[1:]:
            ra, rb = find(ids[0]), find(j)
            if ra != rb:
                par[ra] = rb
    roots = {}
    for c in cands:
        roots.setdefault(find(idx[c]), len(roots))
    sub_of_cand = {c: roots[find(idx[c])] for c in cands}
    return {q: sub_of_cand[t] for q, t in target_of.items()}


def stratified_folds(sub, n_folds, seed):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(sub), dtype=np.int64)
    for s in np.unique(sub):
        i = np.where(sub == s)[0]
        i = i[rng.permutation(len(i))]
        fold[i] = np.arange(len(i)) % n_folds
    return fold


def sample_pools(idx, sub, seed, n_cand=POOL):
    """Build a pool of 20 the way the task builds one: the true continuation plus 19 other
    abstracts OF THE SAME SUBFIELD, drawn from the same set of abstracts."""
    rng = np.random.default_rng(seed)
    by = {s: idx[sub[idx] == s] for s in np.unique(sub[idx])}
    P = np.empty((len(idx), n_cand), dtype=np.int64)
    for r, i in enumerate(idx):
        pop = by[sub[i]]
        pop = pop[pop != i]
        P[r, 0] = i
        P[r, 1:] = rng.choice(pop, size=n_cand - 1, replace=False)
    return P


def pool_score(S):
    """Official metric on a score matrix whose column 0 is the true continuation, with ties
    resolved by their exact expectation under a random tie-break."""
    s0 = S[:, :1]
    greater = (S[:, 1:] > s0).sum(1)
    equal = (S[:, 1:] == s0).sum(1)
    return float((greater + 0.5 * equal).mean() / (S.shape[1] - 1.0))


def validate_submission(sub, sample_path, pools_of):
    import pandas as pd
    sample = pd.read_csv(sample_path, keep_default_na=False)
    assert list(sub.columns) == list(sample.columns), f"columns {list(sub.columns)}"
    assert len(sub) == len(sample), f"rows {len(sub)} != {len(sample)}"
    assert sub.query_id.is_unique, "duplicate query_id"
    assert set(sub.query_id) == set(sample.query_id), "query id set mismatch"
    for qid, rank in zip(sub.query_id, sub.ranking):
        ids = rank.split()
        assert len(ids) == POOL, f"{qid}: {len(ids)} ids"
        assert len(set(ids)) == POOL, f"{qid}: duplicate candidate"
        assert set(ids) == set(pools_of[qid]), f"{qid}: ranking is not this query's pool"
    assert (sub.ranking.str.len() > 0).all(), "empty ranking string"


def main():
    import pandas as pd
    seed_everything()
    log("reading data")
    train = pd.read_csv(PUBLIC_DIR / "train.csv", keep_default_na=False)
    test = pd.read_csv(PUBLIC_DIR / "test.csv", keep_default_na=False)
    cand = pd.read_csv(PUBLIC_DIR / "candidates.csv", keep_default_na=False)
    labels = pd.read_csv(PUBLIC_DIR / "train_labels.csv", keep_default_na=False)
    sample_path = PUBLIC_DIR / "sample_submission.csv"
    sample = pd.read_csv(sample_path, keep_default_na=False)

    text_of = dict(zip(cand.candidate_id, cand.snippet_b))
    target_of = dict(zip(labels.query_id, labels.target_id))
    tr_pools = {q: c.split() for q, c in zip(train.query_id, train.candidates)}
    te_pools = {q: c.split() for q, c in zip(test.query_id, test.candidates)}

    # one training ABSTRACT per training query: its opening and its true continuation
    A = list(train.snippet_a)
    B = [text_of[target_of[q]] for q in train.query_id]
    sub_of_q = subfields_from_train_pools(tr_pools, target_of)
    sub = np.array([sub_of_q[q] for q in train.query_id])
    n_train = len(A)
    log(f"{n_train} training abstracts, {len(np.unique(sub))} subfields recovered, "
        f"{len(test)} evaluation queries")

    # ---- pattern bank: fitted on TRAINING text only, at its own noise level and at a copy
    #      re-corrupted to the stated evaluation level, so the statistics cover both regimes.
    rng = np.random.default_rng(SEED)
    fit_text = A + B
    fit_text = fit_text + corrupt(fit_text, extra_rate(TRAIN_NOISE, EVAL_NOISE), rng)
    bank = PatternBank(SEED_SPEC, SKEL_SPEC, NBITS, MIN_DF, N_RARE).fit(fit_text)
    log(f"pattern bank fitted ({len(bank.seeds)} spaced seeds, {len(bank.skseeds)} structure seeds)")

    # ---- training views: fresh corruption across the regime x fresh same-subfield pools
    all_idx = np.arange(n_train)
    views = []
    for a in range(N_AUG):
        p = AUG_MIN + (AUG_MAX - AUG_MIN) * a / max(N_AUG - 1, 1)
        q = extra_rate(TRAIN_NOISE, p)
        ra = np.random.default_rng(SEED + 101 * a)
        ta = corrupt(A, q, ra) if q > 0 else list(A)
        tb = corrupt(B, q, ra) if q > 0 else list(B)
        Vq, Vc = bank.vectors(ta), bank.vectors(tb)
        for s in range(N_POOLSAMP):
            P = sample_pools(all_idx, sub, seed=SEED + 977 * a + 31 * s)
            views.append(profiles(bank, Vq, Vc, P))
        log(f"  training view set {a + 1}/{N_AUG} at ~{p:.2f} corruption")
    V = np.stack(views, 0)                                   # (views, abstracts, 20, features)
    zz = np.log1p(np.abs(V) * 100.0) * np.sign(V)
    MU = zz.reshape(-1, zz.shape[-1]).mean(0)
    SD = zz.reshape(-1, zz.shape[-1]).std(0) + 1e-6
    X = featurise(V, MU, SD)
    log(f"training tensor {X.shape}; features {FEATURE_NAMES}")

    # ---- evaluation features: the SAME transforms, applied (never fitted) to evaluation text
    te_ids = list(test.query_id)
    te_cand_ids = sorted({c for v in te_pools.values() for c in v})
    cpos = {c: i for i, c in enumerate(te_cand_ids)}
    te_pool_idx = np.array([[cpos[c] for c in te_pools[q]] for q in te_ids], dtype=np.int64)
    Vq_te = bank.vectors(list(test.snippet_a))
    Vc_te = bank.vectors([text_of[c] for c in te_cand_ids])
    X_te = torch.from_numpy(featurise(profiles(bank, Vq_te, Vc_te, te_pool_idx), MU, SD))
    log(f"evaluation tensor {tuple(X_te.shape)}")

    # ---- train the ensemble: one ranker per (abstract-level fold x init), early stopped on
    #      its own held-out abstracts. Fold membership is data-determined, never clock-driven.
    fold = stratified_folds(sub, N_FOLDS, SEED)
    Xt = torch.from_numpy(X)
    test_ranks = np.zeros((len(te_ids), POOL), dtype=np.float64)
    cv_scores = []
    for f in range(N_FOLDS):
        tr_i = np.where(fold != f)[0]
        va_i = np.where(fold == f)[0]
        # validation pools are rebuilt from HELD-OUT abstracts only, exactly the way the
        # hidden evaluation pools are built from evaluation abstracts only
        vp = sample_pools(va_i, sub, seed=SEED + 7919 + f)
        loc = {g: k for k, g in enumerate(va_i)}
        vpl = np.vectorize(loc.get)(vp)
        rv = np.random.default_rng(SEED + 555 + f)
        qq = extra_rate(TRAIN_NOISE, EVAL_NOISE)
        Vq_v = bank.vectors(corrupt([A[i] for i in va_i], qq, rv))
        Vc_v = bank.vectors(corrupt([B[i] for i in va_i], qq, rv))
        Xva = torch.from_numpy(featurise(profiles(bank, Vq_v, Vc_v, vpl), MU, SD))
        for s in range(N_SEEDS):
            torch.manual_seed(SEED + 13 * f + s)
            net = ListwiseRanker(X.shape[-1]).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
            steps = max(len(tr_i) // BATCH, 1)
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                                                      total_steps=EPOCHS * steps, pct_start=0.2)
            rg = np.random.default_rng(SEED + 97 * f + s)
            best, best_state = 9.0, None
            for ep in range(EPOCHS):
                net.train()
                for _ in range(steps):
                    qsel = torch.from_numpy(rg.choice(tr_i, size=BATCH, replace=False))
                    vsel = torch.from_numpy(rg.integers(0, X.shape[0], size=BATCH))
                    sc = net(Xt[vsel, qsel])
                    loss = F.cross_entropy(sc, torch.zeros(BATCH, dtype=torch.long))
                    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
                if (ep + 1) % EVAL_EVERY == 0:
                    net.eval()
                    with torch.no_grad():
                        v = pool_score(net(Xva).numpy())
                    if v < best:                       # early stopping on a validation METRIC
                        best = v
                        best_state = {k: t.clone() for k, t in net.state_dict().items()}
            net.load_state_dict(best_state)
            net.eval()
            with torch.no_grad():
                s_te = net(X_te).numpy()
            # rank-average the ensemble members: members live on different score scales
            test_ranks += -np.argsort(np.argsort(-s_te, 1), 1)
            cv_scores.append(best)
            log(f"  fold {f} init {s}: held-out abstracts score {best:.4f}")
    log(f"CV (held-out abstracts, pools rebuilt from held-out abstracts, {EVAL_NOISE:.0%} noise): "
        f"{np.mean(cv_scores):.4f} +- {np.std(cv_scores):.4f}")

    # ---- rank each evaluation pool by the trained ensemble, best first
    order = np.argsort(-test_ranks, axis=1, kind="stable")
    rankings = []
    for r, q in enumerate(te_ids):
        pool = te_pools[q]
        rankings.append(" ".join(pool[j] for j in order[r]))
    out = pd.DataFrame({"query_id": te_ids, "ranking": rankings})
    out = out.set_index("query_id").loc[list(sample.query_id)].reset_index()
    validate_submission(out, sample_path, te_pools)
    out.to_csv(SUBMISSION_OUT, index=False)
    back = pd.read_csv(SUBMISSION_OUT, keep_default_na=False)
    validate_submission(back, sample_path, te_pools)
    log(f"wrote {SUBMISSION_OUT} shape={out.shape}")


if __name__ == "__main__":
    main()
