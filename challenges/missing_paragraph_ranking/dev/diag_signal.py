"""DIAGNOSTIC: where does the query->continuation signal actually live? Per-signal AUC of
true pair vs same-subfield decoy, at the evaluation noise level. Train fold only."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

D = Data(sys.argv[1])
fold = make_folds(D.subv, 5, seed=0)
A, B = np.array(D.A, dtype=object), np.array(D.B, dtype=object)
vai = np.where(fold == 0)[0]
pools = build_pools(vai, D.subv, seed=100)
pos = {g: k for k, g in enumerate(vai)}
pl = np.array([[pos[j] for j in row] for row in pools])
rng = np.random.default_rng(7)
R = extra_rate(0.20, 0.30)
qa = corrupt(list(A[vai]), R, rng); cb = corrupt(list(B[vai]), R, rng)

def auc_from_S(S):
    """AUC of column 0 (true) against the 19 decoys, ties at 0.5."""
    s0 = S[:, :1]
    return float(((S[:, 1:] < s0).sum() + 0.5 * (S[:, 1:] == s0).sum()) / (S.shape[0] * 19))

TMAX, LMAX = 32, 20
def tokmat(texts):
    M = np.zeros((len(texts), TMAX, LMAX), dtype=np.uint8); L = np.zeros((len(texts), TMAX), dtype=np.int16)
    for n, t in enumerate(texts):
        for k, w in enumerate(t.split()[:TMAX]):
            w = w[:LMAX]; M[n, k, :len(w)] = np.frombuffer(w.encode(), dtype=np.uint8); L[n, k] = len(w)
    return M, L
Mq, Lq = tokmat(qa); Mc, Lc = tokmat(cb)
alq = (Mq >= 97) & (Mq <= 122)

print("== how often do true pairs share a strongly-agreeing long token? ==")
for minlen, thr in [(6, 0.6), (7, 0.6), (8, 0.6), (6, 0.75), (8, 0.75), (10, 0.6)]:
    S = np.zeros((len(vai), 20))
    for r in range(len(vai)):
        idx = pl[r]; mq, lq = Mq[r], Lq[r]; mc, lc = Mc[idx], Lc[idx]
        eq = (mq[None, :, None, :] == mc[:, None, :, :]) & alq[r][None, :, None, :]
        match = eq.sum(-1).astype(np.float32)
        same = (lq[None, :, None] == lc[:, None, :]) & (lq[None, :, None] >= minlen)
        agree = np.where(same, match / np.maximum(lq[None, :, None], 1), 0.0)
        S[r] = (agree >= thr).sum((1, 2))
    tru = S[:, 0]; dec = S[:, 1:]
    print(f"  minlen>={minlen} agree>={thr}: mean true={tru.mean():.3f} decoy={dec.mean():.3f} "
          f"AUC={auc_from_S(S):.4f}  P(>=1 hit) true={ (tru>0).mean():.3f} decoy={(dec>0).mean():.3f}")

print("\n== TF-IDF variants: which n-gram order and weighting carries it? ==")
tri = np.where(fold != 0)[0]
fit_txt = list(A[tri]) + list(B[tri])
for name, cfg in [("char 2-2", dict(analyzer="char", ngram_range=(2, 2))),
                  ("char 3-3", dict(analyzer="char", ngram_range=(3, 3))),
                  ("char 4-4", dict(analyzer="char", ngram_range=(4, 4))),
                  ("char 5-5", dict(analyzer="char", ngram_range=(5, 5))),
                  ("char 2-5", dict(analyzer="char", ngram_range=(2, 5))),
                  ("char 2-5 no-idf", dict(analyzer="char", ngram_range=(2, 5), use_idf=False))]:
    cfg = dict(min_df=2, sublinear_tf=True, **cfg)
    vec = TfidfVectorizer(**cfg).fit(fit_txt)
    Xa = normalize(vec.transform(qa)); Xb = normalize(vec.transform(cb))
    M = np.asarray((Xa @ Xb.T).todense())
    S = M[np.arange(len(vai))[:, None], pl]
    print(f"  {name:16s} vocab={len(vec.vocabulary_):7d} SCORE={score_from_matrix(S):.4f} AUC={auc_from_S(S):.4f}")

print("\n== is the signal topical (many weak n-grams) or lexical (few strong repeats)? ==")
vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True).fit(fit_txt)
Xa = normalize(vec.transform(qa)); Xb = normalize(vec.transform(cb))
idf = vec.idf_
# contribution of the top-k shared n-grams to the cosine, true vs decoy
Xa_c, Xb_c = Xa.tocsr(), Xb.tocsr()
tops = []
for r in range(300):
    a = Xa_c[r]
    for k, col in enumerate([pl[r][0], pl[r][1]]):
        b = Xb_c[col]
        shared = np.intersect1d(a.indices, b.indices)
        if len(shared) == 0:
            tops.append((k, 0, 0.0, 0.0)); continue
        prod = np.array([a[0, j] * b[0, j] for j in shared])
        o = np.argsort(-prod)
        tops.append((k, len(shared), prod.sum(), prod[o[:5]].sum()))
tops = np.array(tops, dtype=float)
for k, nm in [(0, "TRUE "), (1, "decoy")]:
    s = tops[tops[:, 0] == k]
    print(f"  {nm}: shared n-grams={s[:,1].mean():6.1f}  cos={s[:,2].mean():.4f}  "
          f"top-5 share of cos={100*(s[:,3]/np.maximum(s[:,2],1e-9)).mean():.1f}%")
