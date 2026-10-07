"""Measure the headroom of LOW-RANK TOPIC matching before committing to an architecture.
(a) unsupervised LSA cosine, (b) supervised low-rank cross-covariance (PLS/CCA) learned from
the training pairs. All fitting on the training fold only; evaluation at the 30% eval noise."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

D = Data(sys.argv[1])
fold = make_folds(D.subv, 5, seed=0)
A, B = np.array(D.A, dtype=object), np.array(D.B, dtype=object)
f = 0
tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
pools = build_pools(vai, D.subv, seed=100 + f)
pos = {g: k for k, g in enumerate(vai)}
pl = np.array([[pos[j] for j in row] for row in pools])
R30 = extra_rate(0.20, 0.30)

def sc(Ua, Vb):
    Ua, Vb = normalize(Ua), normalize(Vb)
    return score_from_matrix((Ua @ Vb.T)[np.arange(len(vai))[:, None], pl])

for ng, an, mindf in [((2, 5), "char_wb", 3), ((2, 4), "char", 3), ((2, 3), "char", 2)]:
    rs = np.random.default_rng(5)
    fit = list(A[tri]) + list(B[tri])
    fit = fit + corrupt(fit, R30, rs)                      # fit at both noise levels, train only
    vec = TfidfVectorizer(analyzer=an, ngram_range=ng, min_df=mindf, sublinear_tf=True,
                          dtype=np.float32).fit(fit)
    Xa_tr = normalize(vec.transform(list(A[tri]))); Xb_tr = normalize(vec.transform(list(B[tri])))
    r2 = np.random.default_rng(11)
    va = corrupt(list(A[vai]), R30, r2); vb = corrupt(list(B[vai]), R30, r2)
    Xa_va = normalize(vec.transform(va)); Xb_va = normalize(vec.transform(vb))
    print(f"\n### {an} {ng} min_df={mindf}  V={len(vec.vocabulary_)}")
    print(f"  raw tf-idf cosine                       -> {sc(Xa_va.toarray(), Xb_va.toarray()):.4f}")
    import scipy.sparse as sp
    Z = sp.vstack([Xa_tr, Xb_tr])
    for k in (128, 256, 512):
        t0 = time.time()
        svd = TruncatedSVD(n_components=k, random_state=0, algorithm="randomized", n_iter=5).fit(Z)
        Pa_tr, Pb_tr = svd.transform(Xa_tr), svd.transform(Xb_tr)
        Pa_va, Pb_va = svd.transform(Xa_va), svd.transform(Xb_va)
        s_lsa = sc(Pa_va, Pb_va)
        # supervised low-rank cross-covariance learned from the TRAIN pairs (PLS)
        na = normalize(Pa_tr); nb = normalize(Pb_tr)
        for lam in (1e-2, 1e-1):
            Ca = na.T @ na / len(na) + lam * np.eye(k); Cb = nb.T @ nb / len(nb) + lam * np.eye(k)
            Cab = na.T @ nb / len(na)
            Wa = np.linalg.inv(np.linalg.cholesky(Ca)); Wb = np.linalg.inv(np.linalg.cholesky(Cb))
            M = Wa @ Cab @ Wb.T
            U, S, Vt = np.linalg.svd(M)
            for r in (16, 32, 64):
                La = (Wa.T @ U[:, :r]) * np.sqrt(S[:r]); Lb = (Wb.T @ Vt[:r].T) * np.sqrt(S[:r])
                s_cca = sc(normalize(Pa_va) @ La, normalize(Pb_va) @ Lb)
                print(f"  svd{k:4d} lsa={s_lsa:.4f} | CCA lam={lam:g} rank={r:3d} -> {s_cca:.4f}")
        print(f"     ({time.time()-t0:.0f}s)")
