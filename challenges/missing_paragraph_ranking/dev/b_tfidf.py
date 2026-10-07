"""Baseline 1-5: character / word TF-IDF cosine. Also calibrates the harness: the challenge
publishes 0.3387 for 'char n-gram TF-IDF fitted on training text' on the real evaluation set,
so our 30%-noise validation number for the same method should land near it."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

D = Data(sys.argv[1])
NF = 5
fold = make_folds(D.subv, NF, seed=0)
A, B = np.array(D.A, dtype=object), np.array(D.B, dtype=object)
Q_EXTRA = extra_rate(0.20, 0.30)   # description: 20% in training, 30% at evaluation

def run(cfg, noise, fit_noise, folds=range(NF), pool_seed=0, noise_seeds=(1,)):
    out = []
    for f in folds:
        tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
        pools = build_pools(vai, D.subv, seed=100 + f + 1000 * pool_seed)
        per_seed = []
        for ns in noise_seeds:
            rng = np.random.default_rng(10_000 * ns + f)
            fit_txt = list(A[tri]) + list(B[tri])
            if fit_noise > 0:
                fit_txt = corrupt(fit_txt, fit_noise, np.random.default_rng(777 + f))
            vec = TfidfVectorizer(**cfg).fit(fit_txt)
            va, vb = list(A[vai]), list(B[vai])
            if noise > 0:
                va = corrupt(va, noise, rng); vb = corrupt(vb, noise, rng)
            pos = {g: k for k, g in enumerate(vai)}
            Xa = normalize(vec.transform(va)); Xb = normalize(vec.transform(vb))
            rows = np.array([[pos[j] for j in row] for row in pools])
            S = np.asarray((Xa @ Xb.T).todense())[np.arange(len(vai))[:, None], rows]
            per_seed.append(metrics_from_matrix(S))
        out.append({k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]})
    return {k: (float(np.mean([o[k] for o in out])), float(np.std([o[k] for o in out])))
            for k in out[0]}, out

CFGS = {
  "char_wb 2-5 (reference-style)": dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
  "char 3-5":                      dict(analyzer="char",    ngram_range=(3, 5), min_df=2, sublinear_tf=True),
  "char 2-4":                      dict(analyzer="char",    ngram_range=(2, 4), min_df=2, sublinear_tf=True),
  "char 3-4":                      dict(analyzer="char",    ngram_range=(3, 4), min_df=2, sublinear_tf=True),
  "char 4-4":                      dict(analyzer="char",    ngram_range=(4, 4), min_df=2, sublinear_tf=True),
  "char 3-3":                      dict(analyzer="char",    ngram_range=(3, 3), min_df=2, sublinear_tf=True),
  "char 2-3":                      dict(analyzer="char",    ngram_range=(2, 3), min_df=2, sublinear_tf=True),
  "word 1-1":                      dict(analyzer="word",    ngram_range=(1, 1), min_df=2, sublinear_tf=True),
}
print(f"{'config':34s} {'noise':>6s} {'fit':>5s} | {'SCORE':>7s} {'sd':>6s} {'top1':>6s} {'top5':>6s} {'mrank':>6s}")
for name, cfg in CFGS.items():
    for noise, fitn in [(0.0, 0.0), (Q_EXTRA, 0.0), (Q_EXTRA, Q_EXTRA)]:
        t0 = time.time()
        m, _ = run(cfg, noise, fitn)
        eff = 0.20 if noise == 0 else 0.30
        print(f"{name:34s} {eff:6.2f} {fitn:5.3f} | {m['score'][0]:7.4f} {m['score'][1]:6.4f} "
              f"{m['top1'][0]:6.3f} {m['top5'][0]:6.3f} {m['mean_rank'][0]:6.2f}  ({time.time()-t0:.0f}s)")
