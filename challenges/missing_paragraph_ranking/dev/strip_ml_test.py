"""Rejection class Q1 / T: remove every trained component and see what the remainder scores.
Replicates solution.py's OWN validation (same folds, same pools, same seeds, same pattern bank
fitted on all of train) and scores the raw cosine feature alone, with no ranker at all."""
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

sol_path = Path(sys.argv[1]); PUB = Path(sys.argv[2])
sys.argv = [str(sol_path), str(PUB), "/tmp/unused.csv"]
spec = importlib.util.spec_from_file_location("sol", sol_path)
S = importlib.util.module_from_spec(spec)

spec.loader.exec_module(S)

train = pd.read_csv(PUB/"train.csv", keep_default_na=False)
cand = pd.read_csv(PUB/"candidates.csv", keep_default_na=False)
lab = pd.read_csv(PUB/"train_labels.csv", keep_default_na=False)
text_of = dict(zip(cand.candidate_id, cand.snippet_b)); tgt = dict(zip(lab.query_id, lab.target_id))
tr_pools = {q: c.split() for q, c in zip(train.query_id, train.candidates)}
A = list(train.snippet_a); B = [text_of[tgt[q]] for q in train.query_id]
sub_of_q = S.subfields_from_train_pools(tr_pools, tgt)
sub = np.array([sub_of_q[q] for q in train.query_id])

rng = np.random.default_rng(S.SEED)
fit_text = A + B
fit_text = fit_text + S.corrupt(fit_text, S.extra_rate(S.TRAIN_NOISE, S.EVAL_NOISE), rng)
bank = S.PatternBank(S.SEED_SPEC, S.SKEL_SPEC, S.NBITS, S.MIN_DF, S.N_RARE).fit(fit_text)
fold = S.stratified_folds(sub, S.N_FOLDS, S.SEED)

scores = []
for f in range(S.N_FOLDS):
    va_i = np.where(fold == f)[0]
    vp = S.sample_pools(va_i, sub, seed=S.SEED + 7919 + f)
    loc = {g: k for k, g in enumerate(va_i)}
    vpl = np.vectorize(loc.get)(vp)
    rv = np.random.default_rng(S.SEED + 555 + f)
    qq = S.extra_rate(S.TRAIN_NOISE, S.EVAL_NOISE)
    Vq = bank.vectors(S.corrupt([A[i] for i in va_i], qq, rv))
    Vc = bank.vectors(S.corrupt([B[i] for i in va_i], qq, rv))
    F = S.profiles(bank, Vq, Vc, vpl)
    lab0 = np.zeros(len(va_i), dtype=np.int64)        # pools are built answer-first here
    s = S.pool_score(F[:, :, 0], lab0)                # cos_all ALONE, no trained model
    scores.append(s)
    print(f"  fold {f}: cosine-only {s:.4f}", flush=True)
print(f"\nSTRIP-THE-ML: cosine alone {np.mean(scores):.4f} +- {np.std(scores):.4f}")
print(f"shipped ranker CV          0.3076 +- 0.0109")
print(f"value added by the trained model: {np.mean(scores)-0.3076:+.4f} (positive = ranker helps)")
