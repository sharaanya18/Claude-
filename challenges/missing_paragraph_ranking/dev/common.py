"""Shared dev harness: exact metric, corruption model, abstract-level folds with
test-like pools. Nothing here is fitted on test text."""
import collections, random
from pathlib import Path
import numpy as np, pandas as pd

ALPHA = np.array(list("abcdefghijklmnopqrstuvwxyz"))

# ---------------------------------------------------------------- exact metric
def rank_error(ranking, target):
    """Official per-query error: (r-1)/19 with r the 1-based position of the target.
    Repeated ids count once; a missing target scores 1."""
    seen, r = set(), 0
    for cid in ranking:
        if cid in seen:
            continue
        seen.add(cid); r += 1
        if cid == target:
            return (r - 1) / 19.0
    return 1.0

def mean_rank_error(rankings, targets):
    """rankings: dict qid -> list of ids. targets: dict qid -> id. Missing query = 1."""
    return float(np.mean([rank_error(rankings.get(q, []), t) for q, t in targets.items()]))

def rank_metrics(rankings, targets):
    errs, ranks = [], []
    for q, t in targets.items():
        e = rank_error(rankings.get(q, []), t)
        errs.append(e); ranks.append(e * 19 + 1)
    errs, ranks = np.array(errs), np.array(ranks)
    return dict(score=errs.mean(), top1=(ranks == 1).mean(), top3=(ranks <= 3).mean(),
                top5=(ranks <= 5).mean(), mrr=(1.0 / ranks).mean(), mean_rank=ranks.mean())

def score_from_matrix(S, n=20):
    """S: (Q, n) scores, column 0 is the target. Returns the official metric.
    Ties are resolved by their EXACT expectation under a random tie-break, which is what a
    real pool does (the target sits at a random position), never by column order."""
    s0 = S[:, :1]
    greater = (S[:, 1:] > s0).sum(1)
    equal = (S[:, 1:] == s0).sum(1)
    return float((greater + 0.5 * equal).mean() / (n - 1.0))

def metrics_from_matrix(S, n=20, seed=0):
    s0 = S[:, :1]
    greater = (S[:, 1:] > s0).sum(1)
    equal = (S[:, 1:] == s0).sum(1)
    score = float((greater + 0.5 * equal).mean() / (n - 1.0))
    rng = np.random.default_rng(seed)                 # random tie-break for the top-k views
    jit = rng.random(S.shape) * 1e-12
    pos = (((S + jit)[:, 1:] > (S + jit)[:, :1]).sum(1) + 1).astype(float)
    return dict(score=score, top1=float((pos == 1).mean()), top3=float((pos <= 3).mean()),
                top5=float((pos <= 5).mean()), mrr=float((1.0 / pos).mean()),
                mean_rank=float(pos.mean()))

# ------------------------------------------------------------- corruption model
def corrupt(texts, rate, rng):
    """The challenge's documented noise: each English letter is replaced, with probability
    `rate`, by a uniform random letter of the same case. Spaces/digits/punctuation untouched.
    Used only to raise TRAINING/VALIDATION text toward the stated evaluation noise level."""
    out = []
    for t in texts:
        arr = np.frombuffer(t.encode("ascii"), dtype=np.uint8).copy()
        low = (arr >= 97) & (arr <= 122)
        up = (arr >= 65) & (arr <= 90)
        m = (rng.random(arr.shape[0]) < rate)
        sl, su = low & m, up & m
        if sl.any(): arr[sl] = 97 + rng.integers(0, 26, sl.sum())
        if su.any(): arr[su] = 65 + rng.integers(0, 26, su.sum())
        out.append(arr.tobytes().decode("ascii"))
    return out

def extra_rate(p_from, p_to):
    """Extra corruption that takes text already corrupted at p_from to p_to."""
    return 1.0 - (1.0 - p_to) / (1.0 - p_from)

# --------------------------------------------------------------------- loading
class Data:
    def __init__(self, public_dir):
        P = Path(public_dir)
        self.train = pd.read_csv(P/"train.csv", keep_default_na=False)
        self.test = pd.read_csv(P/"test.csv", keep_default_na=False)
        self.cand = pd.read_csv(P/"candidates.csv", keep_default_na=False)
        self.labels = pd.read_csv(P/"train_labels.csv", keep_default_na=False)
        self.sample = pd.read_csv(P/"sample_submission.csv", keep_default_na=False)
        self.cmap = dict(zip(self.cand.candidate_id, self.cand.snippet_b))
        self.tgt = dict(zip(self.labels.query_id, self.labels.target_id))
        self.tr_pools = {q: c.split() for q, c in zip(self.train.query_id, self.train.candidates)}
        self.te_pools = {q: c.split() for q, c in zip(self.test.query_id, self.test.candidates)}
        # one training ABSTRACT per training query: opening a_i and contribution b_i
        self.qids = list(self.train.query_id)
        self.A = list(self.train.snippet_a)
        self.B = [self.cmap[self.tgt[q]] for q in self.qids]
        self.sub = subfields_from_pools(self.tr_pools, self.tgt)
        self.subv = np.array([self.sub[q] for q in self.qids])

def subfields_from_pools(pools, tgt):
    """Union-find over 'appeared in the same TRAIN pool'. Recovers the six research
    subfields from training structure only; used to build same-subfield negatives."""
    cands = sorted({c for v in pools.values() for c in v})
    idx = {c: i for i, c in enumerate(cands)}
    par = list(range(len(cands)))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for pool in pools.values():
        ids = [idx[c] for c in pool]
        for j in ids[1:]:
            ra, rb = find(ids[0]), find(j)
            if ra != rb: par[ra] = rb
    roots = sorted(collections.Counter(find(i) for i in range(len(cands))).items(),
                   key=lambda kv: -kv[1])
    lbl = {r: k for k, (r, _) in enumerate(roots)}
    sub_of_cand = {c: lbl[find(idx[c])] for c in cands}
    return {q: sub_of_cand[t] for q, t in tgt.items()}

# ------------------------------------------------------------------ CV folds
def make_folds(subv, n_folds=5, seed=0):
    """Abstract-level folds, stratified by subfield so every fold has enough same-subfield
    candidates to build pools of 20."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(subv), dtype=int)
    for s in np.unique(subv):
        idx = np.where(subv == s)[0]
        idx = idx[rng.permutation(len(idx))]
        fold[idx] = np.arange(len(idx)) % n_folds
    return fold

def build_pools(val_idx, subv, n_cand=20, seed=0):
    """Mirror the hidden split: evaluation pools contain ONLY held-out abstracts of the
    query's own subfield. Returns (Q, n_cand) index array whose column 0 is the target."""
    rng = np.random.default_rng(seed)
    bysub = {s: val_idx[subv[val_idx] == s] for s in np.unique(subv[val_idx])}
    pools = np.empty((len(val_idx), n_cand), dtype=np.int64)
    for r, i in enumerate(val_idx):
        pop = bysub[subv[i]]
        pop = pop[pop != i]
        pools[r, 0] = i
        pools[r, 1:] = rng.choice(pop, size=n_cand - 1, replace=False)
    return pools
