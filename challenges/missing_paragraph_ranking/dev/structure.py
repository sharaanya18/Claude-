"""Recover the latent subfield blocks from the TRAIN pool co-occurrence graph (train-only),
and characterise how pools are built. Needed to build honest validation pools."""
import sys, collections
from pathlib import Path
import numpy as np, pandas as pd

P = Path(sys.argv[1])
tr = pd.read_csv(P/"train.csv", keep_default_na=False)
te = pd.read_csv(P/"test.csv", keep_default_na=False)
lab = pd.read_csv(P/"train_labels.csv", keep_default_na=False)
tgt = dict(zip(lab.query_id, lab.target_id))
tr_pools = {q: c.split() for q, c in zip(tr.query_id, tr.candidates)}
te_pools = {q: c.split() for q, c in zip(te.query_id, te.candidates)}

# Each training abstract = (query_id, its target candidate). Map candidate -> abstract index.
cands = sorted({c for v in tr_pools.values() for c in v})
cidx = {c: i for i, c in enumerate(cands)}
N = len(cands)

# union-find over "appeared in the same pool"
par = list(range(N))
def find(a):
    while par[a] != a:
        par[a] = par[par[a]]; a = par[a]
    return a
def uni(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: par[ra] = rb
for q, pool in tr_pools.items():
    ids = [cidx[c] for c in pool]
    for j in ids[1:]:
        uni(ids[0], j)
comp = collections.Counter(find(i) for i in range(N))
print("== TRAIN pool co-occurrence components ==")
print("n components:", len(comp), " sizes:", sorted(comp.values(), reverse=True))

lbl = {r: k for k, r in enumerate(sorted(comp, key=lambda r: -comp[r]))}
sub_of_cand = {c: lbl[find(cidx[c])] for c in cands}
# query's own subfield = subfield of its target
sub_of_query = {q: sub_of_cand[tgt[q]] for q in tr_pools}
print("queries per subfield:", collections.Counter(sub_of_query.values()))
# sanity: is every pool entirely within one subfield?
ok = all(len({sub_of_cand[c] for c in pool}) == 1 for pool in tr_pools.values())
print("every train pool is single-subfield:", ok)
print("query subfield == its pool subfield:",
      all(sub_of_query[q] == sub_of_cand[pool[0]] for q, pool in tr_pools.items()))

# TEST: same check (structure only, not used as a feature)
tecands = sorted({c for v in te_pools.values() for c in v})
tci = {c: i for i, c in enumerate(tecands)}
par = list(range(len(tecands)))
for q, pool in te_pools.items():
    ids = [tci[c] for c in pool]
    for j in ids[1:]: uni(ids[0], j)
tecomp = collections.Counter(find(i) for i in range(len(tecands)))
print("\n== TEST pool components (structure only) ==")
print("n components:", len(tecomp), " sizes:", sorted(tecomp.values(), reverse=True))

# per-subfield candidate counts and pool coverage
print("\n== POOL DESIGN ==")
for s in sorted(set(sub_of_cand.values())):
    cs = [c for c in cands if sub_of_cand[c] == s]
    qs = [q for q in tr_pools if sub_of_query[q] == s]
    print(f"subfield {s}: {len(cs)} candidates, {len(qs)} queries, pool slots {len(qs)*20}, "
          f"appearances/cand {len(qs)*20/len(cs):.2f}")

np.save(Path(sys.argv[2])/"sub_of_cand.npy", np.array([sub_of_cand[c] for c in cands]))
pd.DataFrame({"candidate_id": cands, "subfield": [sub_of_cand[c] for c in cands]}).to_csv(
    Path(sys.argv[2])/"train_subfields.csv", index=False)
print("\nsaved train_subfields.csv")
