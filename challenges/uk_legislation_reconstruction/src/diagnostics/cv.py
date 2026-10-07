"""Full Act-grouped CV: retrieval reranker + edit gate, scored with the exact metric."""
import sys, json, pickle, time, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'src'); import index as IX
sys.modules['__main__'].Corpus = IX.Corpus
import lightgbm as lgb
from lib import norm
from feats import FeatureBuilder, FEATS
from metric import f1, text_score, mtok, edits
from pipe import make_query, retrieval_matrix, build_plan, select_ids, apply_gated, dnum
from gate import plan_rows, GFEATS

T0 = time.time()
def log(m): print("[%5.0fs] %s" % (time.time()-T0, m), flush=True)

D = 'dataset/public/'
C = pickle.load(open('working/corpus.pkl', 'rb'))
FB = FeatureBuilder(C)
tr = pd.read_csv(D+'train.csv', keep_default_na=False)
tt = pd.read_csv(D+'train_targets.csv', keep_default_na=False)
G = {r.item_id: (set(json.loads(r.amending_ids)), norm(r.text_at_date)) for r in tt.itertuples()}
rows = list(tr.itertuples())
queries = [make_query(r) for r in rows]
log("queries %d" % len(queries))

X, qi, cu = retrieval_matrix(C, FB, queries)
y = np.array([1 if C.uid[cu[k]] in G[rows[qi[k]].item_id][0] else 0 for k in range(len(cu))])
log("retrieval matrix %s pos=%d" % (X.shape, y.sum()))

acts = tr.act_citation.to_numpy()
cnt = collections.Counter(acts); fold_of = {}; load = [0]*5
for a, n in sorted(cnt.items(), key=lambda x: -x[1]):
    j = int(np.argmin(load)); fold_of[a] = j; load[j] += n
fq = np.array([fold_of[a] for a in acts])

RP = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
          verbose=-1, seed=42, num_threads=4, deterministic=True, force_row_wise=True)
oof = np.zeros(len(y))
for f in range(5):
    m = fq[qi] != f
    oof[~m] = lgb.train(RP, lgb.Dataset(X[m], label=y[m]), num_boost_round=400).predict(X[~m])
log("retrieval OOF done")

THR_R = 0.25
# ---- gate training data from the SELECTED (OOF) provisions, so the gate sees inference-time plans
GX, GY, gfold, gq = [], [], [], []
plans = []
for k, r in enumerate(rows):
    m = qi == k
    p = oof[m]; uu = C.uid[cu[m]]; rr = cu[m]
    keep = np.argsort(-p)
    sel = [j for j in keep if p[j] >= THR_R] or ([keep[0]] if len(keep) else [])
    prows = [int(rr[j]) for j in sel]; pscore = [float(p[j]) for j in sel]
    instr, ctxs = build_plan(C, queries[k], prows, pscore)
    gx, gy, _ = plan_rows(queries[k]['en'], instr, ctxs, G[r.item_id][1])
    plans.append((instr, ctxs, [int(uu[j]) for j in sel]))
    GX += gx; GY += gy; gfold += [fq[k]]*len(gx); gq += [k]*len(gx)
GX = np.asarray(GX, dtype=np.float32).reshape(-1, len(GFEATS)); GY = np.asarray(GY, dtype=np.float32)
gfold = np.asarray(gfold); gq = np.asarray(gq)
log("gate matrix %s  gain>0 %.3f  gain<0 %.3f" % (GX.shape, (GY > 1e-9).mean(), (GY < -1e-9).mean()))

GP = dict(objective='regression', learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
          verbose=-1, seed=42, num_threads=4, deterministic=True, force_row_wise=True)
goof = np.zeros(len(GY))
for f in range(5):
    m = gfold != f
    if m.sum() and (~m).sum():
        goof[~m] = lgb.train(GP, lgb.Dataset(GX[m], label=GY[m]), num_boost_round=300).predict(GX[~m])
log("gate OOF done")

def score_all(thr_g):
    ids_f1, txt_f1 = [], []
    for k, r in enumerate(rows):
        instr, ctxs, kept = plans[k]
        gp = goof[gq == k]
        out = apply_gated(queries[k]['en'], instr, ctxs, gp, thr_g)
        tru_i, tru_t = G[r.item_id]
        ids_f1.append(f1(set(kept), set(int(x) for x in tru_i)))
        txt_f1.append(text_score(r.enacted_text, out, tru_t))
    a, b = float(np.mean(ids_f1)), float(np.mean(txt_f1))
    return 100*(0.40*a + 0.60*b), 100*a, 100*b

print("\n thr_gate |  total |  idF1 | textF1")
best = (-1, None)
for tg in [-9, -0.02, -0.01, -0.005, -0.002, 0.0, 0.002, 0.005, 0.01, 0.02, 0.04]:
    s, a, b = score_all(tg)
    print("  %7.3f | %6.2f | %5.2f | %6.2f" % (tg, s, a, b))
    if s > best[0]: best = (s, tg)
log("BEST total %.2f at gate thr %s" % best)
np.save('working/cv_oof.npy', oof); np.save('working/cv_goof.npy', goof)
pickle.dump(dict(fold_of=fold_of, fq=fq, qi=qi, cu=cu, gq=gq, GY=GY), open('working/cvstate.pkl','wb'))
imp = sorted(zip(GFEATS, lgb.train(GP, lgb.Dataset(GX, label=GY), num_boost_round=300).feature_importance('gain')), key=lambda x: -x[1])
print("gate top feats:", [(a, int(b)) for a, b in imp[:14]])
