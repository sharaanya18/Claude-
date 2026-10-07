"""Act-grouped CV for the retrieval (amending_ids) term."""
import sys, json, pickle, time, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
import index as IX
sys.modules['__main__'].Corpus = IX.Corpus
from lib import norm
from feats import FeatureBuilder, FEATS
from metric import f1
import lightgbm as lgb
import datetime

D = 'dataset/public/'
C = pickle.load(open('working/corpus.pkl', 'rb'))
if not hasattr(C, 'dnum'):
    C.dnum = np.array([datetime.date(*(int(x) for x in d[:10].split('-'))).toordinal() for d in C.date], float)
FB = FeatureBuilder(C)
print("idf built", flush=True)

tr = pd.read_csv(D+'train.csv', keep_default_na=False)
tt = pd.read_csv(D+'train_targets.csv', keep_default_na=False)
gold = {r.item_id: set(json.loads(r.amending_ids)) for r in tt.itertuples()}

def mkQ(r):
    en = norm(r.enacted_text)
    sec = norm(r.section_label).replace('s. ', '').upper()
    at = norm(r.act_title)
    return dict(sec=sec, en=en, enl=en.lower(), enset=set(en.lower().split()),
                atitle=at, atitle_n=at, atset=set(at.lower().split()),
                acit=r.act_citation, cno=r.act_citation.split('c.')[-1].strip(),
                qd=float(datetime.date(*(int(x) for x in r.date[:10].split('-'))).toordinal()),
                qlen=np.log1p(len(en)), qsecnum=float(''.join(ch for ch in sec if ch.isdigit()) or 0), ncand=0)

t0 = time.time()
X, y, qi, cu = [], [], [], []
for k, r in enumerate(tr.itertuples()):
    Q = mkQ(r)
    cand = C.candidates(Q['acit'], Q['sec'])
    Q['ncand'] = float(len(cand))
    g = gold[r.item_id]
    for i in cand:
        X.append(FB.row(Q, int(i))); y.append(1 if C.uid[i] in g else 0); qi.append(k); cu.append(int(i))
X = np.asarray(X, dtype=np.float32); y = np.asarray(y); qi = np.asarray(qi); cu = np.asarray(cu)
print("features %s pos=%d in %.0fs" % (X.shape, y.sum(), time.time()-t0), flush=True)

acts = tr.act_citation.to_numpy()
uacts = sorted(set(acts))
# 5 folds grouped by Act, balanced by query count
cnt = collections.Counter(acts); folds = {}
load = [0]*5
for a, n in sorted(cnt.items(), key=lambda x: -x[1]):
    j = int(np.argmin(load)); folds[a] = j; load[j] += n
fold_q = np.array([folds[a] for a in acts])
print("fold sizes", load)

PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=42, num_threads=4, deterministic=True, force_row_wise=True)
NROUND = 400
oof = np.zeros(len(y))
for f in range(5):
    m = fold_q[qi] != f
    d = lgb.Dataset(X[m], label=y[m])
    bst = lgb.train(PARAMS, d, num_boost_round=NROUND)
    oof[~m] = bst.predict(X[~m])
print("trained", flush=True)

def evaluate(thr, topk=None, minone=True):
    sc = []
    for k, r in enumerate(tr.itertuples()):
        m = qi == k
        p = oof[m]; u = C.uid[cu[m]]
        o = np.argsort(-p)
        keep = [u[j] for j in o if p[j] >= thr]
        if topk: keep = keep[:topk]
        if minone and not keep and len(o): keep = [u[o[0]]]
        sc.append(f1(set(int(x) for x in keep), set(int(x) for x in gold[r.item_id])))
    return float(np.mean(sc))

best = (0, None)
for thr in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6]:
    s = evaluate(thr)
    if s > best[0]: best = (s, thr)
    print("  thr %.2f -> idF1 %.4f" % (thr, s))
print("BEST idF1 %.4f at thr %.2f  => 0.40 term contributes %.2f pts" % (best[0], best[1], 40*best[0]))
imp = sorted(zip(FEATS, bst.feature_importance('gain')), key=lambda x: -x[1])
print("top feats:", [(a, int(b)) for a, b in imp[:18]])
np.save('working/oof_ret.npy', oof); np.save('working/X_ret.npy', X); np.save('working/y_ret.npy', y)
np.save('working/qi_ret.npy', qi); np.save('working/cu_ret.npy', cu); np.save('working/fold_q.npy', fold_q)
