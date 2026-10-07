"""Fast Act-grouped end-to-end CV: parallel over queries, instructions memoised.

Same numbers as a single-process run (work is split by query, every fit is seeded and every
fold assignment is a function of the data), just 4 cores instead of 1.
"""
import sys, os, json, pickle, time, collections, itertools
import numpy as np, pandas as pd
sys.path.insert(0, 'src'); import index as IX
sys.modules['__main__'].Corpus = IX.Corpus
import lightgbm as lgb
import multiprocessing as mp
from lib import norm
from feats import FeatureBuilder, FEATS
from metric import f1, text_score, mtok, edits
from pipe import make_query, build_plan
from gate import GFEATS, oracle_walk, model_walk

T0 = time.time()
def log(m): print("[%5.0fs] %s" % (time.time()-T0, m), flush=True)
D = 'dataset/public/'
NPROC = 4
EX = ['r_days','r_idf','r_pnapp','n_q','s_max_pnins']
PLAN_THRS = (0.35, 0.20, 0.10)

C = pickle.load(open('working/corpus.pkl','rb'))
FB = FeatureBuilder(C)
tr = pd.read_csv(D+'train.csv', keep_default_na=False)
tt = pd.read_csv(D+'train_targets.csv', keep_default_na=False)
G = {r.item_id: (set(json.loads(r.amending_ids)), norm(r.text_at_date)) for r in tt.itertuples()}
ROWS = list(tr.itertuples())
QS = [make_query(r) for r in ROWS]
log("loaded")

def _feat_chunk(ks):
    out = []
    for k in ks:
        Q = QS[k]
        cand = C.candidates(Q['acit'], Q['sec'])
        Q['ncand'] = float(len(cand))
        out.append((k, [FB.row(Q, int(i)) for i in cand], [int(i) for i in cand]))
    return out

def chunks(seq, n):
    seq = list(seq); return [seq[i::n] for i in range(n)]

if __name__ == '__main__':
    with mp.Pool(NPROC) as pool:
        parts = pool.map(_feat_chunk, chunks(range(len(QS)), NPROC))
    res = {k: (f, c) for part in parts for k, f, c in part}
    X, qi, cu = [], [], []
    for k in range(len(QS)):
        f, c = res[k]
        QS[k]['ncand'] = float(len(c))
        X += f; qi += [k]*len(c); cu += c
    X = np.asarray(X, dtype=np.float32).reshape(-1, len(FEATS))
    qi = np.asarray(qi); cu = np.asarray(cu)
    Z = np.zeros((len(cu), len(EX)), dtype=np.float32)
    ix = {k: FEATS.index(k) for k in ('days','idf_ov','p_napp','p_nins')}
    for k in range(len(QS)):
        m = np.where(qi == k)[0]
        if not len(m): continue
        for j, col in enumerate(('days','idf_ov','p_napp')):
            Z[m, j] = np.argsort(np.argsort(-X[m, ix[col]], kind='stable')) / max(1, len(m)-1)
        Z[m, 3] = len(m); Z[m, 4] = X[m, ix['p_nins']].max()
    X = np.hstack([X, Z])
    y = np.array([1 if C.uid[cu[k]] in G[ROWS[qi[k]].item_id][0] else 0 for k in range(len(cu))])
    log("features %s pos=%d" % (X.shape, y.sum()))
    cnt = collections.Counter(tr.act_citation); load = [0]*5; fo = {}
    for a, n in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
        j = int(np.argmin(load)); fo[a] = j; load[j] += n
    fq = np.array([fo[a] for a in tr.act_citation])
    RP = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=NPROC, deterministic=True, force_row_wise=True)
    GP = dict(RP); GP.update(num_leaves=31, min_data_in_leaf=20, lambda_l2=2.0)
    def bag(p0, Xt, yt, rd):
        o = []
        for sd in (42, 202):
            p = dict(p0); p.update(seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
            o.append(lgb.train(p, lgb.Dataset(Xt, label=yt), num_boost_round=rd))
        return o
    def pr(ms, Xt): return np.mean([m.predict(Xt) for m in ms], axis=0) if len(Xt) else np.zeros(0)
    oof = np.zeros(len(y))
    for f in range(5):
        m = fq[qi] != f
        oof[~m] = pr(bag(RP, X[m], y[m], 400), X[~m])
    np.save('working/fast_oof.npy', oof); np.save('working/fast_qi.npy', qi)
    np.save('working/fast_cu.npy', cu); np.save('working/fast_fq.npy', fq)
    np.save('working/fast_X.npy', X); np.save('working/fast_y.npy', y)
    def idf1_at(thr):
        sc = []
        for k in range(len(QS)):
            m = np.where(qi == k)[0]; p = oof[m]
            keep = [int(C.uid[cu[m][j]]) for j in range(len(p)) if p[j] >= thr] or \
                   ([int(C.uid[cu[m][int(np.argmax(p))]])] if len(p) else [])
            sc.append(f1(set(keep), set(int(x) for x in G[ROWS[k].item_id][0])))
        return float(np.mean(sc))
    IDF1 = {t: idf1_at(t) for t in (0.15,0.20,0.25,0.30,0.35)}
    log("id F1: " + " ".join("%.2f:%.4f" % (t, v) for t, v in IDF1.items()))

    def _teach(ks):
        out = []
        for k in ks:
            m = np.where(qi == k)[0]; p = oof[m]; cols = cu[m]
            o = np.argsort(-p, kind='stable')
            for thr in PLAN_THRS:
                sel = [j for j in o if p[j] >= thr] or ([o[0]] if len(o) else [])
                instr, ctxs = build_plan(C, QS[k], [int(cols[j]) for j in sel],
                                        [float(p[j]) for j in sel], thr)
                gx, gy, _, sc = oracle_walk(QS[k]['en'], instr, ctxs, G[ROWS[k].item_id][1])
                out.append((k, thr, gx, gy, sc))
        return out
    with mp.Pool(NPROC) as pool:
        parts = pool.map(_teach, chunks(range(len(QS)), NPROC))
    GX, GY, gfold, orc = [], [], [], collections.defaultdict(list)
    for part in parts:
        for k, thr, gx, gy, sc in part:
            GX += gx; GY += gy; gfold += [fq[k]]*len(gx); orc[thr].append(sc)
    GX = np.asarray(GX, dtype=np.float32).reshape(-1, len(GFEATS)); GY = np.asarray(GY, dtype=np.int32)
    gfold = np.asarray(gfold)
    log("gate matrix %s keep=%.3f | oracle text: %s" % (GX.shape, GY.mean(),
        " ".join("%.2f:%.4f" % (t, float(np.mean(v))) for t, v in sorted(orc.items()))))
    gm = {f: bag(GP, GX[gfold != f], GY[gfold != f], 300) for f in range(5)}
    for f in range(5):
        for i, mm in enumerate(gm[f]):
            mm.save_model('working/gate_f%d_%d.txt' % (f, i))
    log("gate folds trained")

    def _eval(args):
        ks, plan_thr, gth = args
        out = []
        boost = {f: [lgb.Booster(model_file='working/gate_f%d_%d.txt' % (f, i)) for i in (0, 1)]
                 for f in range(5)}
        for k in ks:
            m = np.where(qi == k)[0]; p = oof[m]; cols = cu[m]
            o = np.argsort(-p, kind='stable')
            sel = [j for j in o if p[j] >= plan_thr] or ([o[0]] if len(o) else [])
            instr, ctxs = build_plan(C, QS[k], [int(cols[j]) for j in sel],
                                     [float(p[j]) for j in sel], plan_thr)
            ms = boost[fq[k]]
            def predict(row, ms=ms):
                a = np.asarray(row, dtype=np.float32).reshape(1, -1)
                return float(np.mean([mm.predict(a)[0] for mm in ms]))
            _, _, txt = model_walk(QS[k]['en'], instr, ctxs, predict, gth)
            out.append((k, text_score(ROWS[k].enacted_text, txt, G[ROWS[k].item_id][1])))
        return out
    jobs = [(c, pt, gt) for pt in PLAN_THRS for gt in (0.30, 0.45, 0.60)
            for c in chunks(range(len(QS)), NPROC)]
    with mp.Pool(NPROC) as pool:
        got = pool.map(_eval, jobs)
    acc = collections.defaultdict(dict)
    for job, part in zip(jobs, got):
        for k, v in part: acc[(job[1], job[2])][k] = v
    print("\n  plan  gate |  text  | total @ best id thr")
    best = (-1, None)
    for (pt, gt), d in sorted(acc.items()):
        T = float(np.mean([d[k] for k in range(len(QS))]))
        for it, iv in IDF1.items():
            tot = 100*(0.4*iv + 0.6*T)
            if tot > best[0]: best = (tot, (it, pt, gt, T, iv))
        tot25 = 100*(0.4*IDF1[0.25] + 0.6*T)
        print("  %.2f  %.2f | %.4f | %.2f" % (pt, gt, T, tot25))
    log("BEST total %.2f  (id thr %.2f, plan %.2f, gate %.2f; id %.4f text %.4f)" %
        (best[0], best[1][0], best[1][1], best[1][2], best[1][4], best[1][3]))
    pickle.dump(dict(acc=dict(acc), IDF1=IDF1), open('working/fastcv.pkl','wb'))
