"""Fast tuning pass: reuses the cached feature matrix and retrieval OOF from fastcv.py.

Covers the three questions left open: how the score splits by Act (the train Acts are far
more concentrated than the test Acts, so the query-weighted mean may understate what a
flatter test set would give), whether a second-stage reranker lifts the id term, and where
the gate threshold and plan width really sit.
"""
import sys, json, pickle, time, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'src'); import index as IX
sys.modules['__main__'].Corpus = IX.Corpus
import lightgbm as lgb
import multiprocessing as mp
from lib import norm
from feats import FEATS
from metric import f1, text_score
from pipe import make_query, build_plan
from gate import GFEATS, oracle_walk, model_walk

T0 = time.time()
def log(m): print("[%5.0fs] %s" % (time.time()-T0, m), flush=True)
D = 'dataset/public/'
NPROC = 4
EX = ['r_days','r_idf','r_pnapp','n_q','s_max_pnins']
FE = FEATS + EX
C = pickle.load(open('working/corpus.pkl','rb'))
tr = pd.read_csv(D+'train.csv', keep_default_na=False)
tt = pd.read_csv(D+'train_targets.csv', keep_default_na=False)
G = {r.item_id: (set(json.loads(r.amending_ids)), norm(r.text_at_date)) for r in tt.itertuples()}
ROWS = list(tr.itertuples()); QS = [make_query(r) for r in ROWS]
X = np.load('working/fast_X.npy'); y = np.load('working/fast_y.npy')
oof = np.load('working/fast_oof.npy'); qi = np.load('working/fast_qi.npy')
cu = np.load('working/fast_cu.npy'); fq = np.load('working/fast_fq.npy')
ACT = tr.act_citation.to_numpy()
log("loaded X=%s" % (X.shape,))

RP = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
          verbose=-1, num_threads=NPROC, deterministic=True, force_row_wise=True)
def bag(p0, Xt, yt, rd):
    o = []
    for sd in (42, 202):
        p = dict(p0); p.update(seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
        o.append(lgb.train(p, lgb.Dataset(Xt, label=yt), num_boost_round=rd))
    return o
def pr(ms, Xt): return np.mean([m.predict(Xt) for m in ms], axis=0) if len(Xt) else np.zeros(0)
def id_per_query(p, thr):
    out = np.zeros(len(QS))
    for k in range(len(QS)):
        m = np.where(qi == k)[0]
        if not len(m): continue
        pp = p[m]
        keep = [int(C.uid[cu[m][j]]) for j in range(len(pp)) if pp[j] >= thr] or \
               [int(C.uid[cu[m][int(np.argmax(pp))]])]
        out[k] = f1(set(keep), set(int(x) for x in G[ROWS[k].item_id][0]))
    return out

# ---------------- 1. second-stage reranker ----------------
SF = ['s','s_rank','s_frac_max','s_gap_up','s_gap_dn','s_sum','s_n_hi','s_n_mid','s_mean','s_std','n_q2']
GAPF = ['g_before','g_after','is_near_before','is_near_after','n_before','frac_before']
S = np.zeros((len(cu), len(SF)+len(GAPF)), dtype=np.float32)
for k in range(len(QS)):
    m = np.where(qi == k)[0]
    if not len(m): continue
    p = oof[m]; srt = np.sort(p)[::-1]; r = np.argsort(np.argsort(-p, kind='stable'))
    qd = QS[k]['qd']; dd = np.array([C.dnum[int(cu[j])] for j in m])
    bef = dd[dd <= qd]; aft = dd[dd > qd]
    gb = (qd - bef.max()) if len(bef) else 20000.0
    ga = (aft.min() - qd) if len(aft) else 20000.0
    for t, j in enumerate(m):
        S[j, 0] = p[t]; S[j, 1] = r[t]; S[j, 2] = p[t]/max(1e-9, srt[0])
        S[j, 3] = (srt[r[t]-1]-p[t]) if r[t] > 0 else 0.0
        S[j, 4] = (p[t]-srt[r[t]+1]) if r[t]+1 < len(srt) else p[t]
        S[j, 5] = p.sum(); S[j, 6] = (p > 0.5).sum(); S[j, 7] = ((p > 0.2) & (p <= 0.5)).sum()
        S[j, 8] = p.mean(); S[j, 9] = p.std(); S[j, 10] = len(m)
        S[j, 11] = gb; S[j, 12] = ga
        S[j, 13] = 1.0 if (len(bef) and dd[t] == bef.max()) else 0.0
        S[j, 14] = 1.0 if (len(aft) and dd[t] == aft.min()) else 0.0
        S[j, 15] = len(bef); S[j, 16] = len(bef)/max(1, len(m))
KEEP = ['days','p_napp','p_chg','a_self','f_in','min_dist','tgt_pos','q_found_frac','is_si','cyear']
ki = [FE.index(c) for c in KEEP]
X2 = np.hstack([S, X[:, ki]]); FE2 = SF+GAPF+KEEP
P2 = dict(RP); P2.update(num_leaves=15, min_data_in_leaf=40, lambda_l2=3.0)
o2 = np.zeros(len(y))
for f in range(5):
    m = fq[qi] != f
    o2[~m] = pr(bag(P2, X2[m], y[m], 300), X2[~m])
cands = {'stage1': oof, 'stage2': o2, 'blend': 0.5*oof+0.5*o2}
best_id = (-1, None, None)
for nm, p in cands.items():
    for thr in (0.10,0.15,0.20,0.25,0.30,0.35,0.40):
        v = id_per_query(p, thr)
        if v.mean() > best_id[0]: best_id = (v.mean(), nm, thr)
    log("%-7s id F1: %s" % (nm, " ".join("%.2f:%.4f" % (t, id_per_query(p, t).mean())
                                         for t in (0.15,0.20,0.25,0.30,0.35))))
log("best id = %.4f via %s @ %.2f" % best_id)
PID = cands[best_id[1]]; IDQ = id_per_query(PID, best_id[2])
np.save('working/best_id_scores.npy', PID)

# ---------------- 2. gate, finer grid ----------------
PLAN_THRS = (0.35, 0.20, 0.10)
def _teach(ks):
    out = []
    for k in ks:
        m = np.where(qi == k)[0]; p = oof[m]; cols = cu[m]
        o = np.argsort(-p, kind='stable')
        for thr in PLAN_THRS:
            sel = [j for j in o if p[j] >= thr] or ([o[0]] if len(o) else [])
            instr, ctxs = build_plan(C, QS[k], [int(cols[j]) for j in sel], [float(p[j]) for j in sel], thr)
            gx, gy, _, sc = oracle_walk(QS[k]['en'], instr, ctxs, G[ROWS[k].item_id][1])
            out.append((k, thr, gx, gy, sc))
    return out
def chunks(seq, n):
    seq = list(seq); return [seq[i::n] for i in range(n)]
if __name__ == '__main__':
    with mp.Pool(NPROC) as pool:
        parts = pool.map(_teach, chunks(range(len(QS)), NPROC))
    GX, GY, gfold, orc = [], [], [], collections.defaultdict(list)
    for part in parts:
        for k, thr, gx, gy, sc in part:
            GX += gx; GY += gy; gfold += [fq[k]]*len(gx); orc[thr].append(sc)
    GX = np.asarray(GX, dtype=np.float32).reshape(-1, len(GFEATS)); GY = np.asarray(GY, dtype=np.int32)
    gfold = np.asarray(gfold)
    log("gate %s keep=%.3f | oracle text %s" % (GX.shape, GY.mean(),
        " ".join("%.2f:%.4f" % (t, float(np.mean(v))) for t, v in sorted(orc.items()))))
    GP = dict(RP); GP.update(num_leaves=63, min_data_in_leaf=15, lambda_l2=2.0, learning_rate=0.04)
    for f in range(5):
        for i, mm in enumerate(bag(GP, GX[gfold != f], GY[gfold != f], 500)):
            mm.save_model('working/gt_f%d_%d.txt' % (f, i))
    log("gate trained")
    def _eval(args):
        ks, pt, gt, twopass = args
        bo = {f: [lgb.Booster(model_file='working/gt_f%d_%d.txt' % (f, i)) for i in (0,1)] for f in range(5)}
        out = []
        for k in ks:
            m = np.where(qi == k)[0]; p = oof[m]; cols = cu[m]
            o = np.argsort(-p, kind='stable')
            sel = [j for j in o if p[j] >= pt] or ([o[0]] if len(o) else [])
            instr, ctxs = build_plan(C, QS[k], [int(cols[j]) for j in sel], [float(p[j]) for j in sel], pt)
            if twopass:
                instr = instr + instr; ctxs = ctxs + ctxs
            ms = bo[fq[k]]
            def predict(row, ms=ms):
                a = np.asarray(row, dtype=np.float32).reshape(1, -1)
                return float(np.mean([mm.predict(a)[0] for mm in ms]))
            _, _, txt = model_walk(QS[k]['en'], instr, ctxs, predict, gt)
            out.append((k, text_score(ROWS[k].enacted_text, txt, G[ROWS[k].item_id][1])))
        return out
    jobs = [(c, pt, gt, tp) for pt in (0.20, 0.10) for gt in (0.10,0.20,0.30,0.40)
            for tp in (False, True) for c in chunks(range(len(QS)), NPROC)]
    with mp.Pool(NPROC) as pool:
        got = pool.map(_eval, jobs)
    acc = collections.defaultdict(dict)
    for job, part in zip(jobs, got):
        for k, v in part: acc[(job[1], job[2], job[3])][k] = v
    print("\n plan gate 2pass |  text  | total")
    best = (-1, None, None)
    for key, d in sorted(acc.items()):
        tq = np.array([d[k] for k in range(len(QS))])
        tot = 100*(0.4*IDQ.mean() + 0.6*tq.mean())
        print("  %.2f %.2f %-5s | %.4f | %.2f" % (key[0], key[1], key[2], tq.mean(), tot))
        if tot > best[0]: best = (tot, key, tq)
    log("BEST total %.2f at plan=%.2f gate=%.2f twopass=%s (id %.4f text %.4f)" %
        (best[0], best[1][0], best[1][1], best[1][2], IDQ.mean(), best[2].mean()))
    # ---------------- 3. per-Act breakdown ----------------
    TQ = best[2]
    tot_q = 0.4*IDQ + 0.6*TQ
    print("\n  Act          n   idF1   text  total")
    per = []
    for a in sorted(set(ACT)):
        m = ACT == a
        per.append((a, m.sum(), IDQ[m].mean(), TQ[m].mean(), 100*tot_q[m].mean()))
    for a, n, i_, t_, v in sorted(per, key=lambda x: -x[1]):
        print("  %-11s %4d  %.3f  %.3f  %5.1f" % (a, n, i_, t_, v))
    w = 100*tot_q.mean(); u = float(np.mean([p[4] for p in per]))
    log("query-weighted total %.2f  |  Act-unweighted mean %.2f  (test Acts are far flatter)" % (w, u))
    pickle.dump(dict(IDQ=IDQ, TQ=TQ, best=best[1], per=per), open('working/tune.pkl','wb'))
