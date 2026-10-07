"""Second-stage reranker: per-query calibration on top of the first-stage scores.

Set-F1 needs a decision about HOW MANY provisions to claim, which a per-candidate score
cannot express.  Stage 2 therefore sees, for each candidate, the first-stage score together
with its position in that query's score distribution and the shape of that distribution, plus
the structure the task's construction implies: the query date is the midpoint between two
consecutive in-force dates, so the boundary between claimed and unclaimed candidates should
fall in a gap of the candidates' dates, not in the middle of a cluster.
"""
import sys,json,pickle,time,collections
import numpy as np,pandas as pd
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
import lightgbm as lgb
from lib import norm
from feats import FeatureBuilder,FEATS
from metric import f1
from pipe import make_query,retrieval_matrix
T0=time.time()
def log(m): print("[%5.0fs] %s"%(time.time()-T0,m),flush=True)
D='dataset/public/'
C=pickle.load(open('working/corpus.pkl','rb')); FB=FeatureBuilder(C)
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
G={r.item_id:set(json.loads(r.amending_ids)) for r in tt.itertuples()}
rows=list(tr.itertuples()); queries=[make_query(r) for r in rows]
X,qi,cu=retrieval_matrix(C,FB,queries)
EX=['r_days','r_idf','r_pnapp','n_q','s_max_pnins']
Z=np.zeros((len(cu),len(EX)),dtype=np.float32)
ix={k:FEATS.index(k) for k in ('days','idf_ov','p_napp','p_nins')}
for k in range(len(rows)):
    m=np.where(qi==k)[0]
    if not len(m): continue
    for j,col in enumerate(('days','idf_ov','p_napp')):
        Z[m,j]=np.argsort(np.argsort(-X[m,ix[col]],kind='stable'))/max(1,len(m)-1)
    Z[m,3]=len(m); Z[m,4]=X[m,ix['p_nins']].max()
# date-gap features around the query date
GAPF=['g_before','g_after','g_local','is_near_before','is_near_after','n_before','frac_before']
W=np.zeros((len(cu),len(GAPF)),dtype=np.float32)
for k in range(len(rows)):
    m=np.where(qi==k)[0]
    if not len(m): continue
    qd=queries[k]['qd']; dd=np.array([C.dnum[int(cu[j])] for j in m])
    bef=dd[dd<=qd]; aft=dd[dd>qd]
    gb=qd-bef.max() if len(bef) else 20000.0
    ga=aft.min()-qd if len(aft) else 20000.0
    nb=len(bef)
    for t,j in enumerate(m):
        d=C.dnum[int(cu[j])]
        W[j,0]=gb; W[j,1]=ga
        near=np.sort(np.abs(dd-d)); W[j,2]=near[1] if len(near)>1 else 20000.0
        W[j,3]=1.0 if (len(bef) and d==bef.max()) else 0.0
        W[j,4]=1.0 if (len(aft) and d==aft.min()) else 0.0
        W[j,5]=nb; W[j,6]=nb/max(1,len(m))
X1=np.hstack([X,Z,W]); FE1=FEATS+EX+GAPF
y=np.array([1 if C.uid[cu[k]] in G[rows[qi[k]].item_id] else 0 for k in range(len(cu))])
cnt=collections.Counter(tr.act_citation); load=[0]*5; fo={}
for a,n in sorted(cnt.items(),key=lambda x:(-x[1],x[0])):
    j=int(np.argmin(load)); fo[a]=j; load[j]+=n
fq=np.array([fo[a] for a in tr.act_citation])
RP=dict(objective='binary',learning_rate=0.05,num_leaves=31,min_data_in_leaf=20,feature_fraction=0.8,
        bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=4,
        deterministic=True,force_row_wise=True)
def bag(p0,Xt,yt,rd=400):
    o=[]
    for sd in (42,202):
        p=dict(p0); p.update(seed=sd,bagging_seed=sd,feature_fraction_seed=sd,data_random_seed=sd)
        o.append(lgb.train(p,lgb.Dataset(Xt,label=yt),num_boost_round=rd))
    return o
def pr(ms,Xt): return np.mean([m.predict(Xt) for m in ms],axis=0) if len(Xt) else np.zeros(0)
def ev(p,thr):
    sc=[]
    for k in range(len(rows)):
        m=np.where(qi==k)[0]; pp=p[m]
        keep=[int(C.uid[cu[m][j]]) for j in range(len(pp)) if pp[j]>=thr] or \
             ([int(C.uid[cu[m][int(np.argmax(pp))]])] if len(pp) else [])
        sc.append(f1(set(keep),set(int(x) for x in G[rows[k].item_id])))
    return float(np.mean(sc))
# ---- stage 1 (with and without the gap features)
for nm,Xa,FEa in (("base",np.hstack([X,Z]),FEATS+EX),("+dategaps",X1,FE1)):
    o=np.zeros(len(y))
    for f in range(5):
        m=fq[qi]!=f; o[~m]=pr(bag(RP,Xa[m],y[m]),Xa[~m])
    best=max((ev(o,t),t) for t in (0.15,0.20,0.25,0.30,0.35))
    print("stage1 %-10s idF1 best %.4f @%.2f"%(nm,best[0],best[1]),flush=True)
    if nm=="+dategaps": o1=o; Xs=Xa; FEs=FEa
# ---- stage 2: score-distribution features on top of stage-1 OOF scores
SF=['s','s_rank','s_frac_max','s_gap_up','s_gap_dn','s_sum','s_n_hi','s_n_mid','s_mean','s_std','n_q2']
S=np.zeros((len(cu),len(SF)),dtype=np.float32)
for k in range(len(rows)):
    m=np.where(qi==k)[0]
    if not len(m): continue
    p=o1[m]; srt=np.sort(p)[::-1]; r=np.argsort(np.argsort(-p,kind='stable'))
    for t,j in enumerate(m):
        S[j,0]=p[t]; S[j,1]=r[t]; S[j,2]=p[t]/max(1e-9,srt[0])
        S[j,3]=(srt[r[t]-1]-p[t]) if r[t]>0 else 0.0
        S[j,4]=(p[t]-srt[r[t]+1]) if r[t]+1<len(srt) else p[t]
        S[j,5]=p.sum(); S[j,6]=(p>0.5).sum(); S[j,7]=((p>0.2)&(p<=0.5)).sum()
        S[j,8]=p.mean(); S[j,9]=p.std(); S[j,10]=len(m)
KEEP=['days','p_napp','p_chg','a_self','f_in','min_dist','tgt_pos','q_found_frac','is_si','g_before','g_after','n_before']
ki=[FEs.index(c) for c in KEEP]
X2=np.hstack([S,Xs[:,ki]]); FE2=SF+KEEP
o2=np.zeros(len(y))
P2=dict(RP); P2.update(num_leaves=15,min_data_in_leaf=40,lambda_l2=3.0)
for f in range(5):
    m=fq[qi]!=f; o2[~m]=pr(bag(P2,X2[m],y[m],300),X2[~m])
for t in (0.15,0.20,0.25,0.30,0.35,0.40,0.45):
    print("  stage2 thr %.2f -> %.4f"%(t,ev(o2,t)))
best2=max((ev(o2,t),t) for t in (0.15,0.20,0.25,0.30,0.35,0.40,0.45))
print("stage2 best idF1 %.4f @%.2f"%best2)
bl=max((ev(0.5*o1+0.5*o2,t),t) for t in (0.15,0.20,0.25,0.30,0.35,0.40))
print("blend  best idF1 %.4f @%.2f  (%.0fs)"%(bl[0],bl[1],time.time()-T0))
np.save('working/o1.npy',o1); np.save('working/o2.npy',o2)
m=bag(P2,X2,y,300)[0]
print("stage2 top:",[(a,int(b)) for a,b in sorted(zip(FE2,m.feature_importance('gain')),key=lambda x:-x[1])[:12]])
