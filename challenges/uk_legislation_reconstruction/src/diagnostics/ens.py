"""Test model diversity for both stages (runtime headroom is large)."""
import sys,json,pickle,time,collections
import numpy as np,pandas as pd
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
import lightgbm as lgb
from lib import norm
from feats import FeatureBuilder,FEATS
from metric import f1,text_score
from pipe import make_query,retrieval_matrix,build_plan,apply_gated
from gate import plan_rows,GFEATS
T0=time.time()
def log(m): print("[%5.0fs] %s"%(time.time()-T0,m),flush=True)
D='dataset/public/'
C=pickle.load(open('working/corpus.pkl','rb')); FB=FeatureBuilder(C)
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
G={r.item_id:(set(json.loads(r.amending_ids)),norm(r.text_at_date)) for r in tt.itertuples()}
rows=list(tr.itertuples()); queries=[make_query(r) for r in rows]
X,qi,cu=retrieval_matrix(C,FB,queries)
EX=['r_days','r_idf','r_pnapp','n_q','s_max_pnins']
Z=np.zeros((len(cu),len(EX)),dtype=np.float32)
ix={k:FEATS.index(k) for k in ('days','idf_ov','p_napp','p_nins')}
for k in range(len(rows)):
    m=np.where(qi==k)[0]
    if not len(m): continue
    for j,col in enumerate(('days','idf_ov','p_napp')):
        Z[m,j]=np.argsort(np.argsort(-X[m,ix[col]]))/max(1,len(m)-1)
    Z[m,3]=len(m); Z[m,4]=X[m,ix['p_nins']].max()
X=np.hstack([X,Z])
y=np.array([1 if C.uid[cu[k]] in G[rows[qi[k]].item_id][0] else 0 for k in range(len(cu))])
cnt=collections.Counter(tr.act_citation); load=[0]*5; fo={}
for a,n in sorted(cnt.items(),key=lambda x:(-x[1],x[0])):
    j=int(np.argmin(load)); fo[a]=j; load[j]+=n
fq=np.array([fo[a] for a in tr.act_citation])
BASE=dict(objective='binary',learning_rate=0.05,num_leaves=31,min_data_in_leaf=20,feature_fraction=0.8,
          bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=4,
          deterministic=True,force_row_wise=True)
ALT=dict(BASE); ALT.update(num_leaves=63,min_data_in_leaf=10,learning_rate=0.03,extra_trees=True,
                           feature_fraction=0.6,lambda_l2=5.0)
def oof_for(cfgs,rounds):
    o=np.zeros(len(y))
    for f in range(5):
        m=fq[qi]!=f; ps=[]
        for params,rd in cfgs:
            for sd in (42,202):
                p=dict(params); p.update(seed=sd,bagging_seed=sd,feature_fraction_seed=sd,data_random_seed=sd)
                ps.append(lgb.train(p,lgb.Dataset(X[m],label=y[m]),num_boost_round=rd).predict(X[~m]))
        o[~m]=np.mean(ps,axis=0)
    return o
def ev(o,thr=0.25):
    sc=[]
    for k,r in enumerate(rows):
        m=qi==k; p=o[m]; u=C.uid[cu[m]]; ordr=np.argsort(-p)
        keep=[int(u[j]) for j in ordr if p[j]>=thr] or ([int(u[ordr[0]])] if len(ordr) else [])
        sc.append(f1(set(keep),set(int(x) for x in G[r.item_id][0])))
    return float(np.mean(sc))
for name,cfgs,rd in [("base400",[(BASE,400)],0),("base800",[(BASE,800)],0),
                     ("base+alt",[(BASE,400),(ALT,700)],0)]:
    o=oof_for(cfgs,0) if False else None
    o=np.zeros(len(y))
    for f in range(5):
        m=fq[qi]!=f; ps=[]
        for params,rr in cfgs:
            for sd in (42,202):
                p=dict(params); p.update(seed=sd,bagging_seed=sd,feature_fraction_seed=sd,data_random_seed=sd)
                ps.append(lgb.train(p,lgb.Dataset(X[m],label=y[m]),num_boost_round=rr).predict(X[~m]))
        o[~m]=np.mean(ps,axis=0)
    print("%-10s idF1 @0.20 %.4f @0.25 %.4f @0.30 %.4f  (%.0fs)"%(name,ev(o,0.20),ev(o,0.25),ev(o,0.30),time.time()-T0),flush=True)
    np.save('working/oof_%s.npy'%name,o)
