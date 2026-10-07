"""Does the text plan want a different retrieval threshold from the id list?

The two terms are scored separately: a borderline provision that supplies one correct edit
can pay for itself in the text term (the gate can still veto its edits) while costing
precision in the id term.  Here the id threshold is held at 0.25 and the threshold used to
build the reconstruction plan is swept.
"""
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
Z=np.zeros((len(cu),5),dtype=np.float32)
ix={k:FEATS.index(k) for k in ('days','idf_ov','p_napp','p_nins')}
for k in range(len(rows)):
    m=np.where(qi==k)[0]
    if not len(m): continue
    for j,col in enumerate(('days','idf_ov','p_napp')):
        Z[m,j]=np.argsort(np.argsort(-X[m,ix[col]],kind='stable'))/max(1,len(m)-1)
    Z[m,3]=len(m); Z[m,4]=X[m,ix['p_nins']].max()
X=np.hstack([X,Z])
y=np.array([1 if C.uid[cu[k]] in G[rows[qi[k]].item_id][0] else 0 for k in range(len(cu))])
cnt=collections.Counter(tr.act_citation); load=[0]*5; fo={}
for a,n in sorted(cnt.items(),key=lambda x:(-x[1],x[0])):
    j=int(np.argmin(load)); fo[a]=j; load[j]+=n
fq=np.array([fo[a] for a in tr.act_citation])
RP=dict(objective='binary',learning_rate=0.05,num_leaves=31,min_data_in_leaf=20,feature_fraction=0.8,
        bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=4,
        deterministic=True,force_row_wise=True)
GP=dict(RP); GP.update(num_leaves=15,min_data_in_leaf=30,lambda_l2=2.0)
def bag(p0,Xt,yt,rd):
    out=[]
    for sd in (42,202):
        p=dict(p0); p.update(seed=sd,bagging_seed=sd,feature_fraction_seed=sd,data_random_seed=sd)
        out.append(lgb.train(p,lgb.Dataset(Xt,label=yt),num_boost_round=rd))
    return out
def pred(ms,Xt): return np.mean([m.predict(Xt) for m in ms],axis=0) if len(Xt) else np.zeros(0)
oof=np.zeros(len(y))
for f in range(5):
    m=fq[qi]!=f
    oof[~m]=pred(bag(RP,X[m],y[m],400),X[~m])
log("retrieval OOF")
ID_THR=0.25
idf1=[]
for k in range(len(rows)):
    m=np.where(qi==k)[0]; p=oof[m]
    keep=[int(C.uid[cu[m][j]]) for j in range(len(p)) if p[j]>=ID_THR] or \
         ([int(C.uid[cu[m][int(np.argmax(p))]])] if len(p) else [])
    idf1.append(f1(set(keep),set(int(x) for x in G[rows[k].item_id][0])))
IDF1=float(np.mean(idf1)); log("id F1 at %.2f = %.4f"%(ID_THR,IDF1))
for PLAN_THR in (0.25,0.20,0.15,0.10,0.05):
    GX,GY,gq,plans=[],[],[],[]
    for k in range(len(rows)):
        m=np.where(qi==k)[0]; p=oof[m]; o=np.argsort(-p,kind='stable')
        sel=[j for j in o if p[j]>=PLAN_THR] or ([o[0]] if len(o) else [])
        pr=[int(cu[m][j]) for j in sel]; ps=[float(p[j]) for j in sel]
        instr,ctxs=build_plan(C,queries[k],pr,ps)
        gx,gy,_=plan_rows(queries[k]['en'],instr,ctxs,G[rows[k].item_id][1])
        plans.append((instr,ctxs)); GX+=gx; GY+=gy; gq+=[k]*len(gx)
    GX=np.asarray(GX,dtype=np.float32).reshape(-1,len(GFEATS)); GY=np.asarray(GY,dtype=np.float32)
    gq=np.asarray(gq); glab=(GY>1e-9).astype(int); gfold=fq[gq] if len(gq) else np.zeros(0,int)
    goof=np.zeros(len(GY))
    for f in range(5):
        m2=gfold!=f
        if m2.sum()>50 and (~m2).sum(): goof[~m2]=pred(bag(GP,GX[m2],glab[m2],300),GX[~m2])
    best=None
    for gth in (-9,0.15,0.20,0.25,0.30,0.35):
        t=[text_score(rows[k].enacted_text,
                      apply_gated(queries[k]['en'],plans[k][0],plans[k][1],goof[gq==k],gth),
                      G[rows[k].item_id][1]) for k in range(len(rows))]
        T=float(np.mean(t)); tot=100*(0.4*IDF1+0.6*T)
        if best is None or tot>best[0]: best=(tot,gth,T)
    print("plan_thr %.2f  (n_instr %d) -> best gate %.2f : text %.4f  TOTAL %.2f"%(
        PLAN_THR,len(GY),best[1],best[2],best[0]),flush=True)
