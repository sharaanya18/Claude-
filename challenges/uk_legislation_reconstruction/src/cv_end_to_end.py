"""End-to-end Act-grouped CV with the sequential (imitation-trained) edit gate."""
import sys,json,pickle,time,collections
import numpy as np,pandas as pd
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
import lightgbm as lgb
from lib import norm
from feats import FeatureBuilder,FEATS
from metric import f1,text_score
from pipe import make_query,retrieval_matrix,build_plan
from gate import GFEATS,oracle_walk,model_walk
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
GP=dict(RP); GP.update(num_leaves=31,min_data_in_leaf=20,lambda_l2=2.0)
def bag(p0,Xt,yt,rd):
    out=[]
    for sd in (42,202):
        p=dict(p0); p.update(seed=sd,bagging_seed=sd,feature_fraction_seed=sd,data_random_seed=sd)
        out.append(lgb.train(p,lgb.Dataset(Xt,label=yt),num_boost_round=rd))
    return out
def pr(ms,Xt): return np.mean([m.predict(Xt) for m in ms],axis=0) if len(Xt) else np.zeros(0)
oof=np.zeros(len(y))
for f in range(5):
    m=fq[qi]!=f
    oof[~m]=pr(bag(RP,X[m],y[m],400),X[~m])
log("retrieval OOF")
ID_THR=0.25
idf1=[]
for k in range(len(rows)):
    m=np.where(qi==k)[0]; p=oof[m]
    keep=[int(C.uid[cu[m][j]]) for j in range(len(p)) if p[j]>=ID_THR] or \
         ([int(C.uid[cu[m][int(np.argmax(p))]])] if len(p) else [])
    idf1.append(f1(set(keep),set(int(x) for x in G[rows[k].item_id][0])))
IDF1=float(np.mean(idf1)); log("id F1 @%.2f = %.4f"%(ID_THR,IDF1))

PLAN_THRS=(0.35,0.20,0.10)      # pooled: the gate sees plans of several widths, width is a feature
PLANS={}
def plan_for(k,thr):
    v=PLANS.get((k,thr))
    if v is None:
        m=np.where(qi==k)[0]; p=oof[m]; o=np.argsort(-p,kind='stable')
        sel=[j for j in o if p[j]>=thr] or ([o[0]] if len(o) else [])
        v=build_plan(C,queries[k],[int(cu[m][j]) for j in sel],[float(p[j]) for j in sel],thr)
        PLANS[(k,thr)]=v
    return v
GX,GY,gfold=[],[],[]
orc=[]
for k in range(len(rows)):
    for thr in PLAN_THRS:
        instr,ctxs=plan_for(k,thr)
        gx,gy,out,sc=oracle_walk(queries[k]['en'],instr,ctxs,G[rows[k].item_id][1])
        GX+=gx; GY+=gy; gfold+=[fq[k]]*len(gx)
        if thr==PLAN_THRS[1]: orc.append(sc)
log("plans cached: %d"%len(PLANS))
GX=np.asarray(GX,dtype=np.float32).reshape(-1,len(GFEATS)); GY=np.asarray(GY,dtype=np.int32)
gfold=np.asarray(gfold)
log("gate matrix %s  keep-rate %.3f | oracle text F1 at plan_thr %.2f = %.4f"%(
    GX.shape,GY.mean(),PLAN_THRS[1],float(np.mean(orc))))
gm={}
for f in range(5):
    m=gfold!=f
    gm[f]=bag(GP,GX[m],GY[m],300)
log("gate folds trained")
for PLAN_THR in PLAN_THRS:
    res={}
    for gth in (0.30,0.45,0.60):
        t=[]
        for k in range(len(rows)):
            instr,ctxs=plan_for(k,PLAN_THR)
            ms=gm[fq[k]]
            def predict(row,ms=ms): 
                a=np.asarray(row,dtype=np.float32).reshape(1,-1)
                return float(np.mean([mm.predict(a)[0] for mm in ms]))
            _,_,out=model_walk(queries[k]['en'],instr,ctxs,predict,gth)
            t.append(text_score(rows[k].enacted_text,out,G[rows[k].item_id][1]))
        T=float(np.mean(t)); res[gth]=(100*(0.4*IDF1+0.6*T),100*T)
    best=max(res.items(),key=lambda x:x[1][0])
    print("plan_thr %.2f -> best gate %.2f : text %.2f  TOTAL %.2f   | all: %s"%(
        PLAN_THR,best[0],best[1][1],best[1][0],
        " ".join("%.2f:%.1f"%(g,v[0]) for g,v in res.items())),flush=True)
