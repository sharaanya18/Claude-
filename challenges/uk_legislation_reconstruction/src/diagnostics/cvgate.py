"""CV for the edit gate on top of OOF retrieval."""
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
EXTRA=['r_days','r_idf','r_pnapp','n_q','s_max_pnins']
Z=np.zeros((len(cu),len(EXTRA)),dtype=np.float32)
ix={k:FEATS.index(k) for k in ('days','idf_ov','p_napp','p_nins')}
for k in range(len(rows)):
    m=np.where(qi==k)[0]
    if not len(m): continue
    for j,col in enumerate(('days','idf_ov','p_napp')):
        v=X[m,ix[col]]; Z[m,j]=np.argsort(np.argsort(-v))/max(1,len(m)-1)
    Z[m,3]=len(m); Z[m,4]=X[m,ix['p_nins']].max()
X=np.hstack([X,Z]); FE=FEATS+EXTRA
y=np.array([1 if C.uid[cu[k]] in G[rows[qi[k]].item_id][0] else 0 for k in range(len(cu))])
acts=tr.act_citation.to_numpy(); cnt=collections.Counter(acts); fo={}; load=[0]*5
for a,n in sorted(cnt.items(),key=lambda x:-x[1]):
    j=int(np.argmin(load)); fo[a]=j; load[j]+=n
fq=np.array([fo[a] for a in acts])
RP=dict(objective='binary',learning_rate=0.05,num_leaves=31,min_data_in_leaf=20,feature_fraction=0.8,
        bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,seed=42,num_threads=4,
        deterministic=True,force_row_wise=True)
oof=np.zeros(len(y))
for f in range(5):
    m=fq[qi]!=f
    ps=[]
    for sd in (42,202):
        p=dict(RP); p.update(seed=sd,bagging_seed=sd,feature_fraction_seed=sd)
        ps.append(lgb.train(p,lgb.Dataset(X[m],label=y[m]),num_boost_round=400).predict(X[~m]))
    oof[~m]=np.mean(ps,axis=0)
log("retrieval OOF")
THR=0.25
GX,GY,gfold,gq=[],[],[],[]; plans=[]
for k,r in enumerate(rows):
    m=qi==k; p=oof[m]; uu=C.uid[cu[m]]; rr=cu[m]; o=np.argsort(-p)
    sel=[j for j in o if p[j]>=THR] or ([o[0]] if len(o) else [])
    prows=[int(rr[j]) for j in sel]; ps=[float(p[j]) for j in sel]
    instr,ctxs=build_plan(C,queries[k],prows,ps)
    gx,gy,_=plan_rows(queries[k]['en'],instr,ctxs,G[r.item_id][1])
    plans.append((instr,ctxs,[int(uu[j]) for j in sel]))
    GX+=gx; GY+=gy; gfold+=[fq[k]]*len(gx); gq+=[k]*len(gx)
GX=np.asarray(GX,dtype=np.float32).reshape(-1,len(GFEATS)); GY=np.asarray(GY,dtype=np.float32)
gfold=np.asarray(gfold); gq=np.asarray(gq)
log("gate matrix %s pos=%.3f neg=%.3f"%(GX.shape,(GY>1e-9).mean(),(GY<-1e-9).mean()))
def gate_oof(obj):
    P=dict(objective=obj,learning_rate=0.05,num_leaves=15,min_data_in_leaf=30,feature_fraction=0.8,
           bagging_fraction=0.8,bagging_freq=1,lambda_l2=2.0,verbose=-1,seed=42,num_threads=4,
           deterministic=True,force_row_wise=True)
    lab=(GY>1e-9).astype(int) if obj=='binary' else GY
    out=np.zeros(len(GY))
    for f in range(5):
        m=gfold!=f
        if m.sum()>50 and (~m).sum():
            out[~m]=lgb.train(P,lgb.Dataset(GX[m],label=lab[m]),num_boost_round=300).predict(GX[~m])
    return out
def total(gp,thr):
    a=[];b=[]
    for k,r in enumerate(rows):
        instr,ctxs,kept=plans[k]
        out=apply_gated(queries[k]['en'],instr,ctxs,gp[gq==k],thr)
        a.append(f1(set(kept),set(int(x) for x in G[r.item_id][0])))
        b.append(text_score(r.enacted_text,out,G[r.item_id][1]))
    A,B=float(np.mean(a)),float(np.mean(b)); return 100*(0.4*A+0.6*B),100*A,100*B
for obj,grid in (('binary',[-9,0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6]),
                 ('regression',[-9,-0.005,0.0,0.002,0.005,0.01])):
    gp=gate_oof(obj)
    log("gate=%s"%obj)
    for t in grid:
        s,a,b=total(gp,t); print("   thr %7.3f -> total %6.2f  id %5.2f  text %6.2f"%(t,s,a,b))
