"""Nested (fold-disjoint) evaluation of the two selection thresholds.

The headline CV number picks the retrieval and gate thresholds on the same OOF it reports,
which flatters it.  Here each fold's thresholds are chosen on the OTHER four folds only, so
the reported figure contains no selection on the fold being scored.
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
GP=dict(RP); GP.update(num_leaves=15,min_data_in_leaf=30,lambda_l2=2.0)
def bag(params,Xt,yt,rd):
    out=[]
    for sd in (42,202):
        p=dict(params); p.update(seed=sd,bagging_seed=sd,feature_fraction_seed=sd,data_random_seed=sd)
        out.append(lgb.train(p,lgb.Dataset(Xt,label=yt),num_boost_round=rd))
    return out
def pred(ms,Xt): return np.mean([m.predict(Xt) for m in ms],axis=0) if len(Xt) else np.zeros(0)
oof=np.zeros(len(y))
for f in range(5):
    m=fq[qi]!=f
    oof[~m]=pred(bag(RP,X[m],y[m],400),X[~m])
log("retrieval OOF")
RT=[0.15,0.20,0.25,0.30,0.35]; GT=[-9,0.15,0.20,0.25,0.30,0.35,0.40]
# per (retrieval threshold) -> plans and gate OOF, then per-query scores for every gate threshold
S=np.zeros((len(RT),len(GT),len(rows)))      # total per query
for ri,rth in enumerate(RT):
    GX,GY,gq,plans=[],[],[],[]
    for k in range(len(rows)):
        m=qi==k; p=oof[m]; rr=cu[m]; o=np.argsort(-p,kind='stable')
        sel=[j for j in o if p[j]>=rth] or ([o[0]] if len(o) else [])
        pr=[int(rr[j]) for j in sel]; ps=[float(p[j]) for j in sel]
        instr,ctxs=build_plan(C,queries[k],pr,ps)
        gx,gy,_=plan_rows(queries[k]['en'],instr,ctxs,G[rows[k].item_id][1])
        plans.append((instr,ctxs,[int(C.uid[i]) for i in pr]))
        GX+=gx; GY+=gy; gq+=[k]*len(gx)
    GX=np.asarray(GX,dtype=np.float32).reshape(-1,len(GFEATS)); GY=np.asarray(GY,dtype=np.float32)
    gq=np.asarray(gq); glab=(GY>1e-9).astype(int); gfold=fq[gq] if len(gq) else np.zeros(0,int)
    goof=np.zeros(len(GY))
    for f in range(5):
        m=gfold!=f
        if m.sum()>50 and (~m).sum(): goof[~m]=pred(bag(GP,GX[m],glab[m],300),GX[~m])
    for k in range(len(rows)):
        instr,ctxs,kept=plans[k]
        a=f1(set(kept),set(int(x) for x in G[rows[k].item_id][0]))
        gp=goof[gq==k] if len(gq) else np.zeros(0)
        for gi,gth in enumerate(GT):
            out=apply_gated(queries[k]['en'],instr,ctxs,gp,gth)
            S[ri,gi,k]=0.40*a+0.60*text_score(rows[k].enacted_text,out,G[rows[k].item_id][1])
    log("rth=%.2f done"%rth)
flat=100*S.mean(axis=2)
print("\n          "+"  ".join("g=%5.2f"%g for g in GT))
for ri,rth in enumerate(RT):
    print("r=%.2f  "%rth+"  ".join("%6.2f"%v for v in flat[ri]))
bi=np.unravel_index(np.argmax(flat),flat.shape)
print("\nbest-on-all-folds: r=%.2f g=%.2f -> %.2f  (the optimistic headline)"%(RT[bi[0]],GT[bi[1]],flat[bi]))
# nested: thresholds chosen on the other four folds
tot=[]
for f in range(5):
    tm=fq!=f; vm=fq==f
    inner=S[:,:,tm].mean(axis=2)
    j=np.unravel_index(np.argmax(inner),inner.shape)
    v=100*S[j[0],j[1],vm].mean()
    tot.append((v,vm.sum(),RT[j[0]],GT[j[1]]))
    print("fold %d: inner pick r=%.2f g=%.2f -> held-out total %.2f (n=%d)"%(f,RT[j[0]],GT[j[1]],v,vm.sum()))
w=sum(v*n for v,n,_,_ in tot)/sum(n for _,n,_,_ in tot)
print("NESTED (query-weighted) total = %.2f   |  unweighted fold mean = %.2f"%(w,np.mean([v for v,_,_,_ in tot])))
np.save('working/S_grid.npy',S)
