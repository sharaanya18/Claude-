import sys,json,pickle,time,collections
import numpy as np,pandas as pd
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
import lightgbm as lgb
from feats import FeatureBuilder,FEATS
from metric import f1
from pipe import make_query,retrieval_matrix
from cif import build_cif,cif_feats,CIF_FEATS
T0=time.time()
C=pickle.load(open('working/corpus.pkl','rb')); FB=FeatureBuilder(C)
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
G={r.item_id:set(json.loads(r.amending_ids)) for r in tt.itertuples()}
rows=list(tr.itertuples()); queries=[make_query(r) for r in rows]
X,qi,cu=retrieval_matrix(C,FB,queries)
print("matrix %s in %.0fs"%(X.shape,time.time()-T0),flush=True)
# within-query rank features
EXTRA=['r_days','r_idf','r_pnapp','n_q','s_max_pnins']
Z=np.zeros((len(cu),len(EXTRA)),dtype=np.float32)
ix={k:FEATS.index(k) for k in ('days','idf_ov','p_napp','p_nins')}
for k in range(len(rows)):
    m=np.where(qi==k)[0]
    if not len(m): continue
    for j,(nm,col) in enumerate([('r_days','days'),('r_idf','idf_ov'),('r_pnapp','p_napp')]):
        v=X[m,ix[col]]; Z[m,j]=np.argsort(np.argsort(-v))/max(1,len(m)-1)
    Z[m,3]=len(m); Z[m,4]=X[m,ix['p_nins']].max()
CIF=build_cif(C)
W=np.zeros((len(cu),len(CIF_FEATS)),dtype=np.float32)
for k in range(len(cu)):
    i=int(cu[k]); Q=queries[qi[k]]
    ff=cif_feats(CIF,C.cit[i],C.label[i],Q['qd'],float(C.dnum[i]))
    W[k]=[ff[n] for n in CIF_FEATS]
X=np.hstack([X,Z]); FE=FEATS+EXTRA
y=np.array([1 if C.uid[cu[k]] in G[rows[qi[k]].item_id] else 0 for k in range(len(cu))])
acts=tr.act_citation.to_numpy(); cnt=collections.Counter(acts); fo={}; load=[0]*5
for a,n in sorted(cnt.items(),key=lambda x:-x[1]):
    j=int(np.argmin(load)); fo[a]=j; load[j]+=n
fq=np.array([fo[a] for a in acts])
P=dict(objective='binary',learning_rate=0.05,num_leaves=31,min_data_in_leaf=20,
       feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,
       verbose=-1,seed=42,num_threads=4,deterministic=True,force_row_wise=True)
oof=np.zeros(len(y))
for f in range(5):
    m=fq[qi]!=f
    ps=[]
    for sd in (42,202):
        p=dict(P); p['seed']=sd; p['bagging_seed']=sd; p['feature_fraction_seed']=sd
        ps.append(lgb.train(p,lgb.Dataset(X[m],label=y[m]),num_boost_round=400).predict(X[~m]))
    oof[~m]=np.mean(ps,axis=0)
def ev_ef1(alpha,floor):
    """Expected-F1 selection: pick the k maximising 2*S_k/(k+E|T|) over the ranked list."""
    sc=[]
    for k,r in enumerate(rows):
        m=qi==k; p=oof[m]; u=C.uid[cu[m]]
        o=np.argsort(-p); ps=p[o]
        ET=max(1e-6,alpha*ps.sum()); S=np.cumsum(ps)
        kk=np.arange(1,len(ps)+1)
        e=2*S/(kk+ET)
        best=int(np.argmax(e))+1
        keep=[int(u[j]) for j in o[:best] if ps[list(o).index(j)] if True]
        keep=[int(u[j]) for j in o[:best]]
        keep=[q for q,pp in zip(keep,ps[:best]) if pp>=floor] or keep[:1]
        sc.append(f1(set(keep),set(int(x) for x in G[r.item_id])))
    return float(np.mean(sc))

def ev(thr):
    sc=[]
    for k,r in enumerate(rows):
        m=qi==k; p=oof[m]; u=C.uid[cu[m]]; o=np.argsort(-p)
        keep=[int(u[j]) for j in o if p[j]>=thr] or ([int(u[o[0]])] if len(o) else [])
        sc.append(f1(set(keep),set(int(x) for x in G[r.item_id])))
    return float(np.mean(sc))
for t in [0.15,0.2,0.25,0.3,0.35,0.4]: print("  thr %.2f -> %.4f"%(t,ev(t)))
best=max(((ev(t),('thr',t)) for t in [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5]))
for a in (0.8,1.0,1.2):
    for fl in (0.0,0.05,0.1,0.15):
        v=ev_ef1(a,fl); print("  EF1 alpha=%.1f floor=%.2f -> %.4f"%(a,fl,v))
        if v>best[0]: best=(v,('ef1',a,fl))
print("BEST idF1 %.4f via %s   (%.0fs)"%(best[0],best[1],time.time()-T0))
# per-fold spread of the best threshold rule
pf=[]
for f in range(5):
    sc=[]
    for k,r in enumerate(rows):
        if fq[k]!=f: continue
        m=qi==k; p=oof[m]; u=C.uid[cu[m]]; o=np.argsort(-p)
        keep=[int(u[j]) for j in o if p[j]>=0.25] or ([int(u[o[0]])] if len(o) else [])
        sc.append(f1(set(keep),set(int(x) for x in G[r.item_id])))
    pf.append(np.mean(sc))
print("per-fold idF1 @0.25:",[round(x,4) for x in pf],"mean %.4f std %.4f"%(np.mean(pf),np.std(pf)))
m=lgb.train(P,lgb.Dataset(X,label=y),num_boost_round=400)
print("top:",[(a,int(b)) for a,b in sorted(zip(FE,m.feature_importance('gain')),key=lambda x:-x[1])[:16]])
np.save('working/Xret.npy',X); np.save('working/yret.npy',y); np.save('working/qiret.npy',qi); np.save('working/curet.npy',cu)
pickle.dump(FE,open('working/FE.pkl','wb'))
