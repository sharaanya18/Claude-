import sys; sys.argv=['x','dataset/public','working/x.csv']
import numpy as np, pandas as pd, json, scipy.sparse as sp, time
import solution as S
from common import hungarian, score
L=pd.read_csv('dataset/public/listens.csv').sort_values(['session','relative_position']).reset_index(drop=True)
C=pd.read_csv('dataset/public/continuations.csv'); tr=pd.read_csv('dataset/public/train.csv'); te=pd.read_csv('dataset/public/test.csv')
for d in (tr,te): d['pre']=d.prefixes.map(json.loads); d['cand']=d.candidates.map(json.loads)
Y=tr[[f'match_{i}' for i in range(1,7)]].to_numpy()-1
D=S.Data(L,C,set(x for l in te.pre for x in l)); fit=~D.is_test; fi=np.where(fit)[0]
Xb=D.Ca.copy(); Xb.data[:]=1; nu=np.asarray(Xb.sum(1)).ravel(); df=np.asarray(Xb[fi].sum(0)).ravel()
cos=S.normalize  # reuse
W=S.idf_diag(D.Ca,fi,1.0); F=S.normalize(S.tf_apply(D.Ca,'log')@W).tocsr(); Q=S.normalize(S.tf_apply(D.C1,'log')@W).tocsr()
Pb=D.C1.copy(); Pb.data[:]=1; Pb=Pb.tocsr()
res={}
t0=time.time()
for i,(_,r) in enumerate(tr.iterrows()):
    pre=[D.sid[x] for x in r.pre]; arts=[a for c in r.cand for a in D.cc[c]]
    ss=[]; 
    Tcsc=Xb.tocsc() if i==0 else Tcsc
    lists=[]
    for a in arts:
        s_=Tcsc.indices[Tcsc.indptr[a]:Tcsc.indptr[a+1]]; s_=s_[fit[s_]&~np.isin(s_,pre)]; lists.append(s_)
    U=np.unique(np.concatenate(lists)); pos={u:k for k,u in enumerate(U)}
    K=(Q[pre]@F[U].T).toarray()                                  # (6,|U|) kernel
    # RP3beta-style: item->user prob 1/df_a for prefix artists, user->item prob 1/n_u^g
    Pa=Pb[pre]; Aset=[np.unique(Pa[j].indices) for j in range(6)]
    for g in (0,0.5,1.0):
        for beta in (0,0.5):
            M=np.zeros((6,6))
            for k in range(6):
                v=[]
                for q in (0,1):
                    a=D.cc[r.cand[k]][q]; idx=np.array([pos[u] for u in lists[2*k+q]],int)
                    v.append((K[:,idx]/(nu[lists[2*k+q]]**g)).sum(1)/(max(df[a],1)**beta) if len(idx) else np.zeros(6))
                M[:,k]=np.mean(v,0)
            res.setdefault((g,beta),[]).append(M)
    if i%300==0: print(i,time.time()-t0,flush=True)
for k,v in res.items():
    allv=np.concatenate([m.ravel() for m in v]); sc=np.median(allv[allv>0])
    out={e:round(score(np.array([hungarian(np.log(m+e*sc)) for m in v]),Y),4) for e in (.03,.1,.3)}
    print(k,out)
