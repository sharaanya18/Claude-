import numpy as np, time, scipy.sparse as sp
from feats import *
L,C,tr,te=load(); D=Data(L,C,set(s for l in te.pre for s in l))
fold=np.random.RandomState(0).randint(0,5,len(tr)); Y=np.array(list(tr.y))
Xb=D.Xall.tocsr()
t0=time.time(); res={}
for f in range(5):
    R=tr[fold==f]; held=np.zeros(D.NS,bool)
    for l in R.pre:
        for x in l: held[D.sid[x]]=True
    fi=np.where(~D.is_test&~held)[0]; N=len(fi)
    Xf=Xb[fi]; df=np.asarray(Xf.sum(0)).ravel()
    cols=sorted({a for c in R.cand for ci in c for a in D.cc[ci]}); cm={a:i for i,a in enumerate(cols)}
    Pidx=[D.sid[x] for l in R.pre for x in l]
    Pm=D.X1[Pidx].tocsr()                         # binary prefix bags
    A=np.unique(Pm.indices); amap=-np.ones(D.NA,int); amap[A]=np.arange(len(A))
    Xc=Xf[:,cols].tocsc(); XA=Xf[:,A].tocsc()
    G=(XA.T@Xc).tocsr()                           # (|A|, ncols) co-occurrence counts
    pb=df[cols]/N; dfa=df[A]
    for alpha in (1,5,20):
        # P(b|a) smoothed, log lift; zero-df prefix artists contribute 0
        Gc=G.tocoo()
        lift=np.log((Gc.data+alpha*pb[Gc.col])/(dfa[Gc.row]+alpha)/pb[Gc.col])     # for observed pairs
        base=np.log((alpha*pb[None,:]/(dfa[:,None]+alpha))/pb[None,:])               # unobserved: log(alpha/(df_a+alpha)), same for all b
        # dense per-prefix score = sum_a [base_a + (lift-base)_observed] / n_known
        Lm=sp.csr_matrix((lift-np.log(alpha/(dfa[Gc.row]+alpha)),(Gc.row,Gc.col)),shape=G.shape)
        Pa=Pm[:,A].tocsr()
        for mode in ('sum','mean','wsum'):
            Pw=Pa.copy().astype(np.float32)
            if mode=='wsum': Pw=Pw@sp.diags((1/np.sqrt(dfa+1)).astype(np.float32))
            S=(Pw@Lm).toarray()+np.asarray(Pw@np.log(alpha/(dfa+alpha)))[:,None]
            if mode=='mean': S=S/np.asarray(Pa.sum(1))
            for ri,(idx,r) in enumerate(R.iterrows()):
                M=np.zeros((6,6))
                for j in range(6):
                    for k,ci in enumerate(r.cand): M[j,k]=np.mean([S[ri*6+j,cm[a]] for a in D.cc[ci]])
                res.setdefault((alpha,mode),{})[idx]=M
    print(f,time.time()-t0,flush=True)
for k,dd in res.items():
    print(k,'hung',round(score(np.array([hungarian(dd[i]) for i in tr.index]),Y),4))
