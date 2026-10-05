import numpy as np, time, scipy.sparse as sp
from feats import *
L,C,tr,te=load(); D=Data(L,C,set(s for l in te.pre for s in l))
fold=np.random.RandomState(0).randint(0,5,len(tr)); Y=np.array(list(tr.y))
Xb=D.Xall.tocsr(); t0=time.time(); res={}
for f in range(5):
    R=tr[fold==f]; held=np.zeros(D.NS,bool)
    for l in R.pre:
        for x in l: held[D.sid[x]]=True
    fi=np.where(~D.is_test&~held)[0]; Xf=Xb[fi]; df=np.asarray(Xf.sum(0)).ravel()
    cols=sorted({a for c in R.cand for ci in c for a in D.cc[ci]}); cm={a:i for i,a in enumerate(cols)}
    Pidx=[D.sid[x] for l in R.pre for x in l]; Pm=D.X1[Pidx].tocsr()
    A=np.unique(Pm.indices); Pa=Pm[:,A].tocsr()
    G=(Xf[:,A].T@Xf[:,cols].tocsc()).tocoo()
    for alpha in (0.5,0.8):
        for h in (0,5):
            sim=G.data/((df[A][G.row]**alpha)*(df[cols][G.col]**(1-alpha))+h)
            for q in (1,2,3):
                Sm=sp.csr_matrix((sim**q,(G.row,G.col)),shape=G.shape)
                S=(Pa@Sm).toarray()
                for ri,(idx,r) in enumerate(R.iterrows()):
                    M=np.zeros((6,6))
                    for j in range(6):
                        for k,ci in enumerate(r.cand): M[j,k]=np.mean([S[ri*6+j,cm[a]] for a in D.cc[ci]])
                    res.setdefault((alpha,h,q),{})[idx]=M
    print(f,round(time.time()-t0),flush=True)
for k,dd in res.items():
    allv=np.concatenate([dd[i].ravel() for i in tr.index]); s0=np.median(allv[allv>0])
    out={e:round(score(np.array([hungarian(np.log(dd[i]+e*s0)) for i in tr.index]),Y),4) for e in (.1,.3,1)}
    print('alpha',k[0],'shrink',k[1],'q',k[2],out)
