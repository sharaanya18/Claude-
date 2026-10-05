# DIAGNOSTIC ONLY (never shipped): value of other held-out rows' first halves as co-occurrence data,
# emulating what fitting on test first halves (banned by the rules) would add.
import numpy as np, time, scipy.sparse as sp
from feats import *
L,C,tr,te=load(); D=Data(L,C,set(s for l in te.pre for s in l))
fold=np.random.RandomState(0).randint(0,5,len(tr)); Y=np.array(list(tr.y))
Xb=D.Xall.tocsr(); res={'fit only':{},'fit + held-out first halves':{}}
for f in range(5):
    R=tr[fold==f]; held=np.zeros(D.NS,bool)
    for l in R.pre:
        for x in l: held[D.sid[x]]=True
    fi=np.where(~D.is_test&~held)[0]; hi=np.where(held)[0]
    cols=sorted({a for c in R.cand for ci in c for a in D.cc[ci]}); cm={a:i for i,a in enumerate(cols)}
    Pidx=[D.sid[x] for l in R.pre for x in l]; Pm=D.X1[Pidx].tocsr(); A=np.unique(Pm.indices); Pa=Pm[:,A].tocsr()
    for name,Xf in (('fit only',Xb[fi]),('fit + held-out first halves',sp.vstack([Xb[fi],D.X1[hi]]).tocsr())):
        df=np.asarray(Xf.sum(0)).ravel(); G=(Xf[:,A].T@Xf[:,cols].tocsc()).tocoo()
        sim=G.data/((df[A][G.row]**.5)*(df[cols][G.col]**.5)+5)
        S=(Pa@sp.csr_matrix((sim,(G.row,G.col)),shape=G.shape)).toarray()
        for ri,(idx,r) in enumerate(R.iterrows()):
            M=np.zeros((6,6))
            for j in range(6):
                for k,ci in enumerate(r.cand): M[j,k]=np.mean([S[ri*6+j,cm[a]] for a in D.cc[ci]])
            res[name][idx]=M
for k,dd in res.items():
    allv=np.concatenate([dd[i].ravel() for i in tr.index]); s0=np.median(allv[allv>0])
    print(k, round(score(np.array([hungarian(np.log(dd[i]+.3*s0)) for i in tr.index]),Y),4))
