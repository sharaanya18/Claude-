import numpy as np, pandas as pd, scipy.sparse as sp, time, scipy.linalg as sl
from common import *
from sklearn.preprocessing import normalize
src=open('exp1.py').read().split("rows=tr.copy()")[0]; exec(src)
rows=tr.copy(); fold=np.random.RandomState(0).randint(0,5,len(rows)); Y=np.array(list(rows.y))
t0=time.time()
# only use folds 0,1 for speed
cfgs=[('idf.5',0.5,[1e-1,1,10]),]
out={}
for f in range(2):
    R=rows[fold==f]
    held=np.zeros(NS,bool)
    for l in R.pre:
        for s in l: held[sid[s]]=True
    fi=np.where(~is_test&~held)[0]
    Pidx=[sid[s] for l in R.pre for s in l]
    cols=sorted({a for c in R.cand for ci in c for a in cc[ci]}); cm={a:i for i,a in enumerate(cols)}
    for w in (0.0,0.5,1.0):
        W=sp.diags(idf**w)
        Xf=normalize(X1[fi]@W); Xp=normalize(X1[Pidx]@W)
        K=(Xf@Xf.T).toarray(); Kp=(Xp@Xf.T).toarray()
        for lam in (0.03,0.1,0.3,1.0):
            Kl=K+lam*np.eye(len(fi))
            cf=sl.cho_factor(Kl,lower=True,check_finite=False)
            Wt=sl.cho_solve(cf,Kp.T,check_finite=False).T    # (nP, nfit)
            for tname,T in (('h2',X2),('all',Xall)):
                S=Wt@T[fi][:,cols].toarray()
                res=[]
                for ri,(idx,r) in enumerate(R.iterrows()):
                    M=np.zeros((6,6))
                    for j in range(6):
                        for k,ci in enumerate(r.cand): M[j,k]=np.mean([S[ri*6+j,cm[a]] for a in cc[ci]])
                    res.append((idx,M))
                out.setdefault((w,lam,tname),[]).extend(res)
        print('f',f,'w',w,time.time()-t0,flush=True)
import pickle; pickle.dump(out,open('/tmp/out5.pkl','wb'))
for k,v in out.items():
    ids=[i for i,_ in v]; y=np.array([rows.y[i] for i in ids])
    raw=score(np.array([hungarian(M) for _,M in v]),y)
    # exp transform with temperature relative to std
    ex=score(np.array([hungarian(np.exp(M/(M.std()+1e-9))) for _,M in v]),y)
    print(k,round(raw,4),round(ex,4))
