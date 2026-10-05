import numpy as np, time, scipy.sparse as sp, pickle
from feats import *
import nn
L,C,tr,te=load(); D=Data(L,C,set(s for l in te.pre for s in l))
fold=np.random.RandomState(0).randint(0,5,len(tr)); Y=np.array(list(tr.y))
# target vocab: artists occurring in >=3 first halves among all non-test sessions (continuations are drawn from these)
X1=D.X1; Xall=D.Xall
def run_fold(f,epochs=25,d=256,drop=0.5,aug_n=0,seed=0,verbose=False):
    R=tr[fold==f]; held=np.zeros(D.NS,bool)
    for l in R.pre:
        for x in l: held[D.sid[x]]=True
    fi=np.where(~D.is_test&~held)[0]
    cnt=np.asarray(Xall[fi].sum(0)).ravel(); vin=np.where(cnt>=2)[0]; 
    c1=np.asarray(X1[fi].sum(0)).ravel(); vt=np.where(c1>=2)[0]
    inmap=-np.ones(D.NA,int); inmap[vin]=np.arange(len(vin)); tmap=-np.ones(D.NA,int); tmap[vt]=np.arange(len(vt))
    wvec=np.log((len(fi)+1)/(cnt[vin]+1))**0.5
    Xi=X1[fi][:,vin].tocsr(); has=np.asarray(Xi.sum(1)).ravel()>0
    # targets: second-half artists not in the first half, restricted to target vocab
    X2=(Xall[fi]-X1[fi]).tocsr(); X2.data=np.maximum(X2.data,0); X2.eliminate_zeros()
    Xt=X2[:,vt].tocsr(); ok=(np.asarray(Xt.sum(1)).ravel()>0)&has
    aug=None
    if aug_n:   # extra examples: random 50/50 split of whole-day artist sets (training material only)
        rng=np.random.RandomState(seed+10); A=Xall[fi].tocsr(); rows=[];colsA=[];rows2=[];colsB=[]
        for rep in range(aug_n):
            msk=sp.csr_matrix(rng.rand(len(A.data))<0.5)
            keep=np.where(rng.rand(len(A.data))<0.5,1.0,0.0)
            Ain=A.copy(); Ain.data=keep.astype(np.float32); Ain=Ain.multiply(A).tocsr(); Ain.eliminate_zeros()
            Atg=A-Ain; Atg.eliminate_zeros()
            rows.append(Ain[:,vin]); rows2.append(Atg[:,vt])
        augm=(sp.vstack(rows).tocsr(),sp.vstack(rows2).tocsr()); okk=(np.asarray(augm[0].sum(1)).ravel()>0)&(np.asarray(augm[1].sum(1)).ravel()>0)
        aug=(augm[0][okk],augm[1][okk])
    net=nn.train_net(Xi[ok],Xt[ok],wvec,len(vt),epochs=epochs,d=d,drop=drop,seed=seed,aug=aug,verbose=verbose)
    # score fold prefixes
    Pidx=[D.sid[x] for l in R.pre for x in l]
    Xp=X1[Pidx][:,vin].tocsr(); lp=nn.predict(net,Xp,wvec)
    res={}
    for ri,(idx,r) in enumerate(R.iterrows()):
        M=np.zeros((6,6))
        for k,c in enumerate(r.cand):
            ts=[tmap[a] for a in D.cc[c]]
            for j in range(6): M[j,k]=np.mean([lp[ri*6+j,t] if t>=0 else -20 for t in ts])
        res[idx]=M
    return res
if __name__=='__main__':
    t0=time.time()
    for cfg in [dict(epochs=25,d=256,drop=0.5),dict(epochs=25,d=256,drop=0.5,aug_n=2)]:
        res=run_fold(0,verbose=False,**cfg); ids=list(res)
        y=np.array([tr.y[i] for i in ids])
        print(cfg,'hung',round(score(np.array([hungarian(res[i]) for i in ids]),y),4),'argmax',round(score(np.array([res[i].argmax(1) for i in ids]),y),4),time.time()-t0,flush=True)
    # kNN reference on same fold
    FE,names,_=pickle.load(open('/tmp/FE9.pkl','rb')); ix={n:i for i,n in enumerate(names)}
    sel=np.where(fold==0)[0]
    ref=[np.log(FE[i][...,ix['a1_0']]*.5+FE[i][...,ix['a1_1']]*.5+.01) for i in sel]
    print('kNN same fold hung',score(np.array([hungarian(m) for m in ref]),Y[sel]))
