import numpy as np, time, scipy.sparse as sp
from feats import *
from sklearn.decomposition import TruncatedSVD
L,C,tr,te=load(); D=Data(L,C,set(s for l in te.pre for s in l))
d=pd.DataFrame({'s':D.s,'a':D.a}); cnt=d.groupby(['s','a']).size().reset_index(name='c')
Ca=sp.csr_matrix((cnt.c.values.astype(np.float32),(cnt.s.values,cnt.a.values)),shape=(D.NS,D.NA)).log1p()
d1=pd.DataFrame({'s':D.s[D.h==1],'a':D.a[D.h==1]}); c1=d1.groupby(['s','a']).size().reset_index(name='c')
C1=sp.csr_matrix((c1.c.values.astype(np.float32),(c1.s.values,c1.a.values)),shape=(D.NS,D.NA)).log1p()
fold=np.random.RandomState(0).randint(0,5,len(tr)); Y=np.array(list(tr.y))
t0=time.time(); res={k:{} for k in ('svd64','svd128','svd256')}
for f in range(5):
    R=tr[fold==f]; held=np.zeros(D.NS,bool)
    for l in R.pre:
        for x in l: held[D.sid[x]]=True
    fi=np.where(~D.is_test&~held)[0]
    cntv=np.asarray((Ca[fi]>0).sum(0)).ravel(); vin=np.where(cntv>=2)[0]
    W=sp.diags((np.log((len(fi)+1)/(cntv[vin]+1))**0.5).astype(np.float32))
    Xf=normalize(Ca[fi][:,vin]@W)
    Pidx=[D.sid[x] for l in R.pre for x in l]; Xp=normalize(C1[Pidx][:,vin]@W)
    amap=-np.ones(D.NA,int); amap[vin]=np.arange(len(vin))
    for k in (64,128,256):
        svd=TruncatedSVD(k,random_state=0,n_iter=4).fit(Xf)
        Vt=svd.components_            # (k, nvin) artist embeddings (columns)
        E=normalize(Vt.T)             # artist embedding unit norm
        Pe=normalize(Xp@Vt.T)         # prefix embedding
        for ri,(idx,r) in enumerate(R.iterrows()):
            M=np.zeros((6,6))
            for kk,c in enumerate(r.cand):
                ids=[amap[a] for a in D.cc[c]]; 
                v=np.mean([E[i] if i>=0 else np.zeros(k) for i in ids],0)
                M[:,kk]=Pe[ri*6:(ri+1)*6]@v
            res[f'svd{k}'][idx]=M
    print(f,time.time()-t0,flush=True)
for nm,dd in res.items():
    print(nm,'hung',round(score(np.array([hungarian(dd[i]) for i in tr.index]),Y),4))
