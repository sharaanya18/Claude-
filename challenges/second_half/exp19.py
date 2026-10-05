import numpy as np, pickle, lightgbm as lgb
from feats import *
exec(open('exp14.py').read().split("run(TF,'traits only')")[0])
D=Data(L,C,set(s for l in te.pre for s in l))
tes=D.is_test
# prefix-level scalars
ps={}
for _,r in tr.iterrows():
    for x in r.pre:
        a,b=D.s==D.sid[x],None
for x in set(s for l in tr.pre for s in l):
    idx=np.where((D.s==D.sid[x])&(D.h==1))[0]; sec=L.seconds.values[idx]; art=D.a[idx]
    full=np.where(D.s==D.sid[x])[0]
    gaps=np.diff(sec) if len(sec)>1 else np.array([0])
    ps[x]=[np.log(len(idx)),np.log(len(full)),len(set(art))/len(idx),np.log1p(sec[-1]),np.log1p(np.median(gaps)),np.log1p(gaps.max()),(gaps>3600).sum(),np.log1p(np.mean(gaps))]
# cand scalars: artist df (first halves, non-test), rec/rel counts
c1=np.asarray(D.X1[~tes].sum(0)).ravel()
rec_c=np.bincount(D.rec[~tes[D.s]]) if False else None
cs={}
for c,arts in D.cc.items(): cs[c]=[np.log1p(c1[a]) for a in arts]
Cd=C.sort_values(['continuation','rank']); 
pairs=[]; 
Xe=np.zeros((N,6,6,8+4),np.float32)
for i,(_,r) in enumerate(tr.iterrows()):
    for j,x in enumerate(r.pre):
        for k,c in enumerate(r.cand):
            Xe[i,j,k,:8]=ps[x]; Xe[i,j,k,8:10]=cs[c]; Xe[i,j,k,10]=ps[x][0]-np.mean(cs[c]); Xe[i,j,k,11]=0
base=np.concatenate([X,TF],-1)
S0,_=run(base,'base')
S1,_=run(np.concatenate([base,Xe],-1),'base + prefix scalars x cand pop')
