import numpy as np, pandas as pd
exec(open('exp1.py').read().split("rows=tr.copy()")[0])
# --- signal diagnostics on train rows
tmap={}
for _,r in tr.iterrows():
    for s,m in zip(r.pre,r.y): tmap[s]=r.cand[m]
rel1=L[L.half==1].groupby('session').release.apply(lambda x:set(x.dropna()))
rec1=L[L.half==1].groupby('session').recording.apply(set)
crel=C.groupby('continuation').release.apply(lambda x:set(x.dropna())); crec=C.groupby('continuation').recording.apply(set)
hit_rel=np.mean([len(rel1[s]&crel[c])>0 for s,c in tmap.items()])
hit_rec=np.mean([len(rec1[s]&crec[c])>0 for s,c in tmap.items()])
print('true cont release in own prefix',hit_rel,' recording',hit_rec)
# wrong pairs within the same row
wr=[];wc=[]
for _,r in tr.iterrows():
    for j,s in enumerate(r.pre):
        for k,c in enumerate(r.cand):
            if k!=r.y[j]: wr.append(len(rel1[s]&crel[c])>0); wc.append(len(rec1[s]&crec[c])>0)
print('wrong pair release overlap',np.mean(wr),'recording',np.mean(wc))
# popularity: number of first-half sessions of the continuation artist
pop=np.asarray(X1.sum(0)).ravel()
tp=[];wp=[]
for _,r in tr.iterrows():
    pc=[np.mean([np.log1p(pop[a]) for a in cc[c]]) for c in r.cand]
    tp+= [pc[k] for k in r.y]; wp+=[np.mean(pc)]*6
print('log-pop true cont vs row mean',np.mean(tp),np.mean(wp))
# rank-1 vs rank-2 distance from end of first half? last-played artists of prefix vs cont artist: is it in recent artist neighbours
print('prefix sizes in train rows: row-level spread (min/max per row) median', np.median([max(len(L[L.session==s]) for s in [])] if False else 0))
