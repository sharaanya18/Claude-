import numpy as np
from common import *
L,C,tr,te=load()
print('recording nulls',L.recording.isna().mean(),'release nulls',L.release.isna().mean(),'artist nulls',L.artist.isna().mean())
g=L[L.half==1].groupby('session').release.apply(lambda x:x.isna().mean()); gn=g.to_dict()
ce=C.groupby('continuation').release.apply(lambda x:x.isna().sum()).to_dict()
tp=[];wp=[]
for _,r in tr.iterrows():
    for j,s in enumerate(r.pre):
        for k,c in enumerate(r.cand):
            (tp if k==r.y[j] else wp).append((gn[s],ce[c]))
tp=np.array(tp); wp=np.array(wp)
print('corr true pairs (prefix empty frac, cont empties)',np.corrcoef(tp.T)[0,1],' wrong',np.corrcoef(wp.T)[0,1])
for lo,hi in ((0,.001),(.001,.5),(.5,1.01)):
    m=(tp[:,0]>=lo)&(tp[:,0]<hi); w=(wp[:,0]>=lo)&(wp[:,0]<hi)
    print(lo,hi,'n true',m.sum(),'cont empties mean true',tp[m,1].mean(),'wrong',wp[w,1].mean())
# other metadata: release token frequency scale, e.g. number of distinct recordings per release? prefix length vs cand
