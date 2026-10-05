import numpy as np
from common import *
L,C,tr,te=load()
tes=set(s for l in te.pre for s in l)
Lf=L[~L.session.isin(tes)]
rec_cnt=Lf.recording.value_counts(); art_cnt=Lf.artist.value_counts(); rel_cnt=Lf.release.value_counts()
L['lrec']=np.log1p(L.recording.map(rec_cnt).fillna(0)); L['lart']=np.log1p(L.artist.map(art_cnt).fillna(0)); L['lrel']=np.log1p(L.release.map(rel_cnt).fillna(0))
L['hit']=L.lrec-L.lart; L['emp']=L.release.isna().astype(float)
C['lrec']=np.log1p(C.recording.map(rec_cnt).fillna(0)); C['lart']=np.log1p(C.artist.map(art_cnt).fillna(0)); C['lrel']=np.log1p(C.release.map(rel_cnt).fillna(0))
C['hit']=C.lrec-C.lart; C['emp']=C.release.isna().astype(float)
tr_feats=['lrec','lrel','lart','hit','emp']
P=L[L.half==1].groupby('session')[tr_feats].mean(); Pm=L[L.half==1].groupby('session')[tr_feats].median()
Cf=C.groupby('continuation')[tr_feats].mean()
for ft in tr_feats:
    tp=[];wp=[]
    for _,r in tr.iterrows():
        for j,s in enumerate(r.pre):
            for k,c in enumerate(r.cand):
                (tp if k==r.y[j] else wp).append((P.loc[s,ft],Cf.loc[c,ft]))
    tp=np.array(tp);wp=np.array(wp)
    print(ft,'corr true',round(np.corrcoef(tp.T)[0,1],3),'wrong',round(np.corrcoef(wp.T)[0,1],3))
