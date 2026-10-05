import numpy as np, pickle, time
from lk import *
import json
L=pd.read_csv('dataset/public/listens.csv').sort_values(['session','relative_position']).reset_index(drop=True)
C=pd.read_csv('dataset/public/continuations.csv'); tr=pd.read_csv('dataset/public/train.csv'); te=pd.read_csv('dataset/public/test.csv')
for d in (tr,te): d['pre']=d.prefixes.map(json.loads); d['cand']=d.candidates.map(json.loads)
D=Data(L,C,set(s for l in te.pre for s in l))
fit_mask=~D.is_test
lk=LK(D,fit_mask); t0=time.time()
FEATS=[]
for i,(_,r) in enumerate(tr.iterrows()):
    pre=[D.sid[x] for x in r.pre]; arts=[a for c in r.cand for a in D.cc[c]]
    f=lk.row(pre,arts,np.array(pre)); FEATS.append(f.reshape(6,6,2,-1))
    if i%200==0: print(i,time.time()-t0,flush=True)
FEATS=np.array(FEATS); print(FEATS.shape, FEATS.sum((0,1,2,3)).reshape(3,14,3).sum(-1)[:, :])
pickle.dump(FEATS,open('/tmp/LK20.pkl','wb'))
