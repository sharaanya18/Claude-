import numpy as np, time, pickle
from feats import *
L,C,tr,te=load(); D=Data(L,C,set(s for l in te.pre for s in l))
rows=tr; fold=np.random.RandomState(0).randint(0,5,len(rows)); rng=np.random.RandomState(1)
t0=time.time(); FE=np.zeros((len(rows),6,6,12),np.float32)
for f in range(5):
    R=rows[fold==f]; held=np.zeros(D.NS,bool)
    for l in R.pre:
        for x in l: held[D.sid[x]]=True
    fit_idx=np.where(~D.is_test&~held)[0]; bg=rng.choice(fit_idx[D.n1[fit_idx]>=8],2000,replace=False)
    F,names,_=row_features(D,R,fit_idx,bg); FE[fold==f]=F; print(f,time.time()-t0,flush=True)
pickle.dump((FE,names,fold),open('/tmp/FE9.pkl','wb')); print(names)
