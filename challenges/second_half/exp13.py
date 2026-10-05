import numpy as np, pickle, time
from feats import *
L,C,tr,te=load(); D=Data(L,C,set(s for l in te.pre for s in l))
fold=np.random.RandomState(0).randint(0,5,len(tr)); t0=time.time(); TF=None
for f in range(5):
    R=tr[fold==f]; held=np.zeros(D.NS,bool)
    for l in R.pre:
        for x in l: held[D.sid[x]]=True
    fit_idx=np.where(~D.is_test&~held)[0]
    F,names=trait_features(D,R,fit_idx,C)
    if TF is None: TF=np.zeros((len(tr),)+F.shape[1:],np.float32)
    TF[fold==f]=F; print(f,time.time()-t0,flush=True)
pickle.dump((TF,names),open('/tmp/TF13.pkl','wb'))
