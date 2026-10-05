import numpy as np, pandas as pd, scipy.sparse as sp, time, pickle
from common import *
from sklearn.preprocessing import normalize
src=open('exp1.py').read().split("rows=tr.copy()")[0]
exec(src)
rows=tr.copy(); fold=np.random.RandomState(0).randint(0,5,len(rows))
Mall={}
for f in range(5):
    R=rows[fold==f]
    held=np.zeros(NS,bool)
    for l in R.pre:
        for s in l: held[sid[s]]=True
    fi=np.where(~is_test&~held)[0]
    Pidx=[sid[s] for l in R.pre for s in l]
    cols=sorted({a for c in R.cand for ci in c for a in cc[ci]}); cm={a:i for i,a in enumerate(cols)}
    W=sp.diags(idf**0.5); Pm=normalize(X1[Pidx]@W); Fm=normalize(Xall[fi]@W)
    K=(Pm@Fm.T).toarray(); S=K@Xall[fi][:,cols].toarray()
    for ri,(idx,r) in enumerate(R.iterrows()):
        M=np.zeros((6,6,2))
        for j in range(6):
            for k,ci in enumerate(r.cand):
                M[j,k]=[S[ri*6+j,cm[a]] for a in cc[ci]]
        Mall[idx]=M
pickle.dump(Mall,open('/tmp/Mall.pkl','wb'))
Y=np.array(list(rows.y)); 
def ev(fn):
    pred=[]
    for i in rows.index: pred.append(hungarian(fn(Mall[i])))
    return score(np.array(pred),Y)
m=lambda M:M.mean(-1)
print('raw',ev(m))
for c in (1e-3,1e-2,1e-1,1): print('log',c,ev(lambda M:np.log(m(M)+c)))
print('colz',ev(lambda M:(m(M)-m(M).mean(0))/(m(M).std(0)+1e-9)))
print('colz log.01',ev(lambda M:(lambda L:(L-L.mean(0))/(L.std(0)+1e-9))(np.log(m(M)+1e-2))))
print('rank col',ev(lambda M:np.argsort(np.argsort(m(M),0),0)+np.argsort(np.argsort(m(M),1),1)))
print('sqrt',ev(lambda M:np.sqrt(m(M))))
print('max of two',ev(lambda M:np.log(M.max(-1)+1e-2)), 'min', ev(lambda M:np.log(M.min(-1)+1e-2)))
