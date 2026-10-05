import numpy as np, pandas as pd, scipy.sparse as sp, time
from common import *
from sklearn.preprocessing import normalize
T0=time.time()
L,C,tr,te=load()
aid={a:i for i,a in enumerate(L.artist.unique())}; L['a']=L.artist.map(aid); NA=len(aid)
sid={s:i for i,s in enumerate(L.session.unique())}; L['s']=L.session.map(sid); NS=len(sid)
tes=set(s for l in te.pre for s in l); is_test=np.zeros(NS,bool)
for s in tes: is_test[sid[s]]=True
def mat(df): 
    d=df.drop_duplicates(['s','a']); return sp.csr_matrix((np.ones(len(d),np.float32),(d.s.values,d.a.values)),shape=(NS,NA))
Xall=mat(L); X1=mat(L[L.half==1]); X2=mat(L[L.half==2])
# continuation artist ids
cc=C.groupby('continuation').artist.apply(lambda x:[aid[a] for a in x]).to_dict()
df_idf=np.asarray(Xall.sum(0)).ravel(); idf=np.log((NS+1)/(df_idf+1)).astype(np.float32)
rows=tr.copy(); rng=np.random.RandomState(0); fold=rng.randint(0,5,len(rows)); 
res={}
for f in range(5):
    R=rows[fold==f]
    held=np.zeros(NS,bool)
    for l in R.pre: 
        for s in l: held[sid[s]]=True
    fit=~is_test&~held
    fi=np.where(fit)[0]
    Pidx=[sid[s] for l in R.pre for s in l]
    for name,(w,tgt,gam) in {'knn_h2':(0.5,'h2',1),'knn_all':(0.5,'all',1),'knn_h2_g3':(0.5,'h2',3),'knn_all_g3':(0.5,'all',3)}.items():
        # prefix vectors from first half, idf weighted; fit sessions represented by whole day
        W=sp.diags(idf**w)
        Pm=normalize(X1[Pidx]@W); Fm=normalize(Xall[fi]@W)
        K=(Pm@Fm.T).toarray()**gam
        T=(X2 if tgt=='h2' else Xall)[fi]
        cols=sorted({a for c in R.cand for ci in c for a in cc[ci]}); cm={a:i for i,a in enumerate(cols)}
        S=K@T[:,cols].toarray()   # (nP, ncols)
        for ri,(idx,r) in enumerate(R.iterrows()):
            M=np.zeros((6,6))
            for j in range(6):
                for k,ci in enumerate(r.cand):
                    M[j,k]=np.mean([S[ri*6+j,cm[a]] for a in cc[ci]])
            res.setdefault(name,[]).append((M,r.y,f))
    print('fold',f,time.time()-T0,flush=True)
for name,v in res.items():
    am=[score(np.argmax(M,1)[None],y[None]) for M,y,f in v]
    hu=[hungarian(M) for M,y,f in v]; Y=np.array([y for M,y,f in v])
    print(name,'argmax',score(np.array([np.argmax(M,1) for M,y,f in v]),Y),'hung',score(np.array(hu),Y))
