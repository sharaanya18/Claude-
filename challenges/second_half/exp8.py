import numpy as np, scipy.sparse as sp, time
from common import *
from sklearn.preprocessing import normalize
exec(open('exp1.py').read().split("rows=tr.copy()")[0])
rows=tr.copy(); fold=np.random.RandomState(0).randint(0,5,len(rows)); Y=np.array(list(rows.y))
def tokmat(col,half=None):
    v=L[col].fillna('NA_'+L.artist) if col=='release' else L[col]
    codes,uni=pd.factorize(v); d=pd.DataFrame({'s':L.s.values,'t':codes,'h':L.half.values})
    if half: d=d[d.h==half]
    d=d.drop_duplicates(['s','t']); n=len(uni)
    return sp.csr_matrix((np.ones(len(d),np.float32),(d.s.values,d.t.values)),shape=(NS,n))
R1,Rall=tokmat('release',1),tokmat('release'); Q1,Qall=tokmat('recording',1),tokmat('recording')
def idfw(X,w): df=np.asarray(X.sum(0)).ravel(); return sp.diags(np.log((NS+1)/(df+1))**w)
mats={'art':(X1,Xall),'rel':(R1,Rall),'rec':(Q1,Qall)}
W={k:idfw(v[1],.5) for k,v in mats.items()}
t0=time.time(); res={}
for f in range(5):
    R=rows[fold==f]; held=np.zeros(NS,bool)
    for l in R.pre:
        for s in l: held[sid[s]]=True
    fi=np.where(~is_test&~held)[0]; Pidx=[sid[s] for l in R.pre for s in l]
    cols=sorted({a for c in R.cand for ci in c for a in cc[ci]}); cm={a:i for i,a in enumerate(cols)}
    Tc=Xall[fi][:,cols].toarray()
    Ks={}
    for k,(P1,Pa) in mats.items():
        Ks[k]=(normalize(P1[Pidx]@W[k])@normalize(Pa[fi]@W[k]).T).toarray()
    combos={'art':Ks['art'],'rel':Ks['rel'],'rec':Ks['rec'],'art+rel':Ks['art']+Ks['rel'],'art+rel+rec':Ks['art']+Ks['rel']+Ks['rec'],'art+.5rel':Ks['art']+.5*Ks['rel']}
    for nm,K in combos.items():
        S=K@Tc
        for ri,(idx,r) in enumerate(R.iterrows()):
            M=np.zeros((6,6))
            for j in range(6):
                for k,ci in enumerate(r.cand): M[j,k]=np.mean([S[ri*6+j,cm[a]] for a in cc[ci]])
            res.setdefault(nm,{})[idx]=M
    print(f,time.time()-t0,flush=True)
for nm,d in res.items():
    pred=[hungarian(np.log(d[i]+.01*np.median(d[i][d[i]>0]) if (d[i]>0).any() else d[i]+1e-9)) for i in rows.index]
    print(nm,round(score(np.array(pred),Y),4))
