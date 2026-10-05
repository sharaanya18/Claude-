import numpy as np, time, scipy.sparse as sp
from feats import *
L,C,tr,te=load(); D=Data(L,C,set(s for l in te.pre for s in l))
fold=np.random.RandomState(0).randint(0,5,len(tr)); Y=np.array(list(tr.y))
X1=D.X1.tocsr(); Xa=D.Xall.tocsr(); N2=(Xa-X1); N2.data=np.maximum(N2.data,0); N2.eliminate_zeros(); N2=N2.tocsr()   # new-in-second-half
res={}; t0=time.time()
for f in range(5):
    R=tr[fold==f]; held=np.zeros(D.NS,bool)
    for l in R.pre:
        for x in l: held[D.sid[x]]=True
    fi=np.where(~D.is_test&~held)[0]
    cols=sorted({a for c in R.cand for ci in c for a in D.cc[ci]}); cm={a:i for i,a in enumerate(cols)}
    Pidx=[D.sid[x] for l in R.pre for x in l]; Pm=X1[Pidx].tocsr(); A=np.unique(Pm.indices); Pa=Pm[:,A].tocsr()
    Pa=sp.diags(1/np.asarray(Pa.sum(1)).ravel())@Pa
    for name,(Src,Tgt) in {'drift first->new2nd':(X1[fi],N2[fi]),'whole-day cooc':(Xa[fi],Xa[fi])}.items():
        Dm=(Src[:,A].T@Tgt).tocsr()                          # |A| x NA
        rowsum=np.asarray(Dm.sum(1)).ravel(); colpop=np.asarray(Tgt.sum(0)).ravel(); pb=(colpop+0.5)/(colpop.sum()+0.5*len(colpop))
        Dc=Dm[:,cols].tocoo()
        for beta in (1.0,10.0):
            # mixture P(b|prefix)=sum_a w_a (D[a,b]+beta*p(b))/(rowsum_a+beta); score = log P - log p(b) (lift)
            base=(beta/(rowsum+beta))                          # coefficient of p(b)
            Mv=sp.csr_matrix((Dc.data/(rowsum[Dc.row]+beta),(Dc.row,Dc.col)),shape=Dc.shape)
            S=(Pa@Mv).toarray()+np.asarray(Pa@base)[:,None]*pb[cols][None,:]
            S=np.log(S)-np.log(pb[cols]+1e-12)[None,:]
            for ri,(idx,r) in enumerate(R.iterrows()):
                M=np.zeros((6,6))
                for j in range(6):
                    for k,ci in enumerate(r.cand): M[j,k]=np.mean([S[ri*6+j,cm[a]] for a in D.cc[ci]])
                res.setdefault((name,beta),{})[idx]=M
    print(f,round(time.time()-t0),flush=True)
for k,dd in res.items():
    print(k, round(score(np.array([hungarian(dd[i]) for i in tr.index]),Y),4))
