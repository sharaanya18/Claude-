import numpy as np, time, scipy.sparse as sp
from feats import *
L,C,tr,te=load(); D=Data(L,C,set(s for l in te.pre for s in l))
fold=np.random.RandomState(0).randint(0,5,len(tr)); Y=np.array(list(tr.y))
Xb=D.Xall.tocsr(); a_arr=D.a; s_arr=D.s; h_arr=D.h
starts=np.r_[0,np.where(s_arr[1:]!=s_arr[:-1])[0]+1,len(s_arr)]; span={s_arr[starts[i]]:(starts[i],starts[i+1]) for i in range(len(starts)-1)}
def wbag(sess_list,mode):
    rr,cc,vv=[],[],[]
    for pi,s in enumerate(sess_list):
        a,b=span[s]; k=a+int((h_arr[a:b]==1).sum()); n=k-a; pos=np.arange(n)
        if mode=='uniform': w=np.ones(n)
        elif mode=='linear': w=(pos+1)/n
        elif mode.startswith('exp'): w=float(mode[3:])**(n-1-pos)
        elif mode=='lasthalf': w=(pos>=n//2).astype(float)
        elif mode=='last10': w=(pos>=n-10).astype(float)
        rr+=[pi]*n; cc+=list(a_arr[a:k]); vv+=list(w)
    M=sp.csr_matrix((vv,(rr,cc)),shape=(len(sess_list),D.NA))   # duplicates summed
    M.data=np.minimum(M.data,1.0) if False else M.data
    return M
modes=['uniform','linear','exp0.97','exp0.93','exp0.85','lasthalf','last10']
res={m:{} for m in modes}; t0=time.time()
for f in range(5):
    R=tr[fold==f]; held=np.zeros(D.NS,bool)
    for l in R.pre:
        for x in l: held[D.sid[x]]=True
    fi=np.where(~D.is_test&~held)[0]; Xf=Xb[fi]; df=np.asarray(Xf.sum(0)).ravel()
    cols=sorted({a for c in R.cand for ci in c for a in D.cc[ci]}); cm={a:i for i,a in enumerate(cols)}
    Pidx=[D.sid[x] for l in R.pre for x in l]
    A=np.unique(D.X1[Pidx].indices)
    G=(Xf[:,A].T@Xf[:,cols].tocsc()).tocoo(); sim=G.data/((df[A][G.row]**.5)*(df[cols][G.col]**.5)+5)
    Sm=sp.csr_matrix((sim,(G.row,G.col)),shape=G.shape)
    for m in modes:
        Pw=wbag(Pidx,m)[:,A].tocsr()
        # per artist: max over its plays' weights would ignore repeats; use sum of weights normalised by total weight
        Pw=sp.diags(1/np.maximum(np.asarray(Pw.sum(1)).ravel(),1e-9))@Pw
        S=(Pw@Sm).toarray()
        for ri,(idx,r) in enumerate(R.iterrows()):
            M=np.zeros((6,6))
            for j in range(6):
                for k,ci in enumerate(r.cand): M[j,k]=np.mean([S[ri*6+j,cm[a]] for a in D.cc[ci]])
            res[m][idx]=M
    print(f,round(time.time()-t0),flush=True)
for m,dd in res.items():
    allv=np.concatenate([dd[i].ravel() for i in tr.index]); s0=np.median(allv[allv>0])
    print(f'{m:10s}',{e:round(score(np.array([hungarian(np.log(dd[i]+e*s0)) for i in tr.index]),Y),4) for e in (.1,.3,1)})
