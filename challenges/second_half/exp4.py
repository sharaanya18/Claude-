import numpy as np, pandas as pd, scipy.sparse as sp, time, itertools
from common import *
src=open('exp1.py').read().split("rows=tr.copy()")[0]; exec(src)
# play arrays in order
a_arr=L.a.values; s_arr=L.s.values; h_arr=L.half.values
pos=np.arange(len(L)); 
# index within session
first_idx=np.r_[0,np.where(s_arr[1:]!=s_arr[:-1])[0]+1]; 
rows=tr.copy(); fold=np.random.RandomState(0).randint(0,5,len(rows)); Y=np.array(list(rows.y))
def trans(fit_s, W, decay):
    A=None
    ok_sess=fit_s[s_arr]
    for d in range(1,W+1):
        m=(s_arr[:-d]==s_arr[d:])&ok_sess[:-d]
        r=a_arr[:-d][m]; c=a_arr[d:][m]
        M=sp.csr_matrix((np.full(len(r),decay**(d-1),np.float32),(r,c)),shape=(NA,NA))
        A=M if A is None else A+M
    return A
def prefix_vec(sess, rho, K):
    # weights over last K plays of first half, decay rho per step back
    idx=np.where((s_arr==sess)&(h_arr==1))[0]; idx=idx[-K:]
    n=len(idx); w=rho**np.arange(n-1,-1,-1)
    return idx, w
t0=time.time()
configs=[(W,dec,rho,K,alpha) for W,dec in [(1,1),(3,.7),(10,.8)] for rho,K in [(1,10**6),(.7,10),(.9,30)] for alpha in [0.5,1.0]]
res={c:[] for c in configs}
for f in range(5):
    R=rows[fold==f]
    held=np.zeros(NS,bool)
    for l in R.pre:
        for s in l: held[sid[s]]=True
    fit=~is_test&~held
    cols=sorted({a for c in R.cand for ci in c for a in cc[ci]}); cm={a:i for i,a in enumerate(cols)}
    for W,dec in [(1,1),(3,.7),(10,.8)]:
        A=trans(fit,W,dec)
        Ac=A[:,cols].tocsc(); colsum=np.asarray(A.sum(0)).ravel()[cols]+1e-9
        for rho,K in [(1,10**6),(.7,10),(.9,30)]:
            # prefix vectors matrix
            rr=[];cc_=[];vv=[]
            sess_list=[sid[s] for l in R.pre for s in l]
            for pi,sx in enumerate(sess_list):
                idx,w=prefix_vec(sx,rho,K)
                rr+=[pi]*len(idx); cc_+=list(a_arr[idx]); vv+=list(w)
            Pm=sp.csr_matrix((vv,(rr,cc_)),shape=(len(sess_list),NA))
            Pm=sp.diags(1/np.asarray(Pm.sum(1)).ravel())@Pm
            S0=(Pm@Ac).toarray()
            for alpha in [0.5,1.0]:
                S=S0/(colsum**alpha)
                for ri,(idx,r) in enumerate(R.iterrows()):
                    M=np.zeros((6,6))
                    for j in range(6):
                        for k,ci in enumerate(r.cand):
                            M[j,k]=np.mean([S[ri*6+j,cm[a]] for a in cc[ci]])
                    res[(W,dec,rho,K,alpha)].append((idx,M))
    print('fold',f,time.time()-t0,flush=True)
for c,v in res.items():
    d=dict(v); out={}
    for eps_rel in (1e-2,):
        pred=[]
        for i in rows.index:
            M=d[i]; sc=np.log(M+eps_rel*np.median(M[M>0]) if (M>0).any() else M+1e-9)
            pred.append(hungarian(sc))
        out=score(np.array(pred),Y)
    print(c,round(out,4))
