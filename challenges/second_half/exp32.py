import sys; sys.argv=['x','dataset/public','working/x.csv']
import numpy as np, pandas as pd, json, scipy.sparse as sp, time
import solution_commented as S
from common import hungarian, score
L=pd.read_csv('dataset/public/listens.csv').sort_values(['session','relative_position']).reset_index(drop=True)
C=pd.read_csv('dataset/public/continuations.csv'); tr=pd.read_csv('dataset/public/train.csv'); te=pd.read_csv('dataset/public/test.csv')
for d in (tr,te): d['pre']=d.prefixes.map(json.loads); d['cand']=d.candidates.map(json.loads)
Y=tr[[f'match_{i}' for i in range(1,7)]].to_numpy()-1
D=S.Data(L,C,set(x for l in te.pre for x in l)); fit=~D.is_test
W={1:1.0,2:0.5,3:0.25}
def trans(tok,ntok):
    A=None; ok=fit[D.s]
    for d,w in W.items():
        m=(D.s[:-d]==D.s[d:])&ok[:-d]&(tok[:-d]!=tok[d:])
        M=sp.csr_matrix((np.full(m.sum(),w,np.float32),(tok[:-d][m],tok[d:][m])),shape=(ntok,ntok)); A=M if A is None else A+M
    return A.tocsr()
t0=time.time()
Aa=trans(D.a,D.NA); Ar=trans(D.rel_codes,len(D.rel_uni)); Aq=trans(D.rec_codes,len(D.rec_uni))
outa=np.asarray(Aa.sum(1)).ravel(); outr=np.asarray(Ar.sum(1)).ravel(); outq=np.asarray(Aq.sum(1)).ravel()
print('built',time.time()-t0,flush=True)
def own_pairs(sess,tok):
    cnt={}
    for s in sess:
        a,b=D.span[s]; t=tok[a:b]
        for d,w in W.items():
            for i in range(len(t)-d):
                if t[i]!=t[i+d]: cnt[(t[i],t[i+d])]=cnt.get((t[i],t[i+d]),0)+w
    return cnt
LASTK=5; DEC=0.6
F=np.zeros((len(tr),6,6,8),np.float32)
for ri,(_,r) in enumerate(tr.iterrows()):
    sess=[D.sid[x] for x in r.pre]
    own={'a':own_pairs(sess,D.a),'r':own_pairs(sess,D.rel_codes),'q':own_pairs(sess,D.rec_codes)}
    for j,s in enumerate(sess):
        a0,b0=D.span[s]; k=a0+int((D.h[a0:b0]==1).sum())
        lastA=D.a[k-LASTK:k][::-1]; lastR=D.rel_codes[k-LASTK:k][::-1]; lastQ=D.rec_codes[k-LASTK:k][::-1]
        for kk,c in enumerate(r.cand):
            b1,b2=D.cc[c]; rr=D.crel[c]; qq=D.crec[c]
            def sc(A,out,last,target,ow,first_only=False):
                tot=0.0
                for i,a in enumerate(last[:1] if first_only else last):
                    if target<0: continue
                    v=A[a,target]-ow.get((a,target),0); o=out[a]-sum(w for (x,y),w in ow.items() if x==a)
                    if o>0: tot+=(DEC**i)*max(v,0)/o
                return tot
            F[ri,j,kk,0]=sc(Aa,outa,lastA,b1,own['a'],True); F[ri,j,kk,1]=sc(Aa,outa,lastA,b1,own['a'])
            F[ri,j,kk,2]=sc(Aa,outa,lastA,b2,own['a']); F[ri,j,kk,3]=sc(Ar,outr,lastR,rr[0],own['r'],True)
            F[ri,j,kk,4]=sc(Ar,outr,lastR,rr[0],own['r']); F[ri,j,kk,5]=sc(Aq,outq,lastQ,qq[0],own['q'],True)
            F[ri,j,kk,6]=sc(Aq,outq,lastQ,qq[0],own['q']); F[ri,j,kk,7]=sc(Ar,outr,lastR,rr[1] if len(rr)>1 else -1,own['r'])
    if ri%200==0: print(ri,time.time()-t0,flush=True)
np.save('/tmp/next_feats.npy',F)
names=['art_last1->b1','art_last5->b1','art_last5->b2','rel_last1->r1','rel_last5->r1','rec_last1->q1','rec_last5->q1','rel_last5->r2']
tm=np.zeros((len(tr),6,6),bool)
for i,y in enumerate(Y): tm[i,np.arange(6),y]=True
for f,n in enumerate(names):
    v=F[...,f]; print(f'{n:16s} nonzero true {(v[tm]>0).mean():.3f} wrong {(v[~tm]>0).mean():.3f}  alone-hung {score(np.array([hungarian(np.log(m+1e-4)) for m in v]),Y):.4f}')
