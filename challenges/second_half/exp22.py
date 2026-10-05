import numpy as np, scipy.sparse as sp, json, pandas as pd, time
from solution import Data
L=pd.read_csv('dataset/public/listens.csv').sort_values(['session','relative_position']).reset_index(drop=True)
C=pd.read_csv('dataset/public/continuations.csv'); tr=pd.read_csv('dataset/public/train.csv'); te=pd.read_csv('dataset/public/test.csv')
for d in (tr,te): d['pre']=d.prefixes.map(json.loads); d['cand']=d.candidates.map(json.loads)
D=Data(L,C,set(s for l in te.pre for s in l)); fit=~D.is_test
rel=D.rel_codes; NR=len(D.rel_uni)
d=pd.DataFrame({'s':D.s,'a':D.a,'r':rel}); d=d[fit[d.s]].drop_duplicates(['s','a','r'])
AR=sp.csr_matrix((np.ones(len(d),np.float32),(d.a.values,d.r.values)),shape=(D.NA,NR))   # artist x release, distinct-session counts
# recording-level too: artist x recording
dr=pd.DataFrame({'s':D.s,'a':D.a,'r':D.rec_codes}); dr=dr[fit[dr.s]].drop_duplicates(['s','a','r'])
AQ=sp.csr_matrix((np.ones(len(dr),np.float32),(dr.a.values,dr.r.values)),shape=(D.NA,len(D.rec_uni)))
def own_mat(sess,codes,ncol):
    m=np.isin(D.s,sess); dd=pd.DataFrame({'a':D.a[m],'r':codes[m],'s':D.s[m]}).drop_duplicates()
    return sp.csr_matrix((np.ones(len(dd),np.float32),(dd.a.values,dd.r.values)),shape=(D.NA,ncol))
t0=time.time(); rows=[]
for i,(_,r) in enumerate(tr.iterrows()):
    sess=[D.sid[x] for x in r.pre]
    OW=own_mat(sess,rel,NR); AR_r=(AR-OW).tocsr(); AR_r.data=np.maximum(AR_r.data,0); AR_r.eliminate_zeros()
    res=np.zeros((6,6,3))
    for j,x in enumerate(sess):
        a0,k=D.prefix_plays(x); arts=np.unique(D.a[a0:k]); rels=np.unique(rel[a0:k])
        Rp=AR_r[arts]     # releases of the prefix's artists (all fit plays)
        Rset=set(np.unique(Rp.indices)); 
        for kk,c in enumerate(r.cand):
            b=D.cc[c]; Rb=[set(AR_r[bb].indices) for bb in b]
            res[j,kk,0]=sum(len(set(rels)&rb)>0 for rb in Rb)                  # prefix's own releases appear with candidate artist elsewhere
            res[j,kk,1]=sum(len(Rset&rb)>0 for rb in Rb)                       # any prefix artist shares a release with candidate artist
            res[j,kk,2]=sum(len(set(rels)&rb) for rb in Rb)
    rows.append(res)
    if i%300==0: print(i,time.time()-t0,flush=True)
R=np.array(rows); Y=tr[[f'match_{i}' for i in range(1,7)]].to_numpy()-1
tm=np.zeros((len(tr),6,6),bool)
for i,y in enumerate(Y): tm[i,np.arange(6),y]=True
for f in range(3):
    print('feature',f,'true>0',(R[...,f][tm]>0).mean(),'wrong>0',(R[...,f][~tm]>0).mean())
np.save('/tmp/R22.npy',R)
