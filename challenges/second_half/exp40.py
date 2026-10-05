import numpy as np, time
from gensim.models import Word2Vec
from feats import *
L,C,tr,te=load(); D=Data(L,C,set(s for l in te.pre for s in l))
fold=np.random.RandomState(0).randint(0,5,len(tr)); Y=np.array(list(tr.y))
starts=np.r_[0,np.where(D.s[1:]!=D.s[:-1])[0]+1,len(D.s)]
seqs={D.s[starts[i]]:[str(x) for x in D.a[starts[i]:starts[i+1]]] for i in range(len(starts)-1)}
def dedup(s):   # collapse consecutive repeats (album runs) so windows span different artists
    out=[]; 
    for x in s:
        if not out or out[-1]!=x: out.append(x)
    return out
cfgs={'w5_d64':dict(window=5,vector_size=64),'w20_d128':dict(window=20,vector_size=128),'w50_d128_sg':dict(window=50,vector_size=128)}
res={k:{} for k in cfgs}; t0=time.time()
for f in range(5):
    R=tr[fold==f]; held=np.zeros(D.NS,bool)
    for l in R.pre:
        for x in l: held[D.sid[x]]=True
    fi=np.where(~D.is_test&~held)[0]
    sents=[dedup(seqs[s]) for s in fi]
    for name,kw in cfgs.items():
        m=Word2Vec(sents,sg=1,negative=10,min_count=2,epochs=15,workers=4,seed=0,sample=1e-4,ns_exponent=0.75,**kw)
        wv=m.wv; V=wv.vectors/np.linalg.norm(wv.vectors,axis=1,keepdims=True); key=wv.key_to_index
        for idx,r in R.iterrows():
            P=[]
            for x in r.pre:
                s=D.sid[x]; a,b=D.span[s] if hasattr(D,'span') else (None,None)
                arts=[str(v) for v in D.a[(D.s==s)&(D.h==1)]]
                vs=[V[key[t]] for t in set(arts) if t in key]
                P.append(np.mean(vs,0) if vs else np.zeros(V.shape[1]))
            P=np.array(P); P/=np.linalg.norm(P,axis=1,keepdims=True)+1e-9
            M=np.zeros((6,6))
            for k,c in enumerate(r.cand):
                cv=[V[key[str(a)]] for a in D.cc[c] if str(a) in key]
                M[:,k]=P@np.mean(cv,0) if cv else 0
            res[name][idx]=M
    print(f,round(time.time()-t0),flush=True)
for k,dd in res.items(): print(k,'hung',round(score(np.array([hungarian(dd[i]) for i in tr.index]),Y),4))
