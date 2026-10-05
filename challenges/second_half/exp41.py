import numpy as np, time, sys
from gensim.models import Word2Vec
from feats import *
L,C,tr,te=load(); D=Data(L,C,set(s for l in te.pre for s in l))
fold=np.random.RandomState(0).randint(0,5,len(tr)); Y=np.array(list(tr.y))
starts=np.r_[0,np.where(D.s[1:]!=D.s[:-1])[0]+1,len(D.s)]
seqs={D.s[starts[i]]:[str(x) for x in D.a[starts[i]:starts[i+1]]] for i in range(len(starts)-1)}
pre_arts={}
for l in list(tr.pre):
    for x in l:
        s=D.sid[x]; pre_arts[s]=[str(v) for v in D.a[(D.s==s)&(D.h==1)]]
def dedup(s):
    out=[]
    for x in s:
        if not out or out[-1]!=x: out.append(x)
    return out
base=dict(window=20,vector_size=128,epochs=15,sample=1e-4,negative=10,ns_exponent=0.75,min_count=2,dedup=True)
grid={'base':{},'ep30':dict(epochs=30),'d256':dict(vector_size=256),'w10':dict(window=10),'nodedup':dict(dedup=False),
      'sample1e-3':dict(sample=1e-3),'neg20_ns.5':dict(negative=20,ns_exponent=0.5),'mc1':dict(min_count=1)}
for name,over in grid.items():
    cfg=dict(base); cfg.update(over); res={}; res2={}; res3={}; t0=time.time()
    for f in range(5):
        R=tr[fold==f]; held=np.zeros(D.NS,bool)
        for l in R.pre:
            for x in l: held[D.sid[x]]=True
        fi=np.where(~D.is_test&~held)[0]
        sents=[dedup(seqs[s]) if cfg['dedup'] else seqs[s] for s in fi]
        m=Word2Vec(sents,sg=1,workers=4,seed=0,window=cfg['window'],vector_size=cfg['vector_size'],epochs=cfg['epochs'],sample=cfg['sample'],
                   negative=cfg['negative'],ns_exponent=cfg['ns_exponent'],min_count=cfg['min_count'])
        wv=m.wv; V=wv.vectors/np.linalg.norm(wv.vectors,axis=1,keepdims=True); key=wv.key_to_index
        cnt=np.array([wv.get_vecattr(k,'count') for k in wv.index_to_key],float); idfw=np.log(len(fi)/cnt)
        for idx,r in R.iterrows():
            P=[];P2=[];Ps=[]
            for x in r.pre:
                arts=[key[t] for t in set(pre_arts[D.sid[x]]) if t in key]
                if arts:
                    P.append(V[arts].mean(0)); P2.append((V[arts]*idfw[arts,None]).sum(0)); Ps.append(V[arts])
                else:
                    P.append(np.zeros(V.shape[1])); P2.append(np.zeros(V.shape[1])); Ps.append(np.zeros((1,V.shape[1])))
            P=np.array(P); P/=np.linalg.norm(P,axis=1,keepdims=True)+1e-9; P2=np.array(P2); P2/=np.linalg.norm(P2,axis=1,keepdims=True)+1e-9
            M=np.zeros((6,6)); M2=np.zeros((6,6)); M3=np.zeros((6,6))
            for k,c in enumerate(r.cand):
                cv=[V[key[str(a)]] for a in D.cc[c] if str(a) in key]
                if not cv: continue
                cvm=np.mean(cv,0); M[:,k]=P@cvm; M2[:,k]=P2@cvm
                for j in range(6):
                    sims=Ps[j]@np.array(cv).T; M3[j,k]=np.sort(sims.max(1))[-5:].mean() if False else np.mean(np.sort(sims.ravel())[-5:])
            res[idx]=M; res2[idx]=M2; res3[idx]=M3
    sc=lambda d: round(score(np.array([hungarian(d[i]) for i in tr.index]),Y),4)
    print(f'{name:12s} mean-vec {sc(res)}  idf-vec {sc(res2)}  top5-pair {sc(res3)}  ({round(time.time()-t0)}s)',flush=True)
