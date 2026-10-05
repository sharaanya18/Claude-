import numpy as np, time, itertools, torch
from gensim.models import Word2Vec
from feats import *
torch.set_num_threads(4)
L,C,tr,te=load(); D=Data(L,C,set(s for l in te.pre for s in l))
d=np.load('/tmp/stack_cache_v7.npz'); X0=d['X']; ptr=d['ptr']; fold=d['fold']; Y=d['Y']; N=len(Y)
starts=np.r_[0,np.where(D.s[1:]!=D.s[:-1])[0]+1,len(D.s)]
seqs={D.s[starts[i]]:[str(x) for x in D.a[starts[i]:starts[i+1]]] for i in range(len(starts)-1)}
def dedup(s):
    out=[]
    for x in s:
        if not out or out[-1]!=x: out.append(x)
    return out
pre_arts={D.sid[x]:[str(v) for v in D.a[(D.s==D.sid[x])&(D.h==1)]] for l in tr.pre for x in l}
CFGS={'w20d256':dict(window=20,vector_size=256),'w5d64':dict(window=5,vector_size=64)}
def rw(v):
    o=[]
    for ax in (0,1):
        m=v.mean(ax,keepdims=True); s=v.std(ax,keepdims=True)+1e-6; o+=[(v-m)/s, np.argsort(np.argsort(v,ax),ax).astype(float)]
    return o
E=np.zeros((N,6,6,2*12),np.float32); t0=time.time()
for f in range(5):
    vai=np.where(fold==f)[0]; held=np.zeros(D.NS,bool)
    for i in vai:
        for x in tr.pre.iloc[i]: held[D.sid[x]]=True
    fi=np.where(~D.is_test&~held)[0]; sents=[dedup(seqs[s]) for s in fi]
    for ci,(name,kw) in enumerate(CFGS.items()):
        m=Word2Vec(sents,sg=1,workers=4,seed=0,epochs=15,sample=1e-4,negative=10,min_count=2,**kw)
        wv=m.wv; V=wv.vectors/np.linalg.norm(wv.vectors,axis=1,keepdims=True); key=wv.key_to_index
        for i in vai:
            r=tr.iloc[i]; P=[]; Ps=[]
            for x in r.pre:
                ids=[key[t] for t in set(pre_arts[D.sid[x]]) if t in key]
                P.append(V[ids].mean(0) if ids else np.zeros(V.shape[1])); Ps.append(V[ids] if ids else np.zeros((1,V.shape[1])))
            P=np.array(P); P/=np.linalg.norm(P,axis=1,keepdims=True)+1e-9
            cm=np.zeros((6,6)); c1=np.zeros((6,6)); c2=np.zeros((6,6)); mx=np.zeros((6,6))
            for k,c in enumerate(r.cand):
                vs=[V[key[str(a)]] if str(a) in key else np.zeros(V.shape[1]) for a in D.cc[c]]
                cm[:,k]=P@((vs[0]+vs[1])/2); c1[:,k]=P@vs[0]; c2[:,k]=P@vs[1]
                for j in range(6): mx[j,k]=(Ps[j]@np.array(vs).T).max()
            feats=[cm,c1,c2,mx]+rw(cm)+rw(mx)
            E[i,:,:,ci*12:(ci+1)*12]=np.stack(feats,-1)
    print('fold',f,round(time.time()-t0),flush=True)
np.save('/tmp/emb_feats.npy',E)
from common import hungarian, score
for ci,name in enumerate(CFGS): print(name,'cos alone',round(score(np.array([hungarian(E[i,:,:,ci*12]) for i in range(N)]),Y),4))
PERMS=torch.tensor(list(itertools.permutations(range(6)))); ar=torch.arange(6)
def mlp_oof(X,steps=250):
    F=X.shape[-1]; oof=np.zeros((N,6,6))
    for f in range(5):
        tri=np.where(fold!=f)[0]; vai=np.where(fold==f)[0]
        mu=X[tri].reshape(-1,F).mean(0); sd=X[tri].reshape(-1,F).std(0)+1e-6
        Xt=torch.tensor((X[tri]-mu)/sd).float(); Xv=torch.tensor((X[vai]-mu)/sd).float(); idx=torch.tensor(ptr[tri]); ps=[]
        for s in (0,1,2):
            torch.manual_seed(s); net=torch.nn.Sequential(torch.nn.Dropout(0.2),torch.nn.Linear(F,32),torch.nn.GELU(),torch.nn.Linear(32,1))
            opt=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=1e-2)
            for _ in range(steps):
                net.train(); opt.zero_grad(); ll=net(Xt).squeeze(-1)[:,ar[None,:],PERMS].sum(-1)
                loss=-(ll.gather(1,idx[:,None]).squeeze(1)-torch.logsumexp(ll,1)).mean(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): ps.append(net(Xv).squeeze(-1).numpy())
        oof[vai]=np.mean(ps,0)
    return oof
for name,X in (('v7 stack',X0),('v7 + embeddings',np.concatenate([X0,E],-1)),('embeddings only',E)):
    o=mlp_oof(X.astype(np.float32)); print(name,'MLP OOF',round(score(np.array([hungarian(z) for z in o]),Y),4),flush=True)
