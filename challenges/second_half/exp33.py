import numpy as np, torch, itertools, time
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from common import hungarian, score
torch.set_num_threads(4)
d=np.load('/tmp/stack_cache_v7.npz'); X0=d['X']; ptr=d['ptr']; fold=d['fold']; Y=d['Y']; N=len(Y)
NF=np.log1p(np.load('/tmp/next_feats.npy')*100)
def rw(v):
    o=[]
    for ax in (1,2):
        m=v.mean(ax,keepdims=True); s=v.std(ax,keepdims=True)+1e-6; o.append((v-m)/s)
    return o
extra=[NF[...,i] for i in range(NF.shape[-1])]
for i in (1,2,4):
    extra+=rw(NF[...,i])
X1=np.concatenate([X0,np.stack(extra,-1)],-1).astype(np.float32)
PERMS=np.array(list(itertools.permutations(range(6)))); P_t=torch.tensor(PERMS); ar=torch.arange(6)
def mlp_oof(X,hid=32,steps=250,seeds=(0,1,2),wd=1e-2,drop=0.2):
    F=X.shape[-1]; oof=np.zeros((N,6,6))
    for f in range(5):
        tri=np.where(fold!=f)[0]; vai=np.where(fold==f)[0]
        mu=X[tri].reshape(-1,F).mean(0); sd=X[tri].reshape(-1,F).std(0)+1e-6
        Xt=torch.tensor((X[tri]-mu)/sd).float(); Xv=torch.tensor((X[vai]-mu)/sd).float(); idx=torch.tensor(ptr[tri]); ps=[]
        for s in seeds:
            torch.manual_seed(s)
            net=torch.nn.Sequential(torch.nn.Dropout(drop),torch.nn.Linear(F,hid),torch.nn.GELU(),torch.nn.Linear(hid,1)) if hid else torch.nn.Sequential(torch.nn.Dropout(drop),torch.nn.Linear(F,1))
            opt=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=wd)
            for _ in range(steps):
                net.train(); opt.zero_grad(); ll=net(Xt).squeeze(-1)[:,ar[None,:],P_t].sum(-1)
                loss=-(ll.gather(1,idx[:,None]).squeeze(1)-torch.logsumexp(ll,1)).mean(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): ps.append(net(Xv).squeeze(-1).numpy())
        oof[vai]=np.mean(ps,0)
    return oof
def hgb_oof(X):
    F=X.shape[-1]; oof=np.zeros((N,6,6)); yy=np.zeros((N,6,6))
    for i in range(N): yy[i,np.arange(6),Y[i]]=1
    for f in range(5):
        tri=np.where(fold!=f)[0]; vai=np.where(fold==f)[0]; ps=[]
        for s in (0,1,2):
            m=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.06,max_iter=200,l2_regularization=5.0,min_samples_leaf=80,random_state=s).fit(X[tri].reshape(-1,F),yy[tri].ravel())
            p=m.predict_proba(X[vai].reshape(-1,F))[:,1].reshape(-1,6,6); ps.append(np.log(p/(1-p)))
        oof[vai]=np.mean(ps,0)
    return oof
def marg(Z,t):
    ll=Z[np.arange(6)[None,:],PERMS].sum(1)*t; p=np.exp(ll-ll.max()); p/=p.sum(); M=np.zeros((6,6))
    for i in range(6): np.add.at(M[i],PERMS[:,i],p)
    return M
def nll(Zs,t):
    tot=0
    for Z,y in zip(Zs,Y):
        ll=Z[np.arange(6)[None,:],PERMS].sum(1)*t; k=np.where((PERMS==y[None,:]).all(1))[0][0]
        tot+=-(ll[k]-(ll.max()+np.log(np.exp(ll-ll.max()).sum())))
    return tot/len(Zs)
def to_marg(Z):
    ts=[0.05,0.1,0.2,0.4,0.7,1.0,1.5]; t=ts[int(np.argmin([nll(Z,t) for t in ts]))]
    return np.array([marg(z,t) for z in Z]),t
def dec(M): r,c=linear_sum_assignment(-M); p=np.zeros(6,int); p[r]=c; return p
sc=lambda Ms: round(score(np.array([dec(m) for m in Ms]),Y),4)
t0=time.time(); R={}
R['mlp_base']=mlp_oof(X0); print('mlp on v7 features            ',round(score(np.array([hungarian(z) for z in R['mlp_base']]),Y),4),round(time.time()-t0),flush=True)
R['mlp']=mlp_oof(X1);      print('mlp + next-play features      ',round(score(np.array([hungarian(z) for z in R['mlp']]),Y),4),round(time.time()-t0),flush=True)
R['lin']=mlp_oof(X1,hid=0,steps=200); print('linear listwise + next-play   ',round(score(np.array([hungarian(z) for z in R['lin']]),Y),4),round(time.time()-t0),flush=True)
R['hgb']=hgb_oof(X1);      print('hist-GBDT + next-play         ',round(score(np.array([hungarian(z) for z in R['hgb']]),Y),4),round(time.time()-t0),flush=True)
np.savez('/tmp/oof33.npz',**R)
M={k:to_marg(v) for k,v in R.items()}
for k,(m,t) in M.items(): print(k,'temp',t,'marginal decode',sc(m))
best=None
for w in itertools.product([0,1,2,3,4],repeat=3):
    if w[0]==0 or w[0]<w[1]+w[2]: continue
    W=np.array(w,float)/sum(w); B=W[0]*M['mlp'][0]+W[1]*M['lin'][0]+W[2]*M['hgb'][0]; s=sc(B)
    if best is None or s>best[0]: best=(s,w)
    print('blend',w,s)
print('BEST blend',best)
