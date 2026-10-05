import numpy as np, torch, itertools, time, lightgbm as lgb
from scipy.optimize import linear_sum_assignment
from common import hungarian, score
torch.set_num_threads(4)
d=np.load('/tmp/stack_cache_v7.npz'); X=d['X']; ptr=d['ptr']; fold=d['fold']; Y=d['Y']; N=len(Y); F=X.shape[-1]
PERMS=np.array(list(itertools.permutations(range(6)))); P_t=torch.tensor(PERMS); ar=torch.arange(6)
def fit_torch(Xt,idx,hid,wd=1e-2,steps=250,drop=0.2,seeds=(0,1,2),lr=3e-3):
    nets=[]
    for sd in seeds:
        torch.manual_seed(sd)
        net=torch.nn.Sequential(torch.nn.Dropout(drop),torch.nn.Linear(Xt.shape[-1],hid),torch.nn.GELU(),torch.nn.Linear(hid,1)) if hid else torch.nn.Sequential(torch.nn.Dropout(drop),torch.nn.Linear(Xt.shape[-1],1))
        opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=wd)
        for _ in range(steps):
            net.train(); opt.zero_grad(); ll=net(Xt).squeeze(-1)[:,ar[None,:],P_t].sum(-1)
            loss=-(ll.gather(1,idx[:,None]).squeeze(1)-torch.logsumexp(ll,1)).mean(); loss.backward(); opt.step()
        nets.append(net.eval())
    return nets
def oof_torch(hid,**kw):
    oof=np.zeros((N,6,6))
    for f in range(5):
        tri=np.where(fold!=f)[0]; vai=np.where(fold==f)[0]
        mu=X[tri].reshape(-1,F).mean(0); sd=X[tri].reshape(-1,F).std(0)+1e-6
        nets=fit_torch(torch.tensor((X[tri]-mu)/sd).float(),torch.tensor(ptr[tri]),hid,**kw)
        with torch.no_grad(): oof[vai]=np.mean([n(torch.tensor((X[vai]-mu)/sd).float()).squeeze(-1).numpy() for n in nets],0)
    return oof
def oof_lgb():
    oof=np.zeros((N,6,6)); yy=np.zeros((N,6,6))
    for i in range(N): yy[i,np.arange(6),Y[i]]=1
    for f in range(5):
        tri=np.where(fold!=f)[0]; vai=np.where(fold==f)[0]
        ps=[]
        for sd in (0,1):
            m=lgb.train(dict(objective='binary',learning_rate=0.03,num_leaves=15,min_data_in_leaf=40,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5,verbose=-1,seed=sd,num_threads=4,deterministic=True),lgb.Dataset(X[tri].reshape(-1,F),yy[tri].ravel()),num_boost_round=300)
            ps.append(m.predict(X[vai].reshape(-1,F),raw_score=True).reshape(-1,6,6))
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
def calib(Z):
    ts=[0.05,0.1,0.2,0.4,0.7,1,1.5,2.5]; v=[nll(Z,t) for t in ts]; return ts[int(np.argmin(v))],min(v)
def dec(M): r,c=linear_sum_assignment(-M); p=np.zeros(6,int); p[r]=c; return p
t0=time.time(); R={}
R['mlp32']=oof_torch(32); print('mlp32',round(score(np.array([hungarian(z) for z in R['mlp32']]),Y),4),round(time.time()-t0),flush=True)
R['linear']=oof_torch(0,wd=1e-2,steps=200); print('linear',round(score(np.array([hungarian(z) for z in R['linear']]),Y),4),round(time.time()-t0),flush=True)
R['lgbm']=oof_lgb(); print('lgbm',round(score(np.array([hungarian(z) for z in R['lgbm']]),Y),4),round(time.time()-t0),flush=True)
np.savez('/tmp/oof30.npz',**R)
Ms={}
for k,Z in R.items():
    t,v=calib(Z); Ms[k]=np.array([marg(z,t) for z in Z]); print(k,'temp',t,'nll',round(v,4),'marg-decode',round(score(np.array([dec(m) for m in Ms[k]]),Y),4))
for combo in [('mlp32','linear'),('mlp32','lgbm'),('mlp32','linear','lgbm')]:
    for how in ('prob','logprob'):
        B=np.mean([Ms[k] for k in combo],0) if how=='prob' else np.mean([np.log(Ms[k]+1e-6) for k in combo],0)
        print(combo,how,round(score(np.array([dec(b) for b in B]),Y),4))
