import numpy as np, torch, itertools, time, sys
from common import hungarian, score
torch.set_num_threads(4)
d=np.load('/tmp/stack_cache.npz'); X=d['X']; ptr=torch.tensor(d['ptr']); fold=d['fold']; Y=d['Y']; N=len(Y)
PERMS=torch.tensor(list(itertools.permutations(range(6)))); ar=torch.arange(6)
class Plain(torch.nn.Module):
    def __init__(s,F,hid,drop):
        super().__init__(); s.n=torch.nn.Sequential(torch.nn.Dropout(drop),torch.nn.Linear(F,hid),torch.nn.GELU(),torch.nn.Linear(hid,1))
    def forward(s,x): return s.n(x).squeeze(-1)
class Equi(torch.nn.Module):
    """pair MLP + two rounds of row/column mean-pooled context (permutation-equivariant over prefixes and candidates)"""
    def __init__(s,F,hid,drop,rounds=2):
        super().__init__(); s.inp=torch.nn.Sequential(torch.nn.Dropout(drop),torch.nn.Linear(F,hid),torch.nn.GELU())
        s.mix=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(3*hid,hid),torch.nn.GELU()) for _ in range(rounds)]); s.out=torch.nn.Linear(hid,1)
    def forward(s,x):
        h=s.inp(x)
        for m in s.mix:
            h=h+m(torch.cat([h,h.mean(1,keepdim=True).expand_as(h),h.mean(2,keepdim=True).expand_as(h)],-1))
        return s.out(h).squeeze(-1)
def run(kind,hid=32,wd=1e-2,steps=250,drop=0.2,seeds=(0,1,2),lr=3e-3,rounds=2):
    oof=np.zeros((N,6,6))
    for f in range(5):
        tri=np.where(fold!=f)[0]; vai=np.where(fold==f)[0]
        mu=X[tri].reshape(-1,X.shape[-1]).mean(0); sd=X[tri].reshape(-1,X.shape[-1]).std(0)+1e-6
        Xt=torch.tensor((X[tri]-mu)/sd).float(); Xv=torch.tensor((X[vai]-mu)/sd).float(); idx=ptr[tri]; preds=[]
        for sd_ in seeds:
            torch.manual_seed(sd_); net=Plain(X.shape[-1],hid,drop) if kind=='plain' else Equi(X.shape[-1],hid,drop,rounds)
            opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=wd)
            for _ in range(steps):
                net.train(); opt.zero_grad(); ll=net(Xt)[:,ar[None,:],PERMS].sum(-1)
                loss=-(ll.gather(1,idx[:,None]).squeeze(1)-torch.logsumexp(ll,1)).mean(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): preds.append(net(Xv).numpy())
        oof[vai]=np.mean(preds,0)
    return oof
t0=time.time()
for cfg in [dict(kind='plain'),dict(kind='equi',hid=32,steps=250),dict(kind='equi',hid=32,steps=150),dict(kind='equi',hid=64,steps=200,wd=5e-2)]:
    o=run(**cfg); print(cfg,'hung',round(score(np.array([hungarian(z) for z in o]),Y),4),round(time.time()-t0),flush=True)
    np.save('/tmp/o_%s.npy'%('_'.join(f'{k}{v}' for k,v in cfg.items())),o)
