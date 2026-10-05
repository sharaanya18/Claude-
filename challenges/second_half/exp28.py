import numpy as np, torch, itertools, time, sys
from common import hungarian, score
torch.set_num_threads(4)
d=np.load('/tmp/stack_cache.npz'); X=d['X']; ptr=torch.tensor(d['ptr']); fold=d['fold']; Y=d['Y']; N=len(Y)
LK=np.load('/tmp/LK_counts.npy').astype(np.float32)
PERMS=torch.tensor(list(itertools.permutations(range(6)))); ar=torch.arange(6)
Xo=np.concatenate([X[...,:-7],X[...,-2:]],-1)          # other features: drop separately-trained Z (+4 row-wise Z), keep release links
def rw(z):
    o=[]
    for ax in (1,2):
        m=z.mean(ax,keepdim=True); s=z.std(ax,keepdim=True)+1e-6; o.append((z-m)/s)
    return o
class Joint(torch.nn.Module):
    def __init__(s,Fo,Fk,heads,hid,drop):
        super().__init__(); s.th=torch.nn.Parameter(torch.randn(heads,Fk)*0.1-2.0); s.le=torch.nn.Parameter(torch.zeros(heads))
        s.net=torch.nn.Sequential(torch.nn.Dropout(drop),torch.nn.Linear(Fo+heads*3,hid),torch.nn.GELU(),torch.nn.Linear(hid,1)); s.heads=heads
    def kern(s,K):
        S_=torch.einsum('bijf,hf->bijh',K,torch.nn.functional.softplus(s.th))
        return torch.log(S_+torch.exp(s.le))
    def forward(s,xo,K):
        Z=s.kern(K); parts=[Z]
        for h in range(s.heads):
            parts+= [t.unsqueeze(-1) for t in rw(Z[...,h])]
        # parts: Z (B,6,6,H) + 2 per head -> normalise z-scores already; scale of raw Z handled by net
        f=torch.cat([xo,Z]+[p for p in parts[1:]],-1)
        return s.net(f).squeeze(-1)
def run(heads=4,hid=32,wd=1e-2,steps=250,drop=0.2,seeds=(0,1,2),lr=3e-3,l2=1e-2,lrk=None):
    oof=np.zeros((N,6,6))
    for f in range(5):
        tri=np.where(fold!=f)[0]; vai=np.where(fold==f)[0]
        mu=Xo[tri].reshape(-1,Xo.shape[-1]).mean(0); sd=Xo[tri].reshape(-1,Xo.shape[-1]).std(0)+1e-6
        Xt=torch.tensor((Xo[tri]-mu)/sd).float(); Xv=torch.tensor((Xo[vai]-mu)/sd).float()
        Kt=torch.tensor(LK[tri]); Kv=torch.tensor(LK[vai]); idx=ptr[tri]; preds=[]
        for sd_ in seeds:
            torch.manual_seed(sd_); net=Joint(Xo.shape[-1],LK.shape[-1],heads,hid,drop)
            opt=torch.optim.AdamW([{'params':[net.th,net.le],'lr':lrk or lr*5,'weight_decay':0},{'params':net.net.parameters(),'lr':lr,'weight_decay':wd}])
            for _ in range(steps):
                net.train(); opt.zero_grad(); ll=net(Xt,Kt)[:,ar[None,:],PERMS].sum(-1)
                loss=-(ll.gather(1,idx[:,None]).squeeze(1)-torch.logsumexp(ll,1)).mean()+l2*(torch.nn.functional.softplus(net.th)**2).sum()
                loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): preds.append(net(Xv,Kv).numpy())
        oof[vai]=np.mean(preds,0)
    return oof
t0=time.time()
for cfg in [dict(heads=1),dict(heads=4),dict(heads=4,steps=150),dict(heads=8,hid=32,l2=1e-1)]:
    o=run(**cfg); print(cfg,'hung',round(score(np.array([hungarian(z) for z in o]),Y),4),round(time.time()-t0),flush=True)
    np.save('/tmp/joint_%s.npy'%('_'.join(f'{k}{v}' for k,v in cfg.items())),o)
