import numpy as np, pickle
exec(open('exp14.py').read().split("run(TF,'traits only')")[0])
LKZ=np.load('/tmp/LKZ.npy'); R22=np.load('/tmp/R22.npy')
def rw(v):
    o=[]
    for ax in (1,2):
        m=v.mean(ax,keepdims=True); s=v.std(ax,keepdims=True)+1e-6; o+=[(v-m)/s,np.argsort(np.argsort(v,ax),ax)]
    return o
lk=np.stack([LKZ]+rw(LKZ),-1); lnk=np.log1p(R22)
base=np.concatenate([X,TF],-1)
S0,_=run(base,'base')
S1,_=run(np.concatenate([base,lk],-1),'base+learned kernel')
S2,_=run(np.concatenate([base,lnk],-1),'base+release links')
S3,_=run(np.concatenate([base,lk,lnk],-1),'base+both')

import torch, itertools
torch.set_num_threads(4)
PERMS=torch.tensor(list(itertools.permutations(range(6)))); ar=torch.arange(6)
tidx=torch.tensor([int(np.where((PERMS.numpy()==y).all(1))[0][0]) for y in Y])
XX=np.concatenate([base,lk,lnk],-1).astype(np.float32)
mu=XX.reshape(-1,XX.shape[-1]).mean(0); sd=XX.reshape(-1,XX.shape[-1]).std(0)+1e-6
XT=torch.tensor((XX-mu)/sd)
def train_mlp(Xt,idx,hid=32,wd=1e-2,steps=250,lr=3e-3,seed=0,drop=0.2):
    torch.manual_seed(seed)
    net=torch.nn.Sequential(torch.nn.Dropout(drop),torch.nn.Linear(Xt.shape[-1],hid),torch.nn.GELU(),torch.nn.Linear(hid,1))
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=wd)
    for st in range(steps):
        net.train(); opt.zero_grad(); Z=net(Xt).squeeze(-1); ll=Z[:,ar[None,:],PERMS].sum(-1)
        loss=-(ll.gather(1,idx[:,None]).squeeze(1)-torch.logsumexp(ll,1)).mean(); loss.backward(); opt.step()
    net.eval(); return net
for hid,wd,steps in [(0,0,0)] if False else [(16,1e-2,150),(32,1e-2,250),(32,1e-1,400),(64,1e-1,250)]:
    oofm=np.zeros((N,6,6))
    for f in range(5):
        tri=torch.tensor(np.where(fold!=f)[0]); vai=np.where(fold==f)[0]
        nets=[train_mlp(XT[tri],tidx[tri],hid,wd,steps,seed=s) for s in range(3)]
        with torch.no_grad(): oofm[vai]=np.mean([n(XT[vai]).squeeze(-1).numpy() for n in nets],0)
    print('MLP perm-NLL',hid,wd,steps,'hung',round(score(np.array([hungarian(z) for z in oofm]),Y),4),flush=True)
    np.save(f'/tmp/oofm_{hid}_{wd}_{steps}.npy',oofm)
