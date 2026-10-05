import numpy as np, torch, itertools
from common import hungarian, score
torch.set_num_threads(4)
d=np.load('/tmp/stack_cache_v7.npz'); X=d['X']; ptr=d['ptr']; fold=d['fold']; Y=d['Y']; N=len(Y); F=X.shape[-1]
PERMS=torch.tensor(list(itertools.permutations(range(6)))); ar=torch.arange(6)
rng=np.random.RandomState(0)
for frac in (0.25,0.5,0.75,1.0):
    oof=np.zeros((N,6,6))
    for f in range(5):
        tri=np.where(fold!=f)[0]; tri=rng.permutation(tri)[:int(len(tri)*frac)]; vai=np.where(fold==f)[0]
        mu=X[tri].reshape(-1,F).mean(0); sd=X[tri].reshape(-1,F).std(0)+1e-6
        Xt=torch.tensor((X[tri]-mu)/sd).float(); Xv=torch.tensor((X[vai]-mu)/sd).float(); idx=torch.tensor(ptr[tri]); ps=[]
        for s in (0,1,2):
            torch.manual_seed(s); net=torch.nn.Sequential(torch.nn.Dropout(0.2),torch.nn.Linear(F,32),torch.nn.GELU(),torch.nn.Linear(32,1))
            opt=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=1e-2)
            for _ in range(250):
                net.train(); opt.zero_grad(); ll=net(Xt).squeeze(-1)[:,ar[None,:],PERMS].sum(-1)
                loss=-(ll.gather(1,idx[:,None]).squeeze(1)-torch.logsumexp(ll,1)).mean(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): ps.append(net(Xv).squeeze(-1).numpy())
        oof[np.where(fold==f)[0]]=np.mean(ps,0)
    print(f'train rows {frac:.2f} (~{int(N*0.8*frac)}): OOF {score(np.array([hungarian(z) for z in oof]),Y):.4f}',flush=True)
