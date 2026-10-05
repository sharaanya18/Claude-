import numpy as np, pickle, torch, itertools, json, pandas as pd
from common import hungarian, score
torch.set_num_threads(4)
FE=pickle.load(open('/tmp/LK20.pkl','rb'))            # (N,6,6,2,126)
tr=pd.read_csv('dataset/public/train.csv'); Y=tr[[f'match_{i}' for i in range(1,7)]].to_numpy()-1; N=len(tr)
fold=np.random.RandomState(0).randint(0,5,N)
PERMS=torch.tensor(list(itertools.permutations(range(6))))
true_idx=torch.tensor([int(np.where((PERMS.numpy()==y).all(1))[0][0]) for y in Y])
X=torch.tensor(FE).float().mean(3)                      # mean over the two artists -> (N,6,6,126)
def make(nf,sep_art=False):
    return torch.nn.Parameter(torch.randn(nf)*0.1-2.0)
def Zof(theta,X,eps):
    a=torch.nn.functional.softplus(theta)
    S=(X*a).sum(-1)
    return torch.log(S+eps)
def nll(Z,idx):
    ll=Z[:,torch.arange(6)[None,:],PERMS].sum(-1)        # (B,720)... gather
    return -(ll.gather(1,idx[:,None]).squeeze(1)-torch.logsumexp(ll,1)).mean()
def Zperm(Z):
    return Z[:,torch.arange(6)[None,:].expand(720,6),PERMS]   # (B,720,6)? built below
def nll2(Z,idx):
    B=Z.shape[0]; ar=torch.arange(6)
    ll=Z[:,ar[None,:],PERMS].sum(-1)                      # indexing: Z[b, i, PERMS[p,i]] -> (B,720,6)
    return -(ll.gather(1,idx[:,None]).squeeze(1)-torch.logsumexp(ll,1)).mean()
def fit(Xtr,idx,l2,steps=300,lr=0.05,eps0=1.0):
    theta=make(Xtr.shape[-1]); leps=torch.nn.Parameter(torch.tensor(0.0))
    opt=torch.optim.Adam([theta,leps],lr=lr)
    for st in range(steps):
        opt.zero_grad(); Z=Zof(theta,Xtr,torch.exp(leps)*eps0); loss=nll2(Z,idx)+l2*(torch.nn.functional.softplus(theta)**2).sum()
        loss.backward(); opt.step()
    return theta.detach(),leps.detach(),loss.item()
for l2 in (1e-3,1e-2,1e-1):
    pred=np.zeros((N,6),int); trn=[]
    for f in range(5):
        tri=torch.tensor(np.where(fold!=f)[0]); vai=np.where(fold==f)[0]
        th,le,lo=fit(X[tri],true_idx[tri],l2); trn.append(lo)
        with torch.no_grad(): Z=Zof(th,X[vai],torch.exp(le)).numpy()
        pred[vai]=[hungarian(z) for z in Z]
    print('l2',l2,'learned kernel OOF',round(score(pred,Y),4),'train loss',np.mean(trn),flush=True)
