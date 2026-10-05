"""Dev: bag-of-artists -> second-half-artists network (multinomial likelihood, DAE style)."""
import numpy as np, torch, torch.nn as nn, scipy.sparse as sp
torch.set_num_threads(4)
class Net(nn.Module):
    def __init__(s,nin,nout,d=256,drop=0.5):
        super().__init__(); s.emb=nn.EmbeddingBag(nin,d,mode='sum'); s.ln=nn.LayerNorm(d); s.drop=nn.Dropout(drop)
        s.mlp=nn.Sequential(nn.Linear(d,d),nn.GELU()); s.out=nn.Linear(d,nout)
    def forward(s,idx,off,w):
        h=s.emb(idx,off,per_sample_weights=w); h=s.drop(s.ln(h)); h=h+s.mlp(h); return s.out(s.drop(h))
def csr_batch(X,rows,wvec,drop_p=0.0,rng=None):
    sub=X[rows]; idx=sub.indices.copy(); ptr=sub.indptr
    w=np.take(wvec,idx).astype(np.float32)
    if drop_p>0:
        keep=rng.rand(len(idx))>drop_p; 
        counts=np.add.reduceat(keep.astype(np.int64),ptr[:-1]) if len(idx) else None
        # rebuild
        rowid=np.repeat(np.arange(len(rows)),np.diff(ptr)); idx=idx[keep]; w=w[keep]; rowid=rowid[keep]
        ptr=np.r_[0,np.cumsum(np.bincount(rowid,minlength=len(rows)))]
    # l2-normalise weights per row
    cnt=np.diff(ptr); w=w/np.repeat(np.sqrt(np.maximum(np.bincount(np.repeat(np.arange(len(rows)),cnt),weights=w**2,minlength=len(rows)),1e-9)),cnt)
    return torch.from_numpy(idx.astype(np.int64)),torch.from_numpy(ptr[:-1].astype(np.int64)),torch.from_numpy(w.astype(np.float32))
def train_net(Xin,Xtgt,wvec,nout,epochs=25,bs=256,d=256,lr=2e-3,wd=1e-5,drop=0.5,seed=0,aug=None,verbose=False):
    """Xin: csr (n,nin) first-half bags; Xtgt: csr (n,nout) target bags (multi-hot). aug: optional (Xin2,Xtgt2) extra random-split examples."""
    torch.manual_seed(seed); rng=np.random.RandomState(seed)
    net=Net(Xin.shape[1],nout,d,drop); opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=wd)
    if aug is not None: Xin=sp.vstack([Xin,aug[0]]).tocsr(); Xtgt=sp.vstack([Xtgt,aug[1]]).tocsr()
    n=Xin.shape[0]; steps=epochs*((n+bs-1)//bs); sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,total_steps=steps)
    for ep in range(epochs):
        perm=rng.permutation(n); tot=0
        net.train()
        for b in range(0,n,bs):
            rows=perm[b:b+bs]; idx,off,w=csr_batch(Xin,rows,wvec,drop_p=0.1,rng=rng)
            T=torch.from_numpy(Xtgt[rows].toarray()); T=T/T.sum(1,keepdim=True).clamp(min=1)
            lo=net(idx,off,w); loss=-(T*torch.log_softmax(lo,-1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot+=loss.item()*len(rows)
        if verbose: print(ep,tot/n,flush=True)
    return net
@torch.no_grad()
def predict(net,X,wvec,bs=1024):
    net.eval(); out=[]
    for b in range(0,X.shape[0],bs):
        idx,off,w=csr_batch(X,np.arange(b,min(b+bs,X.shape[0])),wvec); out.append(torch.log_softmax(net(idx,off,w),-1).numpy())
    return np.vstack(out)
