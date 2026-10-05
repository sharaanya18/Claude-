import numpy as np, scipy.sparse as sp, time, pickle
from common import *
from sklearn.preprocessing import normalize
exec(open('exp1.py').read().split("rows=tr.copy()")[0])
rows=tr.copy(); fold=np.random.RandomState(0).randint(0,5,len(rows)); Y=np.array(list(rows.y))
n1=np.asarray(X1.sum(1)).ravel()
rng=np.random.RandomState(1)
t0=time.time(); store={}
for f in range(5):
    R=rows[fold==f]
    held=np.zeros(NS,bool)
    for l in R.pre:
        for s in l: held[sid[s]]=True
    fi=np.where(~is_test&~held)[0]
    Pidx=[sid[s] for l in R.pre for s in l]
    bgi=rng.choice(fi[n1[fi]>=8],2500,replace=False)     # background prefixes = other fit sessions
    cols=sorted({a for c in R.cand for ci in c for a in cc[ci]}); cm={a:i for i,a in enumerate(cols)}
    W=sp.diags(idf**0.5); Fm=normalize(Xall[fi]@W); Tc=Xall[fi][:,cols].toarray()
    for gam in (1,2):
        def S_of(idx,excl=None):
            Pm=normalize(X1[idx]@W); K=(Pm@Fm.T).toarray()
            if excl is not None:
                pos={s:i for i,s in enumerate(fi)}; 
                for r_,s in enumerate(idx): K[r_,pos[s]]=0
            return (K**gam)@Tc
        S=S_of(Pidx); B=S_of(bgi,excl=True)
        mu=B.mean(0); sd=B.std(0)+1e-6; lmu=np.log(B+.01).mean(0); lsd=np.log(B+.01).std(0)+1e-6
        for ri,(idx,r) in enumerate(R.iterrows()):
            for nm,fn in (('raw',lambda x:x),('log',lambda x:np.log(x+.01)),('colz',lambda x:(x-mu_)/sd_),('logz',None)):
                pass
        for ri,(idx,r) in enumerate(R.iterrows()):
            M=np.zeros((6,6,2)); 
            for j in range(6):
                for k,ci in enumerate(r.cand):
                    for q,a in enumerate(cc[ci]): M[j,k,q]=cm[a]
            ci_=M[...,:].astype(int)
            Sr=S[ri*6:(ri+1)*6]          # (6, ncols)
            sc=Sr[:,ci_[0,:,0]]; sc2=Sr[:,ci_[0,:,1]]    # (6 prefixes, 6 cands) for a1,a2
            d={'sc1':sc,'sc2':sc2,'mu1':np.tile(mu[ci_[0,:,0]],(6,1)),'sd1':np.tile(sd[ci_[0,:,0]],(6,1)),'mu2':np.tile(mu[ci_[0,:,1]],(6,1)),'sd2':np.tile(sd[ci_[0,:,1]],(6,1)),
               'lmu1':np.tile(lmu[ci_[0,:,0]],(6,1)),'lsd1':np.tile(lsd[ci_[0,:,0]],(6,1)),'lmu2':np.tile(lmu[ci_[0,:,1]],(6,1)),'lsd2':np.tile(lsd[ci_[0,:,1]],(6,1))}
            store[(gam,idx)]=d
print(time.time()-t0)
pickle.dump(store,open('/tmp/store7.pkl','wb'))
for gam in (1,2):
    for nm in ('raw','log','colz','logz','pctl'):
        pred=[]
        for idx in rows.index:
            d=store[(gam,idx)]
            if nm=='raw': M=(d['sc1']+d['sc2'])/2
            elif nm=='log': M=(np.log(d['sc1']+.01)+np.log(d['sc2']+.01))/2
            elif nm=='colz': M=(d['sc1']-d['mu1'])/d['sd1']+(d['sc2']-d['mu2'])/d['sd2']
            elif nm=='logz': M=(np.log(d['sc1']+.01)-d['lmu1'])/d['lsd1']+(np.log(d['sc2']+.01)-d['lmu2'])/d['lsd2']
            else: M=(d['sc1']-d['mu1'])/d['sd1']  # a1 only
            pred.append(hungarian(M))
        print(gam,nm,round(score(np.array(pred),Y),4))
