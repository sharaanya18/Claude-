import sys; sys.path.insert(0,'.')
import numpy as np
from sim import *
from gp2 import prep, MON
def img_grads(d):
    w=d["weights"]; X=np.concatenate([d["x_batches"].reshape(-1,192),d["x_monitor"],d["x_diag"].reshape(-1,192)])
    y=np.concatenate([d["batch_labels"].ravel(),d["monitor_labels"],d["diagnostic_labels"].ravel()])
    return np.stack([L.gradient(w,X[k:k+1],y[k:k+1]) for k in range(len(y))]).T      # (P,52)
def obs_ops(d, vbar, Jc):
    w=d["weights"]; t=int(d["step"]); sv=np.sqrt(vbar)
    rows=np.concatenate([np.arange(4)+4*(32+k) for k in MON]); Jo=Jc[rows]
    p0=L.probabilities(w,d["x_monitor"])[MON]; lp0=np.log(p0); lp0-=lp0.mean(1,keepdims=True)
    bc1=1-L.BETA1**(t+1); bc2=1-L.BETA2**(t+1)
    gd=[L.gradient(w,d["x_diag"][j],d["diagnostic_labels"][j]) for j in range(2)]
    Hb=[];yb=[]
    for b in range(2):
        Hs=[];ys=[]
        for p in range(2):
            vp=L.BETA2*vbar+(1-L.BETA2)*gd[p]**2; den=bc1*(np.sqrt(vp/bc2)+L.EPS)
            Dp=L.BETA1*sv/den; cp=(1-L.BETA1)*gd[p]/den
            lp=np.log(np.maximum(d["diagnostics"][b,p],1e-300)); lp-=lp.mean(1,keepdims=True)
            ys.append(-(lp-lp0).ravel()/L.LR-Jo@cp); Hs.append(Jo*Dp[None])
        Hb.append(np.concatenate(Hs)); yb.append(np.concatenate(ys))
    return Hb,yb
def gp_grad(d, vbar, hp, n_samp=0, seed=0, Jc=None, G=None):
    """Prior: u = cbar*Ghat@1 + Ghat@dc + e; dc~N(0,sc^2 I), e~N(0,s1 I) (+ s2 Jc^T Jc optionally)."""
    Jc=prep(d) if Jc is None else Jc; G=img_grads(d) if G is None else G
    sv=np.sqrt(vbar); Gh=G/sv[:,None]; P=L.P
    Hb,yb=obs_ops(d,vbar,Jc)
    cbar,sc,s1,s2,sig=hp["cbar"],hp["sc"],hp["s1"],hp["s2"],hp["sig"]
    mu=cbar*Gh.sum(1)
    def Aop(X): return sc**2*Gh@(Gh.T@X)+s1*X+s2*Jc.T@(Jc@X)
    means=np.zeros((2,P)); samples=[np.zeros((2,P)) for _ in range(n_samp)]
    rng=np.random.default_rng(seed)
    for b in range(2):
        H=Hb[b]; y=yb[b]-H@mu
        AHt=Aop(H.T); S=H@AHt; nz=sig**2*np.mean(np.diag(S)); S+=nz*np.eye(len(y)); Sinv=np.linalg.inv(S)
        means[b]=mu+AHt@(Sinv@y)
        for s in range(n_samp):
            u0=sc*Gh@rng.standard_normal(Gh.shape[1])+np.sqrt(s1)*rng.standard_normal(P)+np.sqrt(s2)*Jc.T@rng.standard_normal(Jc.shape[0])
            e=np.sqrt(nz)*rng.standard_normal(len(y))
            samples[s][b]=mu+u0+AHt@(Sinv@(y-H@u0-e))
    def to_state(U):
        out=np.zeros((2,2,P))
        for b in range(2): out[b,0]=U[b]*sv; out[b,1]=vbar
        return out
    return [to_state(means)]+[to_state(s) for s in samples]
