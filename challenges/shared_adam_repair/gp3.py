import sys; sys.path.insert(0,'.')
import numpy as np
from sim import *
from gp2 import prep, MON
GR=np.concatenate([np.zeros(2304),np.ones(12),np.full(48,2),np.full(4,3)]).astype(int)
def gp_joint(d, vbar, hp, n_samp=0, seed=0, Jc=None):
    """Linear-Gaussian joint posterior over u_A,u_B (u=m/sqrt(vbar)). hp: dict(rho,s2,sig,gs=(4 group scales for the iid part))."""
    w=d["weights"]; t=int(d["step"]); Jc=prep(d) if Jc is None else Jc
    obs_rows=np.concatenate([np.arange(4)+4*(32+k) for k in MON]); Jo=Jc[obs_rows]
    p0=L.probabilities(w,d["x_monitor"])[MON]; lp0=np.log(p0); lp0-=lp0.mean(1,keepdims=True)
    bc1=1-L.BETA1**(t+1); bc2=1-L.BETA2**(t+1)
    gd=[L.gradient(w,d["x_diag"][j],d["diagnostic_labels"][j]) for j in range(2)]
    sv=np.sqrt(vbar); P=L.P
    Hb=[];yb=[]
    for b in range(2):
        Hs=[];ys=[]
        for p in range(2):
            vp=L.BETA2*vbar+(1-L.BETA2)*gd[p]**2; den=bc1*(np.sqrt(vp/bc2)+L.EPS)
            Dp=L.BETA1*sv/den; cp=(1-L.BETA1)*gd[p]/den
            lp=np.log(np.maximum(d["diagnostics"][b,p],1e-300)); lp-=lp.mean(1,keepdims=True)
            ys.append(-(lp-lp0).ravel()/L.LR-Jo@cp); Hs.append(Jo*Dp[None])
        Hb.append(np.concatenate(Hs)); yb.append(np.concatenate(ys))
    rho,s2,sig=hp["rho"],hp["s2"],hp["sig"]; gsc=np.asarray(hp.get("gs",(1,1,1,1)),float)[GR]
    def Aop(X):                                 # X (P,k) -> A X, A = diag(gsc) + s2 Jc^T Jc
        return gsc[:,None]*X + s2*Jc.T@(Jc@X)
    n1=Hb[0].shape[0]
    AH=[Aop(h.T) for h in Hb]                    # (P,n1) each
    K=np.array([[1,rho],[rho,1]])
    # S = sum blocks: S[b,b'] = K[b,b'] H_b A H_b'^T
    S=np.block([[K[a,b]*Hb[a]@AH[b] for b in range(2)] for a in range(2)])
    nz=sig**2*np.mean([np.mean(np.diag(h@h.T)) for h in Hb])
    S+=nz*np.eye(2*n1); Sinv=np.linalg.inv(S); y=np.concatenate(yb)
    CHt=lambda alpha: np.stack([sum(K[a,b]*AH[b]@alpha[b*n1:(b+1)*n1] for b in range(2)) for a in range(2)])   # (2,P)
    mean_u=CHt(Sinv@y)
    def to_state(U):
        out=np.zeros((2,2,P))
        for b in range(2): out[b,0]=U[b]*sv; out[b,1]=vbar
        return out
    states=[to_state(mean_u)]
    if n_samp:
        rng=np.random.default_rng(seed)
        Lk=np.linalg.cholesky(K+1e-9*np.eye(2))
        for s in range(n_samp):
            # prior draw: u_b = sum_b' Lk[b,b'] (sqrt(gsc) z1 + sqrt(s2) Jc^T z2)
            z=[np.sqrt(gsc)*rng.standard_normal(P)+np.sqrt(s2)*Jc.T@rng.standard_normal(Jc.shape[0]) for _ in range(2)]
            u0=np.stack([Lk[b,0]*z[0]+Lk[b,1]*z[1] for b in range(2)])
            e=np.sqrt(nz)*rng.standard_normal(2*n1)
            r=y-np.concatenate([Hb[b]@u0[b] for b in range(2)])-e
            states.append(to_state(u0+CHt(Sinv@r)))
    return states
