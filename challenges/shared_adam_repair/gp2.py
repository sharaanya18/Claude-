import sys; sys.path.insert(0,'.')
import numpy as np
from sim import *
from ablate2 import logit_jac, image_set
MON=[0,1,3,6,9,10]
Cm=np.eye(4)-0.25
def center_rows(J): return np.einsum("ab,ibp->iap", Cm, J).reshape(-1, J.shape[-1])
def prep(d):
    w=d["weights"]; J=logit_jac(w,image_set(d)); return center_rows(J)
def gp_samples(d, vbar, s1=1.0, s2=1.0, sig=0.3, n_samp=0, seed=0, Jc=None):
    """Linear-Gaussian posterior over u=m/sqrt(vbar) per branch from the 4 diagnostic observations. Returns list of (2,2,P) states."""
    w=d["weights"]; t=int(d["step"])
    Jc=prep(d) if Jc is None else Jc
    obs_rows=np.concatenate([np.arange(4)+4*(32+k) for k in MON]); Jo=Jc[obs_rows]
    p0=L.probabilities(w,d["x_monitor"])[MON]; lp0=np.log(p0); lp0-=lp0.mean(1,keepdims=True)
    bc1=1-L.BETA1**(t+1); bc2=1-L.BETA2**(t+1)
    gd=[L.gradient(w,d["x_diag"][j],d["diagnostic_labels"][j]) for j in range(2)]
    rng=np.random.default_rng(seed); sv=np.sqrt(vbar)
    means=np.zeros((2,L.P)); samps=[np.zeros((2,L.P)) for _ in range(n_samp)]
    for b in range(2):
        Hs=[];ys=[]
        for p in range(2):
            vp=L.BETA2*vbar+(1-L.BETA2)*gd[p]**2; den=bc1*(np.sqrt(vp/bc2)+L.EPS)
            Dp=L.BETA1*sv/den; cp=(1-L.BETA1)*gd[p]/den
            lp=np.log(np.maximum(d["diagnostics"][b,p],1e-300)); lp-=lp.mean(1,keepdims=True)
            ys.append(-(lp-lp0).ravel()/L.LR-Jo@cp); Hs.append(Jo*Dp[None])
        H=np.concatenate(Hs); y=np.concatenate(ys); HJ=H@Jc.T
        nz=sig**2*np.mean(np.diag(H@H.T))
        S=s1*H@H.T+s2*HJ@HJ.T+nz*np.eye(len(y)); Sinv=np.linalg.inv(S)
        def post(yv, u0):
            a=Sinv@(yv-H@u0); return u0+s1*H.T@a+s2*Jc.T@(HJ.T@a)
        means[b]=post(y,np.zeros(L.P))
        for s in range(n_samp):
            u0=np.sqrt(s1)*rng.standard_normal(L.P)+np.sqrt(s2)*Jc.T@rng.standard_normal(Jc.shape[0])
            e=np.sqrt(nz)*rng.standard_normal(len(y))
            samps[s][b]=post(y-e,u0) if False else post(y,u0)-0  # Matheron: u0 + K H^T S^-1 (y - H u0 - e)
            a=Sinv@(y-H@u0-e); samps[s][b]=u0+s1*H.T@a+s2*Jc.T@(HJ.T@a)
    def to_state(U):
        out=np.zeros((2,2,L.P))
        for b in range(2): out[b,0]=U[b]*sv; out[b,1]=vbar
        return out
    return to_state(means), [to_state(s) for s in samps]
