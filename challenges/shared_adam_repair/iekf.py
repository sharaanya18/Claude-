"""Iterated linear-Gaussian (Gauss-Newton) inference of u=m/sqrt(vbar) per branch using the EXACT diagnostic forward model."""
import sys; sys.path.insert(0,'.')
import numpy as np, torch
from sim import *
import gp2
from bsim import _step, DT
MON=gp2.MON
def diag_forward_factory(d, vbar):
    T=lambda a: torch.as_tensor(np.asarray(a),dtype=DT)
    w=T(d["weights"]); t=int(d["step"]); sv=T(np.sqrt(vbar)); vb=T(vbar)
    xd=T(d["x_diag"]); yd=torch.as_tensor(d["diagnostic_labels"],dtype=torch.long); xm=T(d["x_monitor"])[MON]
    p0=L.probabilities(d["weights"],d["x_monitor"])[MON]; lp0=np.log(p0); lp0-=lp0.mean(1,keepdims=True)
    obs=np.zeros((2,2,6,4))
    for b in range(2):
        for p in range(2):
            lp=np.log(np.maximum(d["diagnostics"][b,p],1e-300)); lp-=lp.mean(1,keepdims=True); obs[b,p]=lp-lp0
    def f(u, vb_=vb, sv_=sv):                 # u (P,) -> (2 probes, 6, 4) centered log-prob change
        outs=[]
        for p in range(2):
            st=(w[None], (sv_*u)[None], vb_[None], t); st=_step(st, xd[p][None], yd[p][None]); wn=st[0][0]
            a=wn[:2304].reshape(192,12); bb=wn[2304:2316]; c=wn[2316:2364].reshape(12,4); dd=wn[2364:]
            lp=torch.log_softmax(torch.tanh(xm@a+bb)@c+dd,dim=1); lp=lp-lp.mean(1,keepdim=True)
            outs.append(lp-T(lp0)+0*lp)
        return torch.stack(outs)
    return f, obs
def iekf(d, vbar, Jc, hp, n_iter=4, u_init=None, prior_mean=None):
    """returns u (2,P) MAP-ish. Prior cov A = s1*diag(gsc) + s2 Jc^T Jc (same for both branches, independent)."""
    s1,s2,sig=hp.get("s1",1.0),hp["s2"],hp["sig"]
    f,obs=diag_forward_factory(d,vbar); P=L.P
    u=np.zeros((2,P)) if u_init is None else u_init.copy()
    for b in range(2):
        y=obs[b].ravel()
        for it in range(n_iter):
            ut=torch.as_tensor(u[b],dtype=DT)
            H=torch.autograd.functional.jacobian(lambda x: f(x).reshape(-1), ut, vectorize=True).numpy()   # (48,P)
            fu=f(ut).reshape(-1).detach().numpy()
            AHt=s1*H.T+s2*Jc.T@(Jc@H.T)
            S=H@AHt; nz=(sig**2)*np.mean(np.diag(S))
            r=y-fu+H@u[b]
            u[b]=AHt@np.linalg.solve(S+nz*np.eye(len(y)),r)
        # final residual
    return u
