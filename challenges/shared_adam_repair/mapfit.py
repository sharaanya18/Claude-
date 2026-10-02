import sys; sys.path.insert(0,'.')
import numpy as np, torch
from sim import *
import gp2
from bsim import _grad, _step, DT
GR=gp2.__dict__.get("GR")
MON=gp2.MON
def map_fit(d, vbar, u_init, Jc, s1=1.0, s2=1.0, sig_rel=0.03, iters=60, z_init=None, prior_draw=None, noise=None, rho=0.0):
    """Exact-observation MAP for u_A,u_B (m=sqrt(vbar)*u). Prior u_b = sqrt(s1) z1_b + sqrt(s2) Jc^T z2_b, z~N(0,I)."""
    T=lambda a: torch.as_tensor(np.asarray(a),dtype=DT)
    w=T(d["weights"]); t=int(d["step"]); sv=T(np.sqrt(vbar)); vb=T(vbar)
    xd=T(d["x_diag"]); yd=torch.as_tensor(d["diagnostic_labels"],dtype=torch.long); xm=T(d["x_monitor"])[MON]
    Jt=T(Jc)
    obs=np.zeros((2,2,6,4))
    p0=L.probabilities(d["weights"],d["x_monitor"])[MON]
    lp0=np.log(p0); lp0-=lp0.mean(1,keepdims=True)
    for b in range(2):
        for p in range(2):
            lp=np.log(np.maximum(d["diagnostics"][b,p],1e-300)); lp-=lp.mean(1,keepdims=True); obs[b,p]=lp-lp0
    obs=T(obs); scale=float(obs.pow(2).mean().sqrt())*sig_rel
    if noise is not None: obs=obs+T(noise)*scale
    # initial z from u_init: use only z2 (response part) by least squares is awkward -> start z1=u_init/sqrt(s1), z2=0
    z1=torch.zeros(2,L.P,dtype=DT); z2=torch.zeros(2,Jc.shape[0],dtype=DT)
    if u_init is not None: z1=T(u_init)/np.sqrt(s1)
    z1.requires_grad_(True); z2.requires_grad_(True)
    ab=np.sqrt(s1); bb=np.sqrt(s2)
    pd_=None if prior_draw is None else (T(prior_draw[0]),T(prior_draw[1]))
    def predict(u):
        outs=[]
        for p in range(2):
            st=(w[None].repeat(2,1), (sv[None]*u), vb[None].repeat(2,1), t)
            st=_step(st, xd[p][None].repeat(2,1,1), yd[p][None].repeat(2,1))
            lp=torch.log_softmax(torch.bmm(torch.tanh(torch.matmul(xm[None],st[0][:,:2304].reshape(2,192,12))+st[0][:,2304:2316][:,None,:]),st[0][:,2316:2364].reshape(2,12,4))+st[0][:,2364:][:,None,:],dim=2)
            lp=lp-lp.mean(2,keepdim=True); outs.append(lp)
        return torch.stack(outs,1)                       # (2 branch, 2 probe, 6, 4)
    opt=torch.optim.LBFGS([z1,z2],lr=1.0,max_iter=iters,line_search_fn="strong_wolfe",tolerance_grad=1e-12,tolerance_change=1e-14)
    def closure():
        opt.zero_grad()
        u=ab*z1+bb*(z2@Jt)
        if pd_ is None: reg=(z1**2).sum()+(z2**2).sum()
        else: reg=((z1-pd_[0])**2).sum()+((z2-pd_[1])**2).sum()
        loss=((predict(u)-obs)**2).sum()/scale**2+reg
        loss.backward(); return loss
    opt.step(closure)
    u=(ab*z1+bb*(z2@Jt)).detach().numpy()
    return u
def to_state(u,vbar):
    out=np.zeros((2,2,L.P))
    for b in range(2): out[b,0]=u[b]*np.sqrt(vbar); out[b,1]=vbar
    return out
