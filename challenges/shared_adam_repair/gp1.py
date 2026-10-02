import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from sim import *
from ablate2 import logit_jac, image_set
from basis import grad_basis, ols_state
MON=[0,1,3,6,9,10]
tr = pd.read_csv(PUB/"train.csv"); lab = np.load("order_labels.npz")
Cm = np.eye(4)-0.25
def center_rows(J): return np.einsum("ab,ibp->iap", Cm, J).reshape(-1, J.shape[-1])
def gp_state(d, vbar, s2=1.0, sig=0.1, J=None):
    w=d["weights"]; t=int(d["step"])
    if J is None: J = logit_jac(w, image_set(d))
    Jc = center_rows(J)                                  # (208,P)
    obs_rows = np.concatenate([np.arange(4)+4*(32+k) for k in MON])
    Jo = Jc[obs_rows]
    p0 = L.probabilities(w, d["x_monitor"])[MON]
    lp0 = np.log(p0); lp0 -= lp0.mean(1,keepdims=True)
    bc1=1-L.BETA1**(t+1); bc2=1-L.BETA2**(t+1)
    gd=[L.gradient(w, d["x_diag"][j], d["diagnostic_labels"][j]) for j in range(2)]
    K = Jc@Jc.T
    out=np.zeros((2,2,L.P))
    # prior covariance of u: A = I + s2 * Jc^T Jc  (scaled so s1=1)
    for b in range(2):
        Hs=[];ys=[]
        for p in range(2):
            vp = (L.BETA2*vbar + (1-L.BETA2)*gd[p]**2)
            den = bc1*(np.sqrt(vp/bc2)+L.EPS)
            Dp = L.BETA1*np.sqrt(vbar)/den*0 + L.BETA1/den   # placeholder, recomputed below
            # m = sqrt(vbar)*u -> m' = .9*sqrt(vbar) u + .1 gd ; e = m'/den
            Dp = L.BETA1*np.sqrt(vbar)/den; cp = (1-L.BETA1)*gd[p]/den
            # but response of images is J u_old where u_old = m/sqrt(vbar) ... use u := m/sqrt(vbar)
            lp = np.log(np.maximum(d["diagnostics"][b,p],1e-300)); lp -= lp.mean(1,keepdims=True)
            dl = (lp-lp0).ravel()                       # change in centered logits (24,)
            y = -dl/L.LR - Jo@cp
            Hs.append(Jo*Dp[None]); ys.append(y)
        H=np.concatenate(Hs); y=np.concatenate(ys)
        # A = I + s2 J^T J ; H A H^T = H H^T + s2 (H J^T)(J H^T)
        HJ = H@Jc.T
        S = H@H.T + s2*HJ@HJ.T + sig**2*np.mean(np.diag(H@H.T))*np.eye(len(y))
        alpha = np.linalg.solve(S, y)
        u = H.T@alpha + s2*Jc.T@(HJ.T@alpha)
        out[b,0]=u; 
    return out
if __name__=="__main__":
    rng=np.random.default_rng(0); idx=rng.choice(len(tr),24,replace=False)
    res={}
    def pick(d,mom,o,qtrue,N=150):
        qs=[quality(d,list(x),moments=mom)[0] for x in o[:N]]; return qtrue[int(np.argmax(qs))]
    for i in idx:
        d=load_request(tr.iloc[i]); M=d["moments"]; o=lab["orders"][i].astype(int); Dt=lab["D"][i]; lo=lab["losses"][i]
        lim=d["request"]["divergence_limit"]; cap=d["request"]["capability_limit"]
        qtrue=np.where((lo>cap).any(1),0,lim/np.maximum(Dt,lim))[:150]
        vbar=M[:,1].mean(0); J=logit_jac(d["weights"],image_set(d))
        for s2 in [0.0,1.0,100.0]:
          for sig in [0.03,0.3]:
            U=gp_state(d,vbar,s2,sig,J)
            newM=M.copy()
            for b in range(2): newM[b,0]=U[b,0]*np.sqrt(vbar); newM[b,1]=vbar
            res.setdefault(f"gp_s2={s2}_sig={sig}",[]).append(pick(d,newM,o,qtrue))
        res.setdefault("true",[]).append(qtrue.max()); res.setdefault("random",[]).append(qtrue.mean())
    print({k:round(float(np.mean(v)),3) for k,v in res.items()})
