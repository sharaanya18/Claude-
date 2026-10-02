import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, time
from sim import *
tr = pd.read_csv(PUB/"train.csv"); lab = np.load("order_labels.npz")
rng = np.random.default_rng(0); idx = rng.choice(len(tr), 24, replace=False)

def logit_jac(w, x):
    """J[i] = d logits_i / d w, shape (n,4,P)"""
    a,b,c,dd = L.unpack(w); n=len(x)
    h = np.tanh(x@a+b); P=len(w); J=np.zeros((n,4,P))
    for i in range(n):
        dh_dz = 1-h[i]**2          # (12,)
        for k in range(4):
            gA = np.outer(x[i], c[:,k]*dh_dz)          # (192,12)
            gb = c[:,k]*dh_dz
            gC = np.zeros((12,4)); gC[:,k]=h[i]
            gD = np.zeros(4); gD[k]=1
            J[i,k]=np.concatenate([gA.ravel(), gb, gC.ravel(), gD])
    return J

def image_set(d):
    X = np.concatenate([d["x_batches"].reshape(-1,192), d["x_monitor"], d["x_diag"].reshape(-1,192)])
    return X
def pick(d, mom, orders, qtrue, N=150):
    qs=[quality(d, list(o), moments=mom)[0] for o in orders[:N]]
    return qtrue[int(np.argmax(qs))]
if __name__=="__main__":
    res={}
    for i in idx:
        d = load_request(tr.iloc[i]); M=d["moments"]; o=lab["orders"][i].astype(int); Dt=lab["D"][i]; lo=lab["losses"][i]
        lim=d["request"]["divergence_limit"]; cap=d["request"]["capability_limit"]
        qtrue=np.where((lo>cap).any(1),0,lim/np.maximum(Dt,lim))[:150]
        J = logit_jac(d["weights"], image_set(d)).reshape(-1, L.P)      # (208,P)
        vbar = M[:,1].mean(0)
        for name, vv in [("vtrue", None), ("vbar", vbar)]:
            newM = M.copy()
            for b in range(2):
                v = M[b,1] if vv is None else vv
                sd = np.sqrt(v)+1e-8
                u = M[b,0]/sd
                r = J@u
                # min-norm in u-space
                ut = np.linalg.lstsq(J, r, rcond=1e-10)[0]
                newM[b,0] = ut*sd
                newM[b,1] = v
            res.setdefault("minnorm_"+name,[]).append(pick(d,newM,o,qtrue))
        res.setdefault("true",[]).append(qtrue.max()); res.setdefault("random",[]).append(qtrue.mean())
        print(i,{k:round(v[-1],3) for k,v in res.items()},flush=True)
    print({k:round(float(np.mean(v)),3) for k,v in res.items()})
    