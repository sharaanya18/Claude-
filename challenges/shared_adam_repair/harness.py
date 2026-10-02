import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, time, torch
from multiprocessing import Pool
from sim import *
from bsim import Sandbox
import gp1
tr = pd.read_csv(PUB/"train.csv"); _l = np.load("order_labels.npz"); lab = {k: _l[k] for k in _l.files}
def val_idx(per_scene=8, seed=0):
    rng=np.random.default_rng(seed); out=[]
    for g,df in tr.groupby("experiment_group"): out+=list(rng.choice(df.index.values, per_scene, replace=False))
    return sorted(out)
N=400
def qlab(i, d):
    lim=d["request"]["divergence_limit"]; cap=d["request"]["capability_limit"]
    return np.where((lab["losses"][i][:N]>cap).any(1),0,lim/np.maximum(lab["D"][i][:N],lim))
FN=None
def _work(i):
    fn=FN
    torch.set_num_threads(1)
    d=load_request(tr.iloc[i]); o=lab["orders"][i].astype(int)[:N]; q=qlab(i,d)
    mom_list=fn(d,i)                                    # list of (2,2,P) states (posterior samples) or one
    if isinstance(mom_list, np.ndarray) and mom_list.ndim==3: mom_list=[mom_list]
    sb=Sandbox(d); lim=d["request"]["divergence_limit"]; cap=d["request"]["capability_limit"]
    qs=[]
    for mom in mom_list:
        D,Lo=sb.run(o,mom,chunk=200)
        qs.append(np.where((Lo>cap).any(1),0,lim/np.maximum(D,lim)))
    qe=np.mean(qs,0)
    return i, q[int(np.argmax(qe))], q.max(), q.mean(), np.corrcoef(qe,q)[0,1] if qe.std()>0 else 0
def evaluate(fn, idx, name=""):
    global FN
    FN=fn; t=time.time()
    with Pool(4) as p: r=p.map(_work, list(idx), chunksize=2)
    r=np.array([x[1:] for x in r])
    print(f"{name:40s} pick {r[:,0].mean():.3f} | best {r[:,1].mean():.3f} rand {r[:,2].mean():.3f} corr {r[:,3].mean():.3f} | {time.time()-t:.0f}s", flush=True)
    return r
