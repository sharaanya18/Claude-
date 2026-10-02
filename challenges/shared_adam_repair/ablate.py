import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, time
from sim import *
from basis import *
tr = pd.read_csv(PUB/"train.csv"); lab = np.load("order_labels.npz")
rng = np.random.default_rng(0)
idx = rng.choice(len(tr), 24, replace=False)
def pick(d, mom, orders, N=150):
    qs=[]
    for o in orders[:N]:
        q,_,_ = quality(d, list(o), moments=mom); qs.append(q)
    return int(np.argmax(qs)), np.array(qs)
res = {}
for i in idx:
    d = load_request(tr.iloc[i]); M = d["moments"]; o = lab["orders"][i].astype(int); Dt = lab["D"][i]; lo = lab["losses"][i]
    lim = d["request"]["divergence_limit"]; cap = d["request"]["capability_limit"]
    qtrue = np.where((lo>cap).any(1), 0, lim/np.maximum(Dt, lim))
    N=150
    G = grad_basis(d); ols = ols_state(d, G)
    variants = {}
    vm = M.copy(); vm[:,1] = M[:,1].mean(0,keepdims=True); variants["v_common"] = vm
    mm = M.copy(); mm[:,0] = M[:,0].mean(0,keepdims=True); variants["m_common"] = mm
    mo = M.copy(); mo[:,0] = ols[:,0]; variants["m_ols"] = mo
    vo = M.copy(); vo[:,1] = ols[:,1]; variants["v_ols"] = vo
    bo = ols.copy(); variants["both_ols"] = bo
    # m: keep true projection on gradient span, drop remainder -> same as m_ols; instead keep ols + remainder shrunk
    for name, mom in variants.items():
        k, qs = pick(d, mom, o, N)
        res.setdefault(name, []).append(qtrue[k])
    res.setdefault("true", []).append(qtrue[:N].max()); res.setdefault("random", []).append(qtrue[:N].mean())
    print(i, {k: round(v[-1],3) for k,v in res.items()}, flush=True)
print({k: round(float(np.mean(v)),3) for k,v in res.items()})
