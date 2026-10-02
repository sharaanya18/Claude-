from harness import *
import gp4
from exp3 import vhat
idx=val_idx(2)   # 24 requests
HP=dict(cbar=0.0,sc=0.04,s1=0.001,s2=0.0,sig=0.1)
NC=4000
def work2(i):
    torch.set_num_threads(1)
    d=load_request(tr.iloc[i]); rng=np.random.default_rng(i); o=np.array([rng.permutation(8) for _ in range(NC)])
    lim=d["request"]["divergence_limit"]; cap=d["request"]["capability_limit"]; sb=Sandbox(d)
    def qof(mom):
        D,Lo=sb.run(o,mom,chunk=250); return np.where((Lo>cap).any(1),0,lim/np.maximum(D,lim))
    qt=qof(d["moments"]); vb=vhat(d,i)
    qe=qof(gp4.gp_grad(d,vb,HP,0)[0])
    out=[]
    for n in [100,400,1000,4000]:
        out.append(qt[np.argmax(qe[:n])])
    return out+[qt[:400].mean(), qt.max()]
if __name__=="__main__":
    with Pool(4) as p: r=np.array(p.map(work2,idx,chunksize=1))
    print("N=100,400,1000,4000 picks:",np.round(r[:,:4].mean(0),3),"random",r[:,4].mean().round(3),"best of 4000",r[:,5].mean().round(3))
