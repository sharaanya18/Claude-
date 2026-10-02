from harness import *
import gp4
from exp3 import vhat
idx=val_idx(8)
HP=dict(cbar=0.0,sc=0.04,s1=0.001,s2=0.0,sig=0.1)
def mk(ns,ss,sp):
    def fn(d,i):
        vh=vhat(d,i); Jc=gp4.prep(d); G=gp4.img_grads(d); rng=np.random.default_rng(i)
        sts=[gp4.gp_grad(d,vh,HP,0,Jc=Jc,G=G)[0]]
        for s in range(ns):
            vb=vh*np.exp(ss*rng.standard_normal()+sp*rng.standard_normal(L.P))
            sts.append(gp4.gp_grad(d,vb,HP,1,seed=1000*i+s,Jc=Jc,G=G)[1])
        return sts
    return fn
if __name__=="__main__":
    evaluate(mk(8,0.0,0.0),idx,"8 post samples, v fixed")
    evaluate(mk(8,0.35,0.3),idx,"8 samples, v perturbed (.35,.3)")
