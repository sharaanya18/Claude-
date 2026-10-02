from harness import *
import gp4
from exp3 import vhat
idx=val_idx(8)
HP=dict(cbar=0.0,sc=0.04,s1=0.001,s2=0.0,sig=0.1)
def mk(ns,vmode,hp=HP):
    def fn(d,i):
        vb=d["moments"][:,1].mean(0) if vmode=="oracle" else vhat(d,i)
        st=gp4.gp_grad(d,vb,hp,ns,seed=i); return st if ns else st[0]
    return fn
if __name__=="__main__":
    evaluate(mk(0,"lgb"),idx,"grad prior, lgb v, mean")
    evaluate(mk(8,"lgb"),idx,"grad prior, lgb v, mean+8 samples")
