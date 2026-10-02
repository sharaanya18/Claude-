from harness import *
import gp2, iekf, mapfit
idx=val_idx(8)
def mk(sig,n_iter,s2=1.0):
    def fn(d,i):
        vb=d["moments"][:,1].mean(0); Jc=gp2.prep(d)
        u=iekf.iekf(d,vb,Jc,dict(s1=1.0,s2=s2,sig=sig),n_iter=n_iter)
        return mapfit.to_state(u,vb)
    return fn
if __name__=="__main__":
    evaluate(mk(0.3,1),idx,"IEKF 1 iter sig=.3 (linear exact-H)")
    evaluate(mk(0.3,4),idx,"IEKF 4 iter sig=.3")
    evaluate(mk(0.1,4),idx,"IEKF 4 iter sig=.1")
