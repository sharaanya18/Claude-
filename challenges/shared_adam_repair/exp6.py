from harness import *
import gp2, mapfit, sys
idx=val_idx(8)
def mk(sig_rel,iters,s2=1.0):
    def fn(d,i):
        vb=d["moments"][:,1].mean(0); Jc=gp2.prep(d)
        m,_=gp2.gp_samples(d,vb,1.0,s2,0.3,0,Jc=Jc); u0=m[:,0]/np.sqrt(vb)[None]
        u=mapfit.map_fit(d,vb,u0,Jc,1.0,s2,sig_rel,iters)
        return mapfit.to_state(u,vb)
    return fn
if __name__=="__main__":
    for sr in [0.1,0.03]:
        evaluate(mk(sr,80),idx,f"nonlinear MAP sig_rel={sr}")
