from harness import *
import gp3
idx=val_idx(8)
def mk(hp,ns=0):
    def fn(d,i):
        vb=d["moments"][:,1].mean(0)
        st=gp3.gp_joint(d,vb,hp,ns,seed=i); return st if ns else st[0]
    return fn
if __name__=="__main__":
    base=dict(rho=0.0,s2=1.0,sig=0.3,gs=(1,1,1,1))
    evaluate(mk(base),idx,"indep s2=1 (should be .532)")
    for rho in [0.3,0.6]: evaluate(mk({**base,"rho":rho}),idx,f"rho={rho}")
    for gs in [(1,10,10,10),(1,0.1,0.1,0.1),(1,1,10,1)]: evaluate(mk({**base,"gs":gs}),idx,f"gs={gs}")
