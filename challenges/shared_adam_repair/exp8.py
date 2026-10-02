from harness import *
import gp4
idx=val_idx(8)
def mk(hp,ns=0):
    def fn(d,i):
        vb=d["moments"][:,1].mean(0); st=gp4.gp_grad(d,vb,hp,ns,seed=i); return st if ns else st[0]
    return fn
if __name__=="__main__":
    base=dict(cbar=0.04,sc=0.04,s1=1.0,s2=0.0,sig=0.1)
    evaluate(mk({**base,"cbar":0.0,"sc":0.0,"s2":1.0,"s1":1.0}),idx,"ref GP (s1=1,s2=1,sig=.1)")
    evaluate(mk({**base,"sc":0.0,"s1":0.0,"s2":0.0,"sig":0.1}) if False else mk({**base,"s1":0.01}),idx,"grad prior cbar=.04 sc=.04 s1=.01")
    evaluate(mk({**base,"s1":1.0}),idx,"grad prior cbar=.04 sc=.04 s1=1")
    evaluate(mk({**base,"cbar":0.0,"s1":0.01}),idx,"grad prior zero mean sc=.04 s1=.01")
    evaluate(mk({**base,"sc":0.02,"s1":0.01}),idx,"grad prior cbar=.04 sc=.02 s1=.01")
