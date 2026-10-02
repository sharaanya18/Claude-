from harness import *
import gp2
idx=val_idx(8)
def mk(s1,s2,sig,ns,vmode="oracle"):
    def f(d):
        vbar=d["moments"][:,1].mean(0)
        m,ss=gp2.gp_samples(d,vbar,s1,s2,sig,ns,seed=int(d["weights"][0]*1e6)%1000)
        return [m]+ss if ns>0 else m
    return f
if __name__=="__main__":
    for (s1,s2,sig,ns) in [(1,1,0.3,0),(1,1,0.3,6),(1,1,0.3,16)]:
        evaluate(mk(s1,s2,sig,ns), idx, f"GP oracle-v s1={s1} s2={s2} sig={sig} ns={ns}")
