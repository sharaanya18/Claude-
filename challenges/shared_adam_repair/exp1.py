from harness import *
from functools import partial
from ablate2 import logit_jac, image_set
idx=val_idx(8)
def f_oracle_v(d, s2=1.0, sig=0.3):
    M=d["moments"]; vbar=M[:,1].mean(0); U=gp1.gp_state(d,vbar,s2,sig); out=M.copy()
    for b in range(2): out[b,0]=U[b,0]*np.sqrt(vbar); out[b,1]=vbar
    return out
if __name__=="__main__":
    r0=evaluate(lambda d: d["moments"], idx, "true state")
    r1=evaluate(f_oracle_v, idx, "GP oracle vbar s2=1 sig=.3")
