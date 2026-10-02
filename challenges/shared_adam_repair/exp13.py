from harness import *
import gp4
from exp3 import vhat
idx=val_idx(8)
HP=dict(cbar=0.0,sc=0.04,s1=0.001,s2=0.0,sig=0.1)
def mk(kind):
    def fn(d,i):
        vo=d["moments"][:,1].mean(0)
        if kind=="x2": vb=vo*2
        elif kind=="x0.5": vb=vo*0.5
        elif kind=="lgb_shape_oracle_scale":
            vh=vhat(d,i); vb=vh*np.exp(np.mean(np.log(vo))-np.mean(np.log(vh)))
        elif kind=="oracle_shape_lgb_scale":
            vh=vhat(d,i); vb=vo*np.exp(np.mean(np.log(vh))-np.mean(np.log(vo)))
        return gp4.gp_grad(d,vb,HP,0)[0]
    return fn
if __name__=="__main__":
    for k in ["x2","x0.5","lgb_shape_oracle_scale","oracle_shape_lgb_scale"]: evaluate(mk(k),idx,k)
