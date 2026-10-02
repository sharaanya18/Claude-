from harness import *
import gp2
idx=val_idx(8)
GR=np.concatenate([np.zeros(2304),np.ones(12),np.full(48,2),np.full(4,3)]).astype(int)
def mix(keep_true_groups):
    def fn(d,i):
        M=d["moments"]; vb=M[:,1].mean(0); sv=np.sqrt(vb)
        m,_=gp2.gp_samples(d,vb,1.0,0.0,0.3,0)
        out=m.copy()
        sel=np.isin(GR,keep_true_groups)
        for b in range(2): out[b,0,sel]=M[b,0,sel]
        return out
    return fn
if __name__=="__main__":
    evaluate(mix([0]),idx,"true W1, GP rest")
    evaluate(mix([1,2,3]),idx,"true b1,W2,b2, GP W1")
    evaluate(mix([2,3]),idx,"true W2,b2, GP W1,b1")
