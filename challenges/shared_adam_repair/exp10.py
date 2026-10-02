from harness import *
import gp4
idx=val_idx(8)
HP=dict(cbar=0.0,sc=0.04,s1=0.001,s2=0.0,sig=0.1)
def mk(cols,hp=HP,extra=None):
    def fn(d,i):
        vb=d["moments"][:,1].mean(0); G=gp4.img_grads(d)[:,cols]
        if extra: G=extra(d,G)
        return gp4.gp_grad(d,vb,hp,0,G=G)[0]
    return fn
if __name__=="__main__":
    evaluate(mk(slice(0,32)),idx,"batch images only")
    evaluate(mk(slice(0,44)),idx,"batch+monitor")
    evaluate(mk(np.r_[0:32,44:52]),idx,"batch+diag")
    # batch-level gradients (8) instead of per-image
    def bl(d,G): return np.stack([G[:,4*j:4*j+4].sum(1) for j in range(8)]+[G[:,k] for k in range(32,52)],1)
    evaluate(mk(slice(0,52),extra=bl),idx,"8 batch grads + monitor/diag imgs")
