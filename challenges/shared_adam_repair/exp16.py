from harness import *
import gp4
from exp11 import img_grads_at
idx=val_idx(8)
HP=dict(cbar=0.0,sc=0.04,s1=0.001,s2=0.0,sig=0.1)
def mk(ks, c, hp=HP, scales=None):
    def fn(d,i):
        vb=d["moments"][:,1].mean(0); w=d["weights"]; G0=gp4.img_grads(d)
        gm=np.mean([L.gradient(w,d["x_batches"][j],d["batch_labels"][j]) for j in range(8)],0)
        direction=c*gm/(np.sqrt(vb)+1e-8)
        cols=[G0]
        for k in ks:
            cols.append(img_grads_at(d, w-k*L.LR*direction))
        return gp4.gp_grad(d,vb,hp,0,G=np.concatenate(cols,1))[0]
    return fn
if __name__=="__main__":
    evaluate(mk([],0.25),idx,"baseline w0 basis (oracle v)")
    evaluate(mk([4,8,12],0.25),idx,"traj k=4,8,12 c=.25")
    evaluate(mk([4,8,12],0.5),idx,"traj k=4,8,12 c=.5")
    evaluate(mk([8,16],0.25),idx,"traj k=8,16 c=.25")
