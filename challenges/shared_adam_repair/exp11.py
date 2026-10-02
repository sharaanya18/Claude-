from harness import *
import gp4
idx=val_idx(8)
HP=dict(cbar=0.0,sc=0.04,s1=0.001,s2=0.0,sig=0.1)
def img_grads_at(d,w):
    X=np.concatenate([d["x_batches"].reshape(-1,192),d["x_monitor"],d["x_diag"].reshape(-1,192)])
    y=np.concatenate([d["batch_labels"].ravel(),d["monitor_labels"],d["diagnostic_labels"].ravel()])
    return np.stack([L.gradient(w,X[k:k+1],y[k:k+1]) for k in range(len(y))]).T
def mk(ks, hp=HP):
    def fn(d,i):
        vb=d["moments"][:,1].mean(0); w=d["weights"]; G0=gp4.img_grads(d)
        gm=np.mean([L.gradient(w,d["x_batches"][j],d["batch_labels"][j]) for j in range(8)],0)
        cols=[G0]
        for k in ks:
            w2=w-k*L.LR*gm/np.sqrt(vb); cols.append(img_grads_at(d,w2))
        return gp4.gp_grad(d,vb,hp,0,G=np.concatenate(cols,1))[0]
    return fn
if __name__=="__main__":
    evaluate(mk([8]),idx,"w0 + path k=8")
    evaluate(mk([4,8,16]),idx,"w0 + path k=4,8,16")
    evaluate(mk([8],dict(HP,sc=0.03)),idx,"w0+k8, sc=.03")
