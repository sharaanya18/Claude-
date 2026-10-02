from harness import *
import gp2
idx=val_idx(2)
def stats(i):
    d=load_request(tr.iloc[i]); M=d["moments"]; vb=M[:,1].mean(0); sv=np.sqrt(vb)
    Jc=gp2.prep(d); u=M[:,0]/sv[None]; r=u@Jc.T   # (2,208)
    rho=np.dot(r[0],r[1])/np.linalg.norm(r[0])/np.linalg.norm(r[1])
    rel=np.linalg.norm(r[0]-r[1])/np.linalg.norm(r[0]+r[1])*2
    out=[rho,rel]
    for s2 in [0.0,1.0,30.0]:
        m,_=gp2.gp_samples(d,vb,1.0,s2,0.3,0,Jc=Jc); uh=m[:,0]/sv[None]; rh=uh@Jc.T
        res=[]
        for rows,name in [(np.arange(0,128),"batch"),(np.arange(128,176),"mon"),(np.arange(176,208),"diag")]:
            res.append(1-np.sum((rh[:,rows]-r[:,rows])**2)/np.sum(r[:,rows]**2))
        # difference response A-B on batch images
        dr=r[0]-r[1]; drh=rh[0]-rh[1]
        res.append(1-np.sum((drh[:128]-dr[:128])**2)/np.sum(dr[:128]**2))
        out+=res
    return out
if __name__=="__main__":
    out=np.array([stats(i) for i in idx]); print(np.round(out.mean(0),3))
    print("cols: rho,relAB | for s2 in 0,1,30: R2 batch, mon, diag, diffAB(batch)")
