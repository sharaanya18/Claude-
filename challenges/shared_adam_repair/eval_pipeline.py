"""Scene-grouped end-to-end evaluation of solution.py's controller (dev only)."""
import sys, time; sys.argv=["x","dataset/public","working/x.csv"]
import numpy as np, pandas as pd
import solution as S
S.DEVICE="cpu"
train=pd.read_csv(S.PUBLIC_DIR/"train.csv")
scenes=sorted(train["experiment_group"].unique()); rng_pool=np.random.default_rng(S.SEED)
pool=np.stack([rng_pool.permutation(8) for _ in range(S.N_POOL)])
res=[]
for k in range(4):
    held=scenes[k::4]; fit_rows=train[~train["experiment_group"].isin(held)]
    vm=S.fit_v_model(fit_rows.iloc[::2]); t=time.time()
    for g in held:
        sub=train[train["experiment_group"]==g]; rows=sub.iloc[np.linspace(3,len(sub)-1,12).astype(int)]
        for n,(_,row) in enumerate(rows.iterrows()):
            d=S.load_request(row,True); vb=np.exp(vm.predict(S.v_features(d,S.per_image_grads(d))))
            sb=S.Sandbox(d); qtrue,_=sb.quality(pool[:512],d["moments"])
            o_full=S.choose_order(d,vb,0.1,pool,np.random.default_rng(n),full=True)
            o_mean=S.choose_order(d,vb,0.1,pool,np.random.default_rng(n),full=False)
            qf=sb.quality(o_full[None],d["moments"])[0][0]; qm=sb.quality(o_mean[None],d["moments"])[0][0]
            qc=sb.quality(np.arange(8)[None],d["moments"])[0][0]
            res.append((k,g,qf,qm,qc,qtrue.mean()))
    r=np.array([x[2:] for x in res if x[0]==k],float)
    print(f"fold {k}: full {r[:,0].mean():.3f} mean-only {r[:,1].mean():.3f} canonical {r[:,2].mean():.3f} random {r[:,3].mean():.3f}  ({time.time()-t:.0f}s)",flush=True)
df=pd.DataFrame(res,columns=["fold","scene","full","mean","canon","rand"])
print(df.groupby("scene")[["full","mean","canon","rand"]].mean().round(3))
print("OVERALL (mean over scenes):",df.groupby("scene")[["full","mean","canon","rand"]].mean().mean().round(3).to_dict())
df.to_csv("eval_pipeline.csv",index=False)
