import numpy as np, lightgbm as lgb
z=np.load("vest_data.npz"); X,Y,G=z["X"],z["Y"],z["G"]
scenes=np.unique(G); rng=np.random.default_rng(0)
fold={s:i%3 for i,s in enumerate(rng.permutation(scenes))}; f=np.array([fold[g] for g in G])
for name,cols in [("all",list(range(X.shape[2]))),("no-req-scalars",list(range(10)))]:
    r2s=[];base=[]
    for k in range(3):
        tr=f!=k; te=f==k
        Xtr=X[tr][:,::3][:,:,cols].reshape(-1,len(cols)); ytr=Y[tr][:,::3].ravel()   # subsample params
        m=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,random_state=0).fit(Xtr,ytr)
        Xte=X[te][:,:,cols].reshape(-1,len(cols)); yte=Y[te].ravel(); p=m.predict(Xte)
        ss=np.sum((yte-yte.mean())**2); r2s.append(1-np.sum((yte-p)**2)/ss)
        # request-level scale error: mean of residual per request
        res=(yte-p).reshape(te.sum(),-1); base.append((res.mean(1).std(), res.std(1).mean()))
    print(name,"R2",np.round(r2s,3),"per-request mean-bias std / within-request resid std",np.round(base,3).tolist())
