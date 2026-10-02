from harness import *
import gp2, lightgbm as lgb
from vest import v_features
idx=val_idx(8)
z=np.load("vest_data.npz"); X,Y,Gs=z["X"],z["Y"],z["G"]
scenes=np.unique(Gs); rng=np.random.default_rng(0)
fold={s:i%3 for i,s in enumerate(rng.permutation(scenes))}; f=np.array([fold[g] for g in Gs])
MODELS={}
for k in range(3):
    tr_=f!=k
    MODELS[k]=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,random_state=0,n_jobs=1).fit(X[tr_][:,::3].reshape(-1,X.shape[2]),Y[tr_][:,::3].ravel())
def vhat(d,i):
    k=fold[tr.experiment_group[i]]; return np.exp(MODELS[k].predict(v_features(d)))
def mk(s1,s2,sig,ns,vmode):
    def fn(d,i):
        M=d["moments"]; vb=M[:,1].mean(0) if vmode=="oracle" else vhat(d,i)
        m,ss=gp2.gp_samples(d,vb,s1,s2,sig,ns,seed=i)
        return [m]+ss if ns>0 else m
    return fn
if __name__=="__main__":
    evaluate(mk(1,1,0.3,0,"oracle"),idx,"oracle v mean-state")
    evaluate(mk(1,1,0.3,0,"lgb"),idx,"lgb v mean-state")
