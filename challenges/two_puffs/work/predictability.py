import sys, json, time, pickle
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, spiro, boot, harness
import lightgbm as lgb

W=harness.W; D=harness.D
t0=time.time()
X=pd.read_csv(f'{W}/X_raw.csv', index_col=0)
P=pd.read_csv(f'{W}/pool_raw.csv', index_col=0)
L=pd.read_csv(f'{W}/latent_R.csv')
train=pd.read_csv(f'{D}/train.csv'); test=pd.read_csv(f'{D}/test.csv')
Y=pd.DataFrame(list(train.target_json.apply(json.loads)))[spiro.TARGETS].values

models=harness.pool_reference(P)
XD=harness.design(X, models)
Xtr=XD.loc[train.id].values; Xte=XD.loc[test.id].values
names=list(XD.columns)
print('design', Xtr.shape, '[%.0fs]'%(time.time()-t0))

yf=np.log(L.Rf2.values); yv=np.log(L.Rv2.values)
strat=harness.strata(Y, train.n_acceptable.values)
fold=harness.folds(strat, 5, 0)

PAR=dict(objective='l2', learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
         feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=5.0, verbose=-1, num_threads=4, seed=1, deterministic=True,
         force_row_wise=True)

def oof_fit(y, nrounds=500):
    o=np.zeros(len(y)); imp=np.zeros(Xtr.shape[1])
    for k in range(5):
        tr=fold!=k; va=fold==k
        ds=lgb.Dataset(Xtr[tr], y[tr], feature_name=names)
        m=lgb.train(PAR, ds, num_boost_round=nrounds)
        o[va]=m.predict(Xtr[va]); imp+=m.feature_importance('gain')
    return o, imp

for nm,y in (('logRf',yf),('logRv',yv)):
    o,imp=oof_fit(y)
    ss=1-np.var(y-o)/np.var(y)
    print('%s: OOF R2=%.4f  corr=%.4f  sd(y)=%.4f sd(resid)=%.4f'%(nm,ss,np.corrcoef(y,o)[0,1],y.std(),(y-o).std()))
    top=np.argsort(-imp)[:20]
    print('  top feats:', [names[i] for i in top])
print('[%.0fs]'%(time.time()-t0))
