"""Experiment 1: generative decode vs the reference nine-target GBDT, 5-fold OOF, exact metric."""
import sys, json, time, pickle
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, spiro, harness, decode as dec
import lightgbm as lgb

W=harness.W; D=harness.D; t0=time.time()
X=pd.read_csv(f'{W}/X_raw.csv', index_col=0); P=pd.read_csv(f'{W}/pool_raw.csv', index_col=0)
L=pd.read_csv(f'{W}/latent_R.csv'); train=pd.read_csv(f'{D}/train.csv')
Y=pd.DataFrame(list(train.target_json.apply(json.loads)))[spiro.TARGETS].values
accvals=pickle.load(open(f'{W}/accvals.pkl','rb'))
models=harness.pool_reference(P); XD=harness.design(X, models)
Xtr=XD.loc[train.id].values; names=list(XD.columns)
sessions=[dec.PreSession(*accvals[p]) for p in train.id]
print('[%.0fs] sessions built'%(time.time()-t0))

yf=np.log(L.Rf2.values); yv=np.log(L.Rv2.values)
strat=harness.strata(Y, train.n_acceptable.values); fold=harness.folds(strat,5,0)
PAR=dict(objective='l2',learning_rate=0.03,num_leaves=15,min_data_in_leaf=40,feature_fraction=0.5,
         bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=4,seed=1,
         deterministic=True,force_row_wise=True)

def oof(y,nr=500):
    o=np.zeros(len(y))
    for k in range(5):
        tr=fold!=k
        m=lgb.train(PAR, lgb.Dataset(Xtr[tr],y[tr],feature_name=names), num_boost_round=nr)
        o[fold==k]=m.predict(Xtr[fold==k])
    return o
of=oof(yf); ov=oof(yv)
print('[%.0fs] latent OOF done'%(time.time()-t0))

# residual law built CROSS-FITTED: for rows in fold k use residuals from the other folds
def decode_oof(of,ov,sf,sv):
    Pr=np.zeros((len(yf),9))
    for k in range(5):
        tr=fold!=k; va=fold==k
        law=dec.ResidualLaw(yf[tr]-of[tr], yv[tr]-ov[tr], sf, sv)
        idx=np.where(va)[0]
        for i in idx:
            Pr[i]=dec.decode_one(sessions[i], of[i], ov[i], law)
    return Pr

for sf,sv in [(1.0,1.0)]:
    Pr=decode_oof(of,ov,sf,sv)
    print('GENERATIVE DECODE scale(%.2f,%.2f):'%(sf,sv), {k:round(v,4) for k,v in spiro.official_score(Pr,Y).items()})
print('[%.0fs]'%(time.time()-t0))

# --- reference-style baseline: nine independent LightGBM regressors on the probabilities ---
Pb=np.zeros((len(yf),9))
for j in range(9):
    for k in range(5):
        tr=fold!=k
        m=lgb.train(PAR, lgb.Dataset(Xtr[tr],Y[tr,j],feature_name=names), num_boost_round=500)
        Pb[fold==k,j]=m.predict(Xtr[fold==k])
Pb=np.clip(Pb,0,1)
print('NINE-TARGET GBDT (reference analogue):', {k:round(v,4) for k,v in spiro.official_score(Pb,Y).items()})
np.save(f'{W}/oof_gen.npy',Pr); np.save(f'{W}/oof_gbdt.npy',Pb)
np.save(f'{W}/oof_latent.npy',np.column_stack([of,ov]))
print('[%.0fs] done'%(time.time()-t0))
