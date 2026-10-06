"""Experiment 2: conditional distributional latent model + exact bootstrap decode."""
import sys, json, time, pickle
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, spiro, harness, decode as dec, dist_decode as dd
import lightgbm as lgb

W=harness.W; D=harness.D; t0=time.time()
X=pd.read_csv(f'{W}/X_raw.csv',index_col=0); P=pd.read_csv(f'{W}/pool_raw.csv',index_col=0)
L=pd.read_csv(f'{W}/latent_R.csv'); train=pd.read_csv(f'{D}/train.csv')
Y=pd.DataFrame(list(train.target_json.apply(json.loads)))[spiro.TARGETS].values
accvals=pickle.load(open(f'{W}/accvals.pkl','rb'))
models=harness.pool_reference(P); XD=harness.design(X,models)
Xtr=XD.loc[train.id].values.astype(np.float64); names=list(XD.columns)
sessions=[dec.PreSession(*accvals[p]) for p in train.id]
yf=np.log(L.Rf2.values); yv=np.log(L.Rv2.values)
strat=harness.strata(Y, train.n_acceptable.values); fold=harness.folds(strat,5,0)

QP=np.array([0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.875,0.925,0.96,0.98,0.99,0.995])
PAR=dict(objective='binary',learning_rate=0.04,num_leaves=15,min_data_in_leaf=340,
         feature_fraction=0.45,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,
         verbose=-1,num_threads=4,seed=1,deterministic=True,force_row_wise=True)
NROUND=400

def ladder(y):
    g=np.quantile(y,QP)
    g=np.maximum.accumulate(g)
    for i in range(1,len(g)):                 # keep levels strictly increasing
        if g[i]<=g[i-1]: g[i]=g[i-1]+1e-4
    return g

def pooled_fit(Xa, ya, grid, nround=NROUND):
    K=len(grid); n=len(ya)
    Xb=np.repeat(Xa,K,axis=0)
    lev=np.tile(grid,n)[:,None]
    Z=np.hstack([Xb,lev])
    lab=(np.repeat(ya,K)>=np.tile(grid,n)).astype(np.float64)
    mc=[0]*Xa.shape[1]+[-1]                   # survival must be non-increasing in the level
    p=dict(PAR); p['monotone_constraints']=mc
    m=lgb.train(p, lgb.Dataset(Z,lab,feature_name=names+['level']), num_boost_round=nround)
    return m

def pooled_pred(m, Xa, grid):
    K=len(grid); n=len(Xa)
    Z=np.hstack([np.repeat(Xa,K,axis=0), np.tile(grid,n)[:,None]])
    return m.predict(Z).reshape(n,K)

gf=ladder(yf); gv=ladder(yv)
print('ladder f', np.round(gf,3)); print('ladder v', np.round(gv,3))

Sf=np.zeros((len(yf),len(gf))); Sv=np.zeros((len(yv),len(gv)))
for k in range(5):
    tr=fold!=k; va=fold==k
    mf=pooled_fit(Xtr[tr], yf[tr], gf); mv=pooled_fit(Xtr[tr], yv[tr], gv)
    Sf[va]=pooled_pred(mf,Xtr[va],gf); Sv[va]=pooled_pred(mv,Xtr[va],gv)
    print('  fold %d [%.0fs]'%(k,time.time()-t0))
np.save(f'{W}/oof_Sf.npy',Sf); np.save(f'{W}/oof_Sv.npy',Sv)
np.save(f'{W}/ladder.npy',np.vstack([gf,gv]))

# PIT-based conditional correlation for the copula (cross-fitted)
def pit(S, grid, y):
    out=np.empty(len(y))
    for i in range(len(y)):
        xs,ys=dd.survival_curve(grid,S[i],ramp=0.25)
        out[i]=1.0-np.interp(y[i],xs,ys)
    return np.clip(out,1e-4,1-1e-4)
zf=dd.norm_ppf(pit(Sf,gf,yf)); zv=dd.norm_ppf(pit(Sv,gv,yv))
rho=float(np.corrcoef(zf,zv)[0,1])
print('copula rho from PIT = %.3f'%rho)

for ramp in [0.15,0.25,0.4]:
    C=dd.Copula(rho)
    Pr=np.array([dd.decode_participant(sessions[i],gf,Sf[i],gv,Sv[i],C,spiro.FEV1_THR,spiro.FVC_THR,ramp)
                 for i in range(len(yf))])
    s=spiro.official_score(Pr,Y)
    print('DISTRIBUTIONAL DECODE ramp=%.2f'%ramp, {k:round(v,4) for k,v in s.items()})
    if ramp==0.25: np.save(f'{W}/oof_dist.npy',Pr)
print('[%.0fs]'%(time.time()-t0))
