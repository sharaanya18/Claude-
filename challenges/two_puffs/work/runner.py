"""Reusable CV runner for the latent-distributional pipeline."""
import sys, json, time, pickle
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, spiro, harness, decode as dec, dist_decode as dd
import lightgbm as lgb

W=harness.W; D=harness.D

def load():
    X=pd.read_csv(f'{W}/X_raw.csv',index_col=0); P=pd.read_csv(f'{W}/pool_raw.csv',index_col=0)
    L=pd.read_csv(f'{W}/latent_R.csv'); train=pd.read_csv(f'{D}/train.csv'); test=pd.read_csv(f'{D}/test.csv')
    Y=pd.DataFrame(list(train.target_json.apply(json.loads)))[spiro.TARGETS].values
    accvals=pickle.load(open(f'{W}/accvals.pkl','rb'))
    models=harness.pool_reference(P); XD=harness.design(X,models)
    return dict(X=X,P=P,L=L,train=train,test=test,Y=Y,accvals=accvals,XD=XD,
                Xtr=XD.loc[train.id].values.astype(np.float64),
                Xte=XD.loc[test.id].values.astype(np.float64),
                names=list(XD.columns),
                sessions=[dec.PreSession(*accvals[p]) for p in train.id],
                sessions_te=[dec.PreSession(*accvals[p]) for p in test.id],
                yf=np.log(L.Rf2.values), yv=np.log(L.Rv2.values))

def uniform_ladder(lo=-0.30, hi=0.60, step=0.03):
    return np.round(np.arange(lo, hi+1e-9, step), 6)

def quantile_ladder(y, qp):
    g=np.maximum.accumulate(np.quantile(y,qp))
    for i in range(1,len(g)):
        if g[i]<=g[i-1]: g[i]=g[i-1]+1e-4
    return g

def pooled_fit(Xa, ya, grid, par, nround, names, cols=None):
    K=len(grid); n=len(ya)
    Z=np.hstack([np.repeat(Xa,K,axis=0), np.tile(grid,n)[:,None]])
    lab=(np.repeat(ya,K)>=np.tile(grid,n)).astype(np.float64)
    p=dict(par); p['monotone_constraints']=[0]*Xa.shape[1]+[-1]
    return lgb.train(p, lgb.Dataset(Z,lab,feature_name=list(names)+['level']), num_boost_round=nround)

def pooled_pred(m, Xa, grid):
    K=len(grid); n=len(Xa)
    Z=np.hstack([np.repeat(Xa,K,axis=0), np.tile(grid,n)[:,None]])
    return m.predict(Z).reshape(n,K)

def pit(S, grid, y, ramp):
    out=np.empty(len(y))
    for i in range(len(y)):
        xs,ys=dd.survival_curve(grid,S[i],ramp)
        out[i]=1.0-np.interp(y[i],xs,ys)
    return np.clip(out,1e-4,1-1e-4)

def run_cv(dat, gf, gv, par, nround, seed=0, n_splits=5, ramp=0.25, ret_models=False):
    Y=dat['Y']; Xtr=dat['Xtr']; yf=dat['yf']; yv=dat['yv']; names=dat['names']
    strat=harness.strata(Y, dat['train'].n_acceptable.values)
    fold=harness.folds(strat, n_splits, seed)
    Sf=np.zeros((len(yf),len(gf))); Sv=np.zeros((len(yv),len(gv)))
    for k in range(n_splits):
        tr=fold!=k; va=fold==k
        mf=pooled_fit(Xtr[tr],yf[tr],gf,par,nround,names)
        mv=pooled_fit(Xtr[tr],yv[tr],gv,par,nround,names)
        Sf[va]=pooled_pred(mf,Xtr[va],gf); Sv[va]=pooled_pred(mv,Xtr[va],gv)
    rho=float(np.corrcoef(dd.norm_ppf(pit(Sf,gf,yf,ramp)), dd.norm_ppf(pit(Sv,gv,yv,ramp)))[0,1])
    C=dd.Copula(rho)
    Pr=np.array([dd.decode_participant(dat['sessions'][i],gf,Sf[i],gv,Sv[i],C,
                 spiro.FEV1_THR,spiro.FVC_THR,ramp) for i in range(len(yf))])
    out=dict(score=spiro.official_score(Pr,Y), rho=rho, fold=fold, Sf=Sf, Sv=Sv, P=Pr)
    return out

def fold_scores(P, Y, fold, n_splits=5):
    return np.array([spiro.official_score(P[fold==k],Y[fold==k])['score'] for k in range(n_splits)])

PAR_BASE=dict(objective='binary',learning_rate=0.04,num_leaves=15,min_data_in_leaf=340,
              feature_fraction=0.45,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,
              verbose=-1,num_threads=4,seed=1,deterministic=True,force_row_wise=True)
