"""Value of information: score as a function of how well the latent response is predicted.

A noisy copy of the true latent is injected as an extra feature (deliberate, for the study only),
the real pipeline is refitted, and the achieved OOF latent R2 is reported next to the score.
This says whether chasing features can reach 0.75 or whether the task has a lower ceiling.
"""
import sys, time
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, runner as R, spiro, harness, dist_decode as dd
import lightgbm as lgb

t0=time.time(); dat=R.load()
QP=np.array([0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.875,0.925,0.96,0.98,0.99,0.995])
gf=R.quantile_ladder(dat['yf'],QP); gv=R.quantile_ladder(dat['yv'],QP)
Y=dat['Y']; yf=dat['yf']; yv=dat['yv']; Xtr=dat['Xtr']
strat=harness.strata(Y, dat['train'].n_acceptable.values); fold=harness.folds(strat,5,0)
rng=np.random.RandomState(2024)
sdf=0.1215; sdv=0.1967
print('noise  R2f    R2v    score   rcs    fds', flush=True)
for mult in [None, 0.9, 0.6, 0.4, 0.25, 0.12]:
    if mult is None:
        Xa=Xtr; names=dat['names']
    else:
        zf=yf+rng.randn(len(yf))*sdf*mult; zv=yv+rng.randn(len(yv))*sdv*mult
        Xa=np.hstack([Xtr, zf[:,None], zv[:,None]]); names=dat['names']+['zf','zv']
    Sf=np.zeros((len(yf),len(gf))); Sv=np.zeros((len(yv),len(gv)))
    o_f=np.zeros(len(yf)); o_v=np.zeros(len(yv))
    P2=dict(R.PAR_BASE)
    for k in range(5):
        tr=fold!=k; va=fold==k
        mf=R.pooled_fit(Xa[tr],yf[tr],gf,P2,400,names); mv=R.pooled_fit(Xa[tr],yv[tr],gv,P2,400,names)
        Sf[va]=R.pooled_pred(mf,Xa[va],gf); Sv[va]=R.pooled_pred(mv,Xa[va],gv)
        # plain l2 model just to report achieved latent R2 on the same features
        pl=dict(objective='l2',learning_rate=0.03,num_leaves=15,min_data_in_leaf=40,feature_fraction=0.5,
                bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=4,seed=1,
                deterministic=True,force_row_wise=True)
        o_f[va]=lgb.train(pl,lgb.Dataset(Xa[tr],yf[tr]),num_boost_round=500).predict(Xa[va])
        o_v[va]=lgb.train(pl,lgb.Dataset(Xa[tr],yv[tr]),num_boost_round=500).predict(Xa[va])
    r2f=1-np.var(yf-o_f)/np.var(yf); r2v=1-np.var(yv-o_v)/np.var(yv)
    rho=float(np.corrcoef(dd.norm_ppf(R.pit(Sf,gf,yf,0.25)), dd.norm_ppf(R.pit(Sv,gv,yv,0.25)))[0,1])
    C=dd.Copula(rho)
    Pr=np.array([dd.decode_participant(dat['sessions'][i],gf,Sf[i],gv,Sv[i],C,spiro.FEV1_THR,spiro.FVC_THR,0.25)
                 for i in range(len(yf))])
    s=spiro.official_score(Pr,Y)
    print('%-6s %.3f  %.3f  %.4f  %.4f  %.4f  [%.0fs]'%(str(mult),r2f,r2v,s['score'],s['rcs'],s['fds'],time.time()-t0), flush=True)
