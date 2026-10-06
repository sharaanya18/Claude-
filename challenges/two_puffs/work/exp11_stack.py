"""Experiment 11: cross-fitted second-stage stack on top of the generative decode.

Does anything remain in the features that the two-latent structure cannot express?  A
regularised GBDT per target on [features, decode outputs, phi, E[t], Var(t)] answers it.
Strictly cross-fitted: the stacker for fold k is trained only on the other folds' OOF rows.
"""
import sys, time, json
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, runner as R, spiro, harness
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

t0=time.time(); dat=R.load()
Y=dat['Y']; Xtr=dat['Xtr']; names=dat['names']
P=np.load(f'{R.W}/e8_P.npy'); Pd=np.load(f'{R.W}/e8_Pd.npy')
Et,Et2=np.load(f'{R.W}/oof_ats_moments.npy').T
phi=np.clip(4*(Et-Et2),0,1)
strat=harness.strata(Y, dat['train'].n_acceptable.values); fold=harness.folds(strat,5,0)
wgt=1.0+4.0*Y[:,8]*(1.0-Y[:,8])

Z=np.hstack([Xtr, P, Pd, phi[:,None], Et[:,None], (Et2-Et**2)[:,None]])
zn=names+['gen_%d'%i for i in range(9)]+['dir_%d'%i for i in range(9)]+['phi','Et','Vt']
Zs=np.hstack([P, Pd, phi[:,None], Et[:,None], (Et2-Et**2)[:,None]])
zs=['gen_%d'%i for i in range(9)]+['dir_%d'%i for i in range(9)]+['phi','Et','Vt']

def stack(Zin, zname, par, nr, tag):
    S=np.zeros_like(P)
    for j in range(9):
        for k in range(5):
            tr=fold!=k
            m=lgb.train(par, lgb.Dataset(Zin[tr],Y[tr,j],feature_name=list(zname),weight=wgt[tr]),
                        num_boost_round=nr)
            S[fold==k,j]=m.predict(Zin[fold==k])
    S=np.clip(S,0,1)
    out={}
    for w in [0.0,0.25,0.5,0.75,1.0]:
        Q=np.clip((1-w)*P+w*S,0,1); out[w]=spiro.official_score(Q,Y)['score']
    best=max(out,key=out.get)
    print('%-22s best w=%.2f score %.4f  | sweep %s [%.0fs]'%(tag,best,out[best],
          ' '.join('%.2f:%.4f'%(k,v) for k,v in out.items()),time.time()-t0), flush=True)
    return S

SOFT=dict(objective='l2',learning_rate=0.02,num_leaves=7,min_data_in_leaf=120,feature_fraction=0.4,
          bagging_fraction=0.7,bagging_freq=1,lambda_l2=50.0,verbose=-1,num_threads=4,seed=5,
          deterministic=True,force_row_wise=True)
S1=stack(Zs,zs,SOFT,300,'stack: preds only')
S2=stack(Z,zn,SOFT,300,'stack: preds+features')
np.save(f'{R.W}/e11_S1.npy',S1); np.save(f'{R.W}/e11_S2.npy',S2)
print('[%.0fs] done'%(time.time()-t0), flush=True)
