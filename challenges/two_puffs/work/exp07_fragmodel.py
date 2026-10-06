"""Direct fragility model + minimal-Brier-cost ordering transfer.

FDS depends ONLY on the ordering of |p-0.5| (any symmetric monotone transform of p leaves it
unchanged), so the only way to move FDS is a genuinely better fragility signal.  We therefore
(a) learn the true fragility 4t(1-t) directly, and (b) impose its ordering on the submitted ats
with weighted isotonic regression, which is the cheapest possible way to buy that ordering in
Brier terms.
"""
import sys, time
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, runner as R, spiro, harness
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

t0=time.time(); dat=R.load()
Y=dat['Y']; Xtr=dat['Xtr']; names=dat['names']
P=np.load(f"{R.W}/oof_dist.npy"); Et,Et2=np.load(f"{R.W}/oof_ats_moments.npy").T
phi=np.clip(4*(Et-Et2),0,1)
tf=4*Y[:,8]*(1-Y[:,8])
strat=harness.strata(Y, dat['train'].n_acceptable.values); fold=harness.folds(strat,5,0)
wgt=1.0+4.0*Y[:,8]*(1.0-Y[:,8])

PL=dict(objective='l2',learning_rate=0.03,num_leaves=15,min_data_in_leaf=40,feature_fraction=0.5,
        bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=4,seed=1,
        deterministic=True,force_row_wise=True)

def oof_model(y, X, par=PL, nr=500):
    o=np.zeros(len(y))
    for k in range(5):
        tr=fold!=k
        o[fold==k]=lgb.train(par, lgb.Dataset(X[tr],y[tr],feature_name=names), num_boost_round=nr).predict(X[fold==k])
    return o

# (a) direct fragility models on the engineered features, and on features + the decode's own signals
f_direct=oof_model(tf, Xtr)
Xaug=np.hstack([Xtr, P, phi[:,None], Et[:,None], (Et2-Et**2)[:,None]])
names_aug=names+['dec_%d'%i for i in range(9)]+['phi','Et','Vt']
_n=names; names=names_aug
f_aug=oof_model(tf, Xaug)
names=_n
print('Somers D of true fragility vs ... [%.0fs]'%(time.time()-t0))
cands={'4p(1-p) decode':4*P[:,8]*(1-P[:,8]), 'phi=E[4t(1-t)]':phi,
       'direct LGB on tf':f_direct, 'direct LGB on tf + decode feats':f_aug,
       'mean(phi_rank, f_aug_rank)':0.5*(pd.Series(phi).rank().values+pd.Series(f_aug).rank().values)}
for nm,g in cands.items():
    d=spiro.somers_d(tf,g); print('  %-34s D=%.4f  FDS=%.4f'%(nm,d,(1+d)/2))

def impose_ordering(p, g, weights, eps=1e-6, nfold=5):
    """Give |p-0.5| the ordering of g at minimal weighted squared cost (cross-fitted isotonic)."""
    a=np.abs(p-0.5); side=np.where(p>=0.5,1.0,-1.0); m=np.zeros_like(a)
    for k in range(nfold):
        tr=fold!=k; va=fold==k
        iso=IsotonicRegression(increasing=False, out_of_bounds='clip')
        iso.fit(g[tr], a[tr], sample_weight=weights[tr])
        m[va]=iso.predict(g[va])
    gr=(g-g.min())/max(g.max()-g.min(),1e-9)
    m=np.clip(m-eps*gr, 0.0, 0.5)
    return np.clip(0.5+side*m, 0.0, 1.0)

print('\nordering transfer, blend sweep:')
for nm,g in cands.items():
    pbar=impose_ordering(P[:,8], g, wgt)
    row=[]
    for w in [0.0,0.25,0.5,0.75,1.0]:
        Q=P.copy(); Q[:,8]=np.clip((1-w)*P[:,8]+w*pbar,0,1)
        s=spiro.official_score(Q,Y); row.append((w,s['score'],s['rcs'],s['fds']))
    best=max(row,key=lambda r:r[1])
    print('  %-34s best w=%.2f score %.4f (rcs %.4f fds %.4f)'%(nm,best[0],best[1],best[2],best[3]))
    print('      full sweep: '+' '.join('w%.2f:%.4f'%(r[0],r[1]) for r in row))
np.save(f"{R.W}/oof_frag.npy", np.column_stack([f_direct,f_aug,phi]))
print('[%.0fs]'%(time.time()-t0))
