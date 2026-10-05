"""Cache stacker inputs (OOF features) from solution.py's own functions for fast stacker iteration."""
import sys; sys.argv=['x','dataset/public','working/x.csv']
import numpy as np, pandas as pd, json, time
import solution_commented as S
L=pd.read_csv('dataset/public/listens.csv').sort_values(['session','relative_position']).reset_index(drop=True)
C=pd.read_csv('dataset/public/continuations.csv'); tr=pd.read_csv('dataset/public/train.csv'); te=pd.read_csv('dataset/public/test.csv')
for d in (tr,te): d['pre']=d.prefixes.map(json.loads); d['cand']=d.candidates.map(json.loads)
Y=tr[[f'match_{i}' for i in range(1,7)]].to_numpy()-1
D=S.Data(L,C,set(x for l in te.pre for x in l)); rng=np.random.RandomState(S.SEED); fold=rng.randint(0,S.N_FOLDS,len(tr)); ffold=rng.randint(0,S.N_FEAT_FOLDS,len(tr)); Xtr=None
for f in range(S.N_FEAT_FOLDS):
    R=tr[ffold==f]; held=np.zeros(D.NS,bool)
    for l in R.pre:
        for x in l: held[D.sid[x]]=True
    fit_idx=np.where(~D.is_test&~held)[0]; bg=rng.choice(fit_idx[D.n1[fit_idx]>=8],S.N_BG,replace=False)
    F=S.build_features(D,R,fit_idx,bg)
    if Xtr is None: Xtr=np.zeros((len(tr),)+F.shape[1:],np.float32)
    Xtr[ffold==f]=F
fit_all=~D.is_test; hk=S.HistKernel(D,fit_all); rg=S.ReleaseGraph(D,fit_all)
LK=S.lk_features(D,hk,tr,True); RL=S.release_features(D,rg,tr,True)
perm_id={tuple(p):i for i,p in enumerate(S.PERMS)}; ptr=np.array([perm_id[tuple(y)] for y in Y])
Z=np.zeros((len(tr),6,6),np.float32)
for f in range(S.N_FOLDS): Z[fold==f]=S.lk_scores(S.lk_fit(LK[fold!=f],ptr[fold!=f]),LK[fold==f])
X=np.concatenate([Xtr,np.stack([np.concatenate([Z[i][...,None]]+[v[...,None] for v in S.rowwise_all(Z[i])]+[RL[i]],-1) for i in range(len(tr))])],-1).astype(np.float32)
np.savez('/tmp/stack_cache_v7.npz',X=X,ptr=ptr,fold=fold,Y=Y); print(X.shape)
