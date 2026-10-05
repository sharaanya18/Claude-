import sys; sys.argv=['x','dataset/public','working/x.csv']
import numpy as np, pandas as pd, json, torch
import solution as S
d=np.load('/tmp/stack_cache.npz'); X=d['X']; ptr=d['ptr']
a=S.predict_mlp(S.fit_mlp(X[:800],ptr[:800]),X[800:]); b=S.predict_mlp(S.fit_mlp(X[:800],ptr[:800]),X[800:])
print('mlp identical:',np.array_equal(a,b),np.abs(a-b).max())
L=pd.read_csv('dataset/public/listens.csv').sort_values(['session','relative_position']).reset_index(drop=True)
C=pd.read_csv('dataset/public/continuations.csv'); tr=pd.read_csv('dataset/public/train.csv'); te=pd.read_csv('dataset/public/test.csv')
for dd in (tr,te): dd['pre']=dd.prefixes.map(json.loads); dd['cand']=dd.candidates.map(json.loads)
D=S.Data(L,C,set(x for l in te.pre for x in l)); fit=~D.is_test; R=tr.iloc[:60]
hk=S.HistKernel(D,fit); f1=S.lk_features(D,hk,R,True); f2=S.lk_features(D,S.HistKernel(D,fit),R,True)
print('lk feats identical:',np.array_equal(f1,f2))
Y=tr[[f'match_{i}' for i in range(1,7)]].to_numpy()-1
pid={tuple(p):i for i,p in enumerate(S.PERMS)}; pp=np.array([pid[tuple(y)] for y in Y[:60]])
m1=S.lk_fit(f1,pp); m2=S.lk_fit(f1,pp); print('lk_fit identical:',torch.equal(m1[0],m2[0]))
rng=np.random.RandomState(0); fit_idx=np.where(fit)[0]; bg=rng.choice(fit_idx[D.n1[fit_idx]>=8],500,replace=False)
g1=S.build_features(D,R,fit_idx,bg); g2=S.build_features(D,R,fit_idx,bg); print('build_features identical:',np.array_equal(g1,g2),np.abs(g1-g2).max())
rg=S.ReleaseGraph(D,fit); print('release identical:',np.array_equal(S.release_features(D,rg,R,True),S.release_features(D,rg,R,True)))
