import sys; sys.argv=['x','dataset/public','working/x.csv']
import numpy as np, pandas as pd, json
import solution_commented as S
from common import hungarian
L=pd.read_csv('dataset/public/listens.csv').sort_values(['session','relative_position']).reset_index(drop=True)
C=pd.read_csv('dataset/public/continuations.csv'); tr=pd.read_csv('dataset/public/train.csv'); te=pd.read_csv('dataset/public/test.csv')
for d in (tr,te): d['pre']=d.prefixes.map(json.loads); d['cand']=d.candidates.map(json.loads)
Y=tr[[f'match_{i}' for i in range(1,7)]].to_numpy()-1
D=S.Data(L,C,set(x for l in te.pre for x in l)); fit=~D.is_test
oof=np.load('/tmp/oof33.npz')['mlp_base']; pred=np.array([hungarian(z) for z in oof]); ok=(pred==Y)
Xb=D.Ca[fit].copy(); Xb.data[:]=1; df=np.asarray(Xb.sum(0)).ravel()
X1b=D.C1[fit].copy(); X1b.data[:]=1; df1=np.asarray(X1b.sum(0)).ravel()
rows=[]
for i,(_,r) in enumerate(tr.iterrows()):
    for j,x in enumerate(r.pre):
        s=D.sid[x]; a,b=D.span[s]; k=a+int((D.h[a:b]==1).sum()); arts=np.unique(D.a[a:k])
        c=r.cand[Y[i,j]]; b1,b2=D.cc[c]
        rows.append(dict(ok=ok[i,j],n1=k-a,uniq=len(arts),known=(df[arts]>=2).mean(),cdf=min(df[b1],df[b2]),cdf1=df[b1],row=i))
R=pd.DataFrame(rows)
print('overall acc',R.ok.mean().round(3))
R['cdf_b']=pd.cut(R.cdf,[-1,2,5,15,50,1e9]); R['n1_b']=pd.qcut(R.n1,4); R['known_b']=pd.cut(R.known,[-.01,.5,.75,.9,1.0])
for c in ('cdf_b','n1_b','known_b'):
    print(R.groupby(c,observed=True).ok.agg(['mean','size']).round(3).to_string()); print()
# row-level: how many rows are all-correct / all-wrong
ra=R.groupby('row').ok.mean(); print('row acc distribution:',ra.value_counts(bins=[-.01,.2,.4,.6,.8,1.0]).sort_index().to_dict())
