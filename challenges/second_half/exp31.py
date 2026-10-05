import sys; sys.argv=['x','dataset/public','working/x.csv']
import numpy as np, pandas as pd, json
import solution_commented as S
L=pd.read_csv('dataset/public/listens.csv').sort_values(['session','relative_position']).reset_index(drop=True)
C=pd.read_csv('dataset/public/continuations.csv'); tr=pd.read_csv('dataset/public/train.csv'); te=pd.read_csv('dataset/public/test.csv')
for d in (tr,te): d['pre']=d.prefixes.map(json.loads); d['cand']=d.candidates.map(json.loads)
Y=tr[[f'match_{i}' for i in range(1,7)]].to_numpy()-1
D=S.Data(L,C,set(x for l in te.pre for x in l))
pos1=[];pos2=[];gap1=[]; same_rel_last=[]; first_new=[]
for _,r in tr.iterrows():
    for j,x in enumerate(r.pre):
        s=D.sid[x]; a,b=D.span[s]; k=a+int((D.h[a:b]==1).sum())
        c=r.cand[Y[_,j] if False else 0]
    for j,x in enumerate(r.pre):
        s=D.sid[x]; a,b=D.span[s]; k=a+int((D.h[a:b]==1).sum())
        c=r.cand[r[f'match_{j+1}']-1]; b1,b2=D.cc[c]
        h2=D.a[k:b]; first=set(D.a[a:k])
        p1=int(np.where(h2==b1)[0][0]); p2=int(np.where(h2==b2)[0][0])
        pos1.append(p1); pos2.append(p2)
        # first new artist in the second half (any, ignoring the >=3 rule)
        nw=[i for i,v in enumerate(h2) if v not in first]
        first_new.append(nw[0]==p1 if nw else False)
        gap1.append(L.seconds.values[k+p1]-L.seconds.values[k-1])
pos1=np.array(pos1); pos2=np.array(pos2)
print('rank-1 first play position in 2nd half: P(=0)',(pos1==0).mean().round(3),'P(<=2)',(pos1<=2).mean().round(3),'median',np.median(pos1))
print('rank-2 position: P(<=2)',(pos2<=2).mean().round(3),'median',np.median(pos2))
print('rank-1 is the first new artist of the second half:',np.mean(first_new).round(3))
print('time gap prefix end -> rank-1 first play (s): median',np.median(gap1),'P(<10min)',(np.array(gap1)<600).mean().round(3))
