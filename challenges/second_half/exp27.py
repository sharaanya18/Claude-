import sys; sys.argv=['x','dataset/public','working/x.csv']
import numpy as np, pandas as pd, json
import solution as S
L=pd.read_csv('dataset/public/listens.csv').sort_values(['session','relative_position']).reset_index(drop=True)
C=pd.read_csv('dataset/public/continuations.csv'); tr=pd.read_csv('dataset/public/train.csv'); te=pd.read_csv('dataset/public/test.csv')
for d in (tr,te): d['pre']=d.prefixes.map(json.loads); d['cand']=d.candidates.map(json.loads)
D=S.Data(L,C,set(x for l in te.pre for x in l)); fit=~D.is_test
hk=S.HistKernel(D,fit); LK=S.lk_features(D,hk,tr,True); np.save('/tmp/LK_counts.npy',LK); print(LK.shape)
