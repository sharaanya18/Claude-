import sys; sys.path.insert(0,'.'); sys.argv=['x','dataset/public','/tmp/claude-0/-home-user-Claude-/a41a6ca1-31b6-5346-821d-5b7116a4627e/scratchpad/w/s.csv']
import solution as S, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP
S.NUM_THREADS=2
tr,te=S.load_data(); y=tr.p_same_reader.values.astype(int); folds=S.make_folds(tr)
emo=sorted(set(tr.emotion_a)|set(tr.emotion_b)); E=np.load('dev/emb_all-MiniLM-L6-v2_tr.npy'); E/=np.linalg.norm(E,axis=1,keepdims=True)
oof={n:np.zeros(len(tr)) for n in ['tfidf_lr','emb_lr','lgbm']}
for a,b in folds:
    oof['tfidf_lr'][b]=S.fit_predict_tfidf(tr.iloc[a],y[a],tr.iloc[b])
    oof['emb_lr'][b]=S.fit_predict_emb(tr.iloc[a],y[a],tr.iloc[b],E[a],E[b])
    p=S.fit_predict_lgb(tr.iloc[a],y[a],tr.iloc[b],emo); oof['lgbm'][b]=np.log(p/(1-p))
for n,v in oof.items(): print(n,np.mean([AP(y[b],v[b]) for _,b in folds]).round(4), round(AP(y,v),4))
z=np.column_stack([(v-v.mean())/v.std() for v in oof.values()]); print('equal blend', np.mean([AP(y[b],z[b].mean(1)) for _,b in folds]).round(4))
np.save('dev/classical_oof.npy',z)
