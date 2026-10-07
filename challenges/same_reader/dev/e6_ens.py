import sys,os; sys.path.insert(0,'dev')
from common import *
tr,te=load(); y=tr.p_same_reader.values.astype(int); F=splits(tr,42,5)
done=[f for f in range(5) if os.path.exists(f'dev/ft2_distilroberta-base_f{f}_e1.npy')]
z=np.load('dev/classical_oof.npy'); ft=np.zeros(len(tr))
for f in done:
    p=np.load(f'dev/ft2_distilroberta-base_f{f}_e1.npy'); ft[F[f][1]]=np.log(p/(1-p))
idx=np.concatenate([F[f][1] for f in done]); ftz=(ft-ft[idx].mean())/ft[idx].std()
M=np.column_stack([z,ftz])
def fm(s): return np.mean([AP(y[F[f][1]],s[F[f][1]]) for f in done])
print('folds',done)
for i,n in enumerate(['tfidf','emb','lgbm','ft']): print(n,round(fm(M[:,i]),4))
print('classical eq',round(fm(M[:,:3].mean(1)),4))
for w in (0.25,0.33,0.4,0.5): print('ft weight',w,round(fm((1-w)*M[:,:3].mean(1)+w*ftz),4))
print('all eq',round(fm(M.mean(1)),4))
