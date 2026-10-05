import numpy as np, pickle, lightgbm as lgb
from common import *
exec(open('exp10.py').read().split("X,n2=derived(FE)")[0])
X,n2=derived(FE); TF,tn=pickle.load(open('/tmp/TF13.pkl','rb'))
def run(Xc,label,rounds=400,seed=0):
    Xf=Xc.reshape(N,36,-1); oof=np.zeros((N,36))
    params=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_data_in_leaf=40,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5,verbose=-1,seed=seed,num_threads=4,deterministic=True)
    for f in range(5):
        tri=fold!=f; vai=fold==f
        m=lgb.train(params,lgb.Dataset(Xf[tri].reshape(-1,Xf.shape[-1]),yy[tri].ravel()),num_boost_round=rounds)
        oof[vai]=m.predict(Xf[vai].reshape(-1,Xf.shape[-1]),raw_score=True).reshape(-1,36)
    S=oof.reshape(N,6,6)
    print(label,'argmax',round(score(S.argmax(2),Y),4),'hung',round(score(np.array([hungarian(s) for s in S]),Y),4),flush=True)
    return S,m
yy=np.zeros((N,6,6))
for i in range(N): yy[i,np.arange(6),Y[i]]=1
yy=yy.reshape(N,36)
run(TF,'traits only')
S,m=run(np.concatenate([X,TF],-1),'artist-cf + traits')
imp=sorted(zip(m.feature_importance('gain'),n2+tn),reverse=True)[:12]; print([(n,int(g)) for g,n in imp])
