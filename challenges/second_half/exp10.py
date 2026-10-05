import numpy as np, pickle, lightgbm as lgb
from common import *
FE,names,fold=pickle.load(open('/tmp/FE9.pkl','rb'))
L,C,tr,te=load(); Y=np.array(list(tr.y)); N=len(tr)
ix={n:i for i,n in enumerate(names)}
def derived(FE):
    g=lambda n:FE[...,ix[n]]
    base={}
    for q in (0,1):
        base[f'la1_{q}']=np.log(g(f'a1_{q}')+.01); base[f'la2_{q}']=np.log(g(f'a2_{q}')+.01); base[f'lr1_{q}']=np.log(g(f'r1_{q}')+.01)
        base[f'lt_{q}']=np.log(g(f't10_{q}')*1+1e-3)
        base[f'cz_{q}']=(g(f'a1_{q}')-g(f'bmu_{q}'))/g(f'bsd_{q}')
    base['la1m']=(base['la1_0']+base['la1_1'])/2; base['lr1m']=(base['lr1_0']+base['lr1_1'])/2; base['ltm']=(base['lt_0']+base['lt_1'])/2; base['czm']=(base['cz_0']+base['cz_1'])/2
    base['la2m']=(base['la2_0']+base['la2_1'])/2
    cols=dict(base); 
    for k in ['la1m','lr1m','ltm','czm','la2m']:
        v=base[k]
        for ax,nm in ((1,'col'),(2,'row')):   # ax=1: across prefixes (for each cand), ax=2: across cands
            m=v.mean(ax,keepdims=True); s=v.std(ax,keepdims=True)+1e-6
            cols[f'{k}_{nm}z']=(v-m)/s
            cols[f'{k}_{nm}rk']=np.argsort(np.argsort(v,ax),ax)
    names2=list(cols); X=np.stack([cols[n] for n in names2],-1)
    return X,names2
X,n2=derived(FE); print(X.shape)
Xf=X.reshape(N,36,-1); yy=np.zeros((N,6,6)); 
for i in range(N): yy[i,np.arange(6),Y[i]]=1
yy=yy.reshape(N,36)
oof=np.zeros((N,36))
params=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_data_in_leaf=40,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5,verbose=-1,seed=0,num_threads=4,deterministic=True)
for f in range(5):
    tri=fold!=f; vai=fold==f
    d=lgb.Dataset(Xf[tri].reshape(-1,Xf.shape[-1]),yy[tri].ravel())
    m=lgb.train(params,d,num_boost_round=400)
    oof[vai]=m.predict(Xf[vai].reshape(-1,Xf.shape[-1]),raw_score=True).reshape(-1,36)
S=oof.reshape(N,6,6)
print('argmax',score(S.argmax(2),Y),'hung',score(np.array([hungarian(s) for s in S]),Y))
print('baseline la1m hung',score(np.array([hungarian(s) for s in X[...,n2.index('la1m')]]),Y))
imp=sorted(zip(m.feature_importance('gain'),n2),reverse=True)[:12]; print([(n,int(g)) for g,n in imp])
