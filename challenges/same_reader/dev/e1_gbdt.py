import sys; sys.path.insert(0,'dev')
from common import *
import lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
tr,te=load()
EM=sorted(set(tr.emotion_a)|set(tr.emotion_b)|set(te.emotion_a)|set(te.emotion_b))
def feats(df):
    X=pd.DataFrame(index=df.index)
    X['cnt']=df.annotator_count.values
    ia=df.emotion_a.map({e:i for i,e in enumerate(EM)}); ib=df.emotion_b.map({e:i for i,e in enumerate(EM)})
    X['a']=pd.Categorical(ia,categories=range(len(EM))); X['b']=pd.Categorical(ib,categories=range(len(EM)))
    # multi-hot of the pair (symmetric) so trees can share info across positions
    for i,e in enumerate(EM): X['has_'+e]=((ia==i)|(ib==i)).astype(int).values
    t=df.text.astype(str)
    X['len']=t.str.len().values; X['nw']=t.str.split().str.len().values
    X['q']=t.str.count(r'\?').values; X['ex']=t.str.count('!').values; X['name']=t.str.count(r'\[NAME\]').values
    X['up']=t.apply(lambda s: sum(c.isupper() for c in s)/max(1,len(s))).values
    return X
def mk(params,rounds,use_feats='all'):
    def fp(trd,y,ted):
        Xt,Xe=feats(trd),feats(ted)
        if use_feats=='struct': 
            c=[c for c in Xt.columns if c in('cnt','a','b') or c.startswith('has_')]; Xt,Xe=Xt[c],Xe[c]
        m=lgb.LGBMClassifier(**params,n_estimators=rounds,random_state=42,deterministic=True,n_jobs=4,verbose=-1)
        m.fit(Xt,y); return m.predict_proba(Xe)[:,1]
    return fp
P=dict(learning_rate=0.03,num_leaves=8,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=5,cat_smooth=20,min_data_per_group=40,cat_l2=10)
for uf in ('struct','all'):
  for r in (150,300):
    print(uf,r); cv(mk(P,r,uf),tr)
