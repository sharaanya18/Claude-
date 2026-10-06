import sys; sys.path.insert(0,'/home/user/Claude-/challenges/chart_change_marks/exp')
import pickle
from common import *
import stage2 as T
import lightgbm as lgb
def prep(name):
    tr,X,Wd,M,marks,truth=load()
    oh,ob=pickle.load(open(f'working/oof_{name}.pkl','rb'))
    F=[];Y=[];G=[];CAND=[]
    for i,(h,b) in enumerate(zip(oh,ob)):
        cands,ft=T.cand_table(h,b,tr.iloc[i],Wd[i]); tx=np.array([m['x'] for m in truth[i]])
        for k,c in enumerate(cands): F.append(ft[k]); Y.append(float(len(tx)>0 and np.abs(tx-c[0]).min()<=5)); G.append(i)
        CAND.append(cands)
    return tr,truth,np.array(F),np.array(Y),np.array(G),CAND
def crossfit(tr,F,Y,G,**kw):
    fo=tr.fold.values[G]; P=np.zeros(len(Y))
    for f in range(5):
        m=lgb.LGBMClassifier(n_estimators=kw.get('n',250),learning_rate=0.03,num_leaves=kw.get('nl',15),min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5,verbose=-1,random_state=0,n_jobs=4)
        m.fit(F[fo!=f],Y[fo!=f],categorical_feature=[14]); P[fo==f]=m.predict_proba(F[fo==f])[:,1]
    return P
def build(tr,CAND,P,thr):
    out=[[] for _ in range(len(tr))]; j=0
    for i,c in enumerate(CAND):
        for k,(x,s,p) in enumerate(c):
            if P[j]>=thr: out[i].append({'x':round(x,2),'p':[float(v) for v in p]})
            j+=1
    return out
