import sys; sys.path.insert(0,'dev'); sys.path.insert(0,'.')
from common import *
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
tr,te=load(); tr=tr.reset_index(drop=True); tr['i']=np.arange(len(tr))
E=np.load('dev/emb_all-MiniLM-L6-v2_tr.npy'); E/=np.linalg.norm(E,axis=1,keepdims=True)
N=np.load('dev/nli_nli-MiniLM2-L6-H768_tr.npy')  # n,2,3
def sm(x): x=x-x.max(-1,keepdims=True); e=np.exp(x); return e/e.sum(-1,keepdims=True)
P=sm(N); ent=P[:,:,1]; con=P[:,:,0]
F=np.column_stack([ent[:,0],ent[:,1],ent.min(1),ent.max(1),np.abs(ent[:,0]-ent[:,1]),ent.prod(1),con[:,0],con[:,1],con.max(1),N[:,0,1],N[:,1,1]])
def cats(df):
    cnt=df.annotator_count.astype(str); a,b=df.emotion_a,df.emotion_b
    return pd.DataFrame({"cnt":cnt,"a":a,"b":b,"ca":cnt+"_"+a,"cb":cnt+"_"+b,"pair":a+"|"+b})
def mk(use_emb,use_nli,W=0.5,C=0.3,NW=0.3):
    def fp(trd,y,ted):
        enc=OneHotEncoder(handle_unknown='ignore'); xt=[enc.fit_transform(cats(trd))]; xe=[enc.transform(cats(ted))]
        if use_emb: xt.append(csr_matrix(W*E[trd.i.values])); xe.append(csr_matrix(W*E[ted.i.values]))
        if use_nli:
            sc=StandardScaler().fit(F[trd.i.values]); xt.append(csr_matrix(NW*sc.transform(F[trd.i.values]))); xe.append(csr_matrix(NW*sc.transform(F[ted.i.values])))
        return LogisticRegression(C=C,max_iter=3000).fit(hstack(xt).tocsr(),y).predict_proba(hstack(xe).tocsr())[:,1]
    return fp
print('struct+emb'); cv(mk(1,0),tr)
for NW in (0.1,0.3,0.6):
    print('struct+emb+nli NW',NW); cv(mk(1,1,NW=NW),tr)
print('struct+nli only'); cv(mk(0,1),tr)
