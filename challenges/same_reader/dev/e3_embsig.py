import sys; sys.path.insert(0,'dev')
from common import *
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
exec(open('dev/e0_orig.py').read().split("tr,te=load()")[0].split("def cats")[1].join(["def cats",""])) if False else None
tr,te=load(); tag=sys.argv[1]; E=np.load(f'dev/emb_{tag}_tr.npy'); E=E/np.linalg.norm(E,axis=1,keepdims=True)
tr=tr.reset_index(drop=True); tr['i']=np.arange(len(tr))
def cats(df):
    cnt=df.annotator_count.astype(str); a,b=df.emotion_a,df.emotion_b
    return pd.DataFrame({"cnt":cnt,"a":a,"b":b,"ca":cnt+"_"+a,"cb":cnt+"_"+b,"pair":a+"|"+b})
for W in (0.5,1,2):
  for C in (0.1,0.3):
    def fp(trd,y,ted):
        enc=OneHotEncoder(handle_unknown='ignore'); xt=hstack([enc.fit_transform(cats(trd)),csr_matrix(W*E[trd.i.values])]).tocsr()
        xe=hstack([enc.transform(cats(ted)),csr_matrix(W*E[ted.i.values])]).tocsr()
        return LogisticRegression(C=C,max_iter=3000).fit(xt,y).predict_proba(xe)[:,1]
    print(tag,'W',W,'C',C); cv(fp,tr)
