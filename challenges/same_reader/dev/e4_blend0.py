import sys; sys.path.insert(0,'dev')
from common import *
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import rankdata
tr,te=load(); tr=tr.reset_index(drop=True); tr['i']=np.arange(len(tr)); y=tr.p_same_reader.values
E=np.load('dev/emb_all-MiniLM-L6-v2_tr.npy'); E/=np.linalg.norm(E,axis=1,keepdims=True)
def cats(df):
    cnt=df.annotator_count.astype(str); a,b=df.emotion_a,df.emotion_b
    return pd.DataFrame({"cnt":cnt,"a":a,"b":b,"ca":cnt+"_"+a,"cb":cnt+"_"+b,"pair":a+"|"+b})
a,b=splits(tr,42,5)[0]
enc=OneHotEncoder(handle_unknown='ignore'); xt=hstack([enc.fit_transform(cats(tr.iloc[a])),csr_matrix(0.5*E[a])]).tocsr(); xe=hstack([enc.transform(cats(tr.iloc[b])),csr_matrix(0.5*E[b])]).tocsr()
lr=LogisticRegression(C=0.3,max_iter=3000).fit(xt,y[a]).predict_proba(xe)[:,1]
print('LR',AP(y[b],lr))
r=lambda v: rankdata(v)/len(v)
for e in (1,2):
    ft=np.load(f'dev/ft_distilroberta-base_f0_e{e}.npy'); print('ft e',e,AP(y[b],ft))
    for w in (0.3,0.5): print(' blend w',w,AP(y[b],(1-w)*r(lr)+w*r(ft)))
