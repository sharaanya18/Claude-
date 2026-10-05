import json, numpy as np, pandas as pd, sys
from sklearn.feature_extraction.text import TfidfVectorizer
P="dataset/public"
tr=pd.read_csv(f"{P}/train.csv")
rows=[]
for h,r in enumerate(tr.itertuples()):
    for e in json.loads(r.episodes): rows.append((h, r.interaction_kind, e["reference"]+" "+e["opening_user"]))
df=pd.DataFrame(rows,columns=["h","kind","topic"])
X=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,stop_words="english").fit_transform(df.topic)
S=(X@X.T).tocoo()
for th in [0.6,0.7,0.8]:
    par=list(range(len(tr)))
    def f(a):
        while par[a]!=a: par[a]=par[par[a]]; a=par[a]
        return a
    for i,j,v in zip(S.row,S.col,S.data):
        if i<j and v>th:
            a,b=f(df.h[i]),f(df.h[j])
            if a!=b: par[a]=b
    g=np.array([f(h) for h in range(len(tr))]); vc=pd.Series(g).value_counts()
    print("th",th,"groups",len(vc),"largest",vc.iloc[:6].tolist(),"singletons",(vc==1).sum(), "mixed-kind groups", sum(tr.interaction_kind[g==c].nunique()>1 for c in vc.index[:20]))
    if th==float(sys.argv[1]): np.save("hearing_groups.npy",g)
# show top-similar pairs examples at 0.7-0.8
import itertools
pairs=[(v,i,j) for i,j,v in zip(S.row,S.col,S.data) if i<j and df.h[i]!=df.h[j] and 0.6<v<0.8][:5]
for v,i,j in pairs: print(round(v,2),"|",df.topic[i][:120],"||",df.topic[j][:120])
