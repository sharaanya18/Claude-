"""Family-aware hearing folds (train only): cluster episode topics into paraphrase families, then partition hearings so families stay together."""
import json, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
P="dataset/public"; K=5
tr=pd.read_csv(f"{P}/train.csv")
fold=np.full(len(tr),-1)
for kind in ["position","correction"]:
    hs=np.where(tr.interaction_kind==kind)[0]
    topics=[]; owner=[]
    for h in hs:
        for e in json.loads(tr.episodes[h]): topics.append(e["reference"]+" "+e["opening_user"]); owner.append(h)
    owner=np.array(owner)
    X=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,stop_words="english").fit_transform(topics)
    Z=TruncatedSVD(200,random_state=0).fit_transform(X); Z/=np.linalg.norm(Z,axis=1,keepdims=True)+1e-9
    fam=AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average",distance_threshold=0.5).fit_predict(Z)
    vc=pd.Series(fam).value_counts(); print(kind,"families",len(vc),"largest",vc.iloc[:5].tolist(),"singleton share",round((vc==1).sum()/len(fam),3))
    H=len(hs); hidx={h:i for i,h in enumerate(hs)}
    hf=[[] for _ in range(H)]
    for f_,o in zip(fam,owner): hf[hidx[o]].append(f_)
    rng=np.random.default_rng(0); a=rng.permutation(H)%K
    cap=int(np.ceil(H/K*1.05))
    for it in range(30):
        moved=0
        cnt={}  # family -> fold counts
        for i in range(H):
            for f_ in hf[i]: cnt.setdefault(f_,np.zeros(K,int))[a[i]]+=1
        size=np.bincount(a,minlength=K)
        for i in rng.permutation(H):
            for f_ in hf[i]: cnt[f_][a[i]]-=1
            sc=np.array([sum(cnt[f_][k] for f_ in hf[i]) for k in range(K)],float)
            sc[size>=cap]=-1; sc[a[i]]+=1e-6
            k=int(np.argmax(sc))
            if k!=a[i]: size[a[i]]-=1; size[k]+=1; a[i]=k; moved+=1
            for f_ in hf[i]: cnt[f_][a[i]]+=1
        if moved==0: break
    split=np.mean([ (c>0).sum()>1 for c in cnt.values()])
    multi=[c for c in cnt.values() if c.sum()>1]; split_multi=np.mean([(c>0).sum()>1 for c in multi])
    print(kind,"fold sizes",np.bincount(a,minlength=K),"share of multi-member families split across folds",round(split_multi,3), "(multi-member families:",len(multi),")")
    fold[hs]=a
np.save("folds.npy",fold); print("saved folds.npy")
