import sys; sys.path.insert(0,'dev')
from common import *
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
def cats(df):
    cnt = df["annotator_count"].astype(int).astype(str); a,b=df["emotion_a"],df["emotion_b"]
    return pd.DataFrame({"cnt":cnt,"a":a,"b":b,"ca":cnt+"_"+a,"cb":cnt+"_"+b,"pair":a+"|"+b})
def mk(use_text, C=0.3):
    def fp(trd, y, ted):
        enc=OneHotEncoder(handle_unknown="ignore"); xt=[enc.fit_transform(cats(trd))]; xe=[enc.transform(cats(ted))]
        if use_text:
            v=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True); xt.append(v.fit_transform(trd.text)); xe.append(v.transform(ted.text))
        return LogisticRegression(C=C,max_iter=3000).fit(hstack(xt).tocsr(),y).predict_proba(hstack(xe).tocsr())[:,1]
    return fp
tr,te=load()
for ut in (False,True):
    print('use_text',ut); print(cv(mk(ut),tr)[0])
