import sys,json,pickle,re
import pandas as pd,numpy as np
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
from lib import norm
from metric import text_score
from apply import reconstruct ,looks_tabular
C=pickle.load(open('working/corpus.pkl','rb'))
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
g={r.item_id:(json.loads(r.amending_ids),norm(r.text_at_date)) for r in tt.itertuples()}
sc=[];nap=[]
for r in tr.itertuples():
    ids,true=g[r.item_id]; sec=norm(r.section_label).replace('s. ','').upper()
    rows=sorted((C.pos[u] for u in ids), key=lambda i:(C.date[i],C.label[i]))
    plan=[(C.text[i],[pp for cc,pp in C.arefs[i] if cc==r.act_citation],
           looks_tabular(C.text[i])) for i in rows]
    out,k=reconstruct(norm(r.enacted_text),plan,sec)
    sc.append(text_score(r.enacted_text,out,true)); nap.append(k)
sc=np.array(sc)
print("ORACLE-IDS text editF1 = %.4f  => 0.60 term %.2f pts | zero: %.3f  applied-ops mean %.1f"%(sc.mean(),60*sc.mean(),(sc==0).mean(),np.mean(nap)))
print("percentiles",np.percentile(sc,[10,25,50,75,90]).round(3))
