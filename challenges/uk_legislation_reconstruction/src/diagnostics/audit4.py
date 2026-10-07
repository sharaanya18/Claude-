import pandas as pd, json, re, sys, collections, textwrap
sys.path.insert(0,'src')
from refs import *
D="dataset/public/"
c=pd.read_csv(D+"corpus.csv",keep_default_na=False); ci=c.set_index('unit_id')
tr=pd.read_csv(D+"train.csv",keep_default_na=False)
tt=pd.read_csv(D+"train_targets.csv",keep_default_na=False); tt['ids']=tt.amending_ids.map(json.loads)
m=tr.merge(tt,on='item_id')
none=[]
for r in m.itertuples():
    atn=norm(r.act_title); yr=r.act_citation.split()[0]; cno=r.act_citation.split('c.')[-1].strip()
    for uid in r.ids:
        g=ci.loc[uid]; T=norm(g.text); C=norm(g.context)
        if atn in T or atn in C or (f'c. {cno}' in T and yr in T) or f'c. {cno}' in C or norm(g.document_title)==atn: continue
        none.append((r,uid,g))
print("NONE count",len(none))
# cluster: which docs
print(collections.Counter(g.document_title for _,_,g in none).most_common(10))
print(collections.Counter(r.act_title for r,_,_ in none).most_common(10))
for r,uid,g in none[:8]:
    print("="*100)
    print(f"Q: {r.act_title} ({r.act_citation}) {r.section_label} @{r.date}")
    print(f"P[{uid}] {g.document_title} | {g.citation} | {g.document_date} | {g.label}")
    print("ctx:",g.context[:200])
    print("txt:",textwrap.shorten(norm(g.text),800))
