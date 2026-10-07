import pandas as pd, json, re, sys, collections
sys.path.insert(0,'src')
from refs import *
D="dataset/public/"
c=pd.read_csv(D+"corpus.csv",keep_default_na=False)
tr=pd.read_csv(D+"train.csv",keep_default_na=False)
tt=pd.read_csv(D+"train_targets.csv",keep_default_na=False); tt['ids']=tt.amending_ids.map(json.loads)
m=tr.merge(tt,on='item_id')
ci=c.set_index('unit_id')
gold=[(r.item_id,r.act_title,r.act_citation,r.section_label,r.date,uid) for r in m.itertuples() for uid in r.ids]
print("gold pairs",len(gold))
# how is the act identified in the gold provision?
cnt=collections.Counter(); datecnt=collections.Counter()
for item,at,ac,sl,qd,uid in gold:
    g=ci.loc[uid]; T=norm(g.text); C=norm(g.context); DT=norm(g.document_title)
    atn=norm(at); secn=sl.replace('s. ','').upper()
    yr=ac.split()[0]; cno=ac.split('c.')[-1].strip()
    short=re.sub(r'^The ','',atn)
    hit=[]
    if atn in T: hit.append('title_in_text')
    if atn in C: hit.append('title_in_ctx')
    if f'c. {cno}' in T and yr in T: hit.append('cit_in_text')
    if f'c. {cno}' in C: hit.append('cit_in_ctx')
    if DT==atn: hit.append('same_doc')
    cnt[tuple(hit) if hit else ('NONE',)]+=1
    sm=sec_mentions(T)
    datecnt[('sec_in_text', secn in sm)]+=1
    datecnt[('docdate<=qd', g.document_date<=qd)]+=1
for k,v in cnt.most_common(20): print(v,k)
print()
for k,v in datecnt.items(): print(k,v)
