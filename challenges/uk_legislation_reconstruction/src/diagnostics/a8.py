import sys,json,pandas as pd; sys.path.insert(0,'src')
from metric import *
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
q={r.item_id:r.enacted_text for r in tr.itertuples()}
truth={r.item_id:(json.loads(r.amending_ids),r.text_at_date) for r in tt.itertuples()}
print("perfect:", score_frame(q,truth,truth))
print("sample-like (empty ids, enacted text):", score_frame(q,{k:([],v) for k,v in q.items()},truth))
print("gold ids, enacted text:", score_frame(q,{k:(truth[k][0],q[k]) for k in q},truth))
print("empty ids, gold text:", score_frame(q,{k:([],truth[k][1]) for k in q},truth))
# sanity: tiny hand cases
en="a b c d e"; 
print("sub d->X:", text_score(en,"a b c X e","a b c X e"), text_score(en,"a b c X e","a b c Y e"))
print("delete all:", text_score(en,"","a b c X e"))
