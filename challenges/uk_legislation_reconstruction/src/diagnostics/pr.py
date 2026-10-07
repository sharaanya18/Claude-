import sys,json,pickle,numpy as np,pandas as pd
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
from lib import norm
from metric import mtok,edits
from apply import reconstruct,parse_instructions
C=pickle.load(open('working/corpus.pkl','rb'))
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
g={r.item_id:(json.loads(r.amending_ids),norm(r.text_at_date)) for r in tt.itertuples()}
TP=FP=FN=0; perq=[]
for r in tr.itertuples():
    ids,true=g[r.item_id]; sec=norm(r.section_label).replace('s. ','').upper()
    rows=sorted((C.pos[u] for u in ids),key=lambda i:(C.date[i],C.label[i]))
    plan=[(C.text[i],[pp for cc,pp in C.arefs[i] if cc==r.act_citation],
           ('Extent of repeal' in C.text[i] or 'Extent of revocation' in C.text[i])) for i in rows]
    out,_=reconstruct(norm(r.enacted_text),plan,sec)
    en=mtok(r.enacted_text)
    ep,_=edits(en,mtok(out)); et,_=edits(en,mtok(true))
    tp=len(ep&et); TP+=tp; FP+=len(ep)-tp; FN+=len(et)-tp
    perq.append((len(ep),len(et),tp))
print("edit-level  P=%.3f  R=%.3f  F1=%.3f"%(TP/(TP+FP),TP/(TP+FN),2*TP/(2*TP+FP+FN)))
pe=np.array([x[0] for x in perq]); te=np.array([x[1] for x in perq])
print("pred edits per q: mean %.0f median %.0f | true edits: mean %.0f median %.0f"%(pe.mean(),np.median(pe),te.mean(),np.median(te)))
print("queries with 0 pred edits: %d ; pred>>true (>3x): %d ; pred<<true (<1/3): %d"%((pe==0).sum(),(pe>3*te).sum(),(pe*3<te).sum()))
