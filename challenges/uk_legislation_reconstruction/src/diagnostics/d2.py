import sys,json,pickle,re,difflib,numpy as np,pandas as pd
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
from lib import norm
from metric import mtok,text_score
from apply import reconstruct,focus,all_instructions
C=pickle.load(open('working/corpus.pkl','rb'))
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
g={r.item_id:(json.loads(r.amending_ids),norm(r.text_at_date)) for r in tt.itertuples()}
out=[]
for r in tr.itertuples():
    ids,true=g[r.item_id]; sec=norm(r.section_label).replace('s. ','').upper()
    rows=sorted((C.pos[u] for u in ids),key=lambda i:(C.date[i],C.label[i]))
    prov=[focus(C.text[i],sec,[pp for cc,pp in C.arefs[i] if cc==r.act_citation]) for i in rows]
    tbl=[('Extent of repeal' in C.text[i] or 'Extent of revocation' in C.text[i]) for i in rows]
    o,_=reconstruct(norm(r.enacted_text),prov,sec,tbl)
    out.append((text_score(r.enacted_text,o,true),r,o,true,len(rows),prov,sec))
out.sort(key=lambda x:-x[0])
print("dist:", [(f"{t:.1f}", int(sum(1 for x in out if t<=x[0]<t+0.1))) for t in np.arange(0,1.0,0.1)])
# show mid-range cases with 1 provision (cleanest signal)
sel=[x for x in out if 0.3<x[0]<0.85 and x[4]<=2][:6]
for sc,r,o,true,n,prov,sec in sel:
    print("="*110); print("score %.3f | %s %s"%(sc,r.act_citation,r.section_label))
    print("PROV:",prov[0][:400])
    a=mtok(o); b=mtok(true)
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes():
        if tag!='equal': print("   MINE %r  != GOLD %r"%(' '.join(a[i1:i2])[:140],' '.join(b[j1:j2])[:140]))
