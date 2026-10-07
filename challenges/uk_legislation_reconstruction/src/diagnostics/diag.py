import sys,json,pickle,re,difflib,collections
import pandas as pd,numpy as np
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
from lib import norm
from metric import text_score,mtok
from apply import reconstruct,focus,parse_instructions,split_clauses
C=pickle.load(open('working/corpus.pkl','rb'))
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
g={r.item_id:(json.loads(r.amending_ids),norm(r.text_at_date)) for r in tt.itertuples()}
res=[]
for r in tr.itertuples():
    ids,true=g[r.item_id]; sec=norm(r.section_label).replace('s. ','').upper()
    rows=sorted((C.pos[u] for u in ids),key=lambda i:(C.date[i],C.label[i]))
    prov=[focus(C.text[i],sec,[pp for cc,pp in C.arefs[i] if cc==r.act_citation]) for i in rows]
    out,k=reconstruct(norm(r.enacted_text),prov,sec)
    nins=sum(len(parse_instructions(p,sec)) for p in prov)
    res.append((text_score(r.enacted_text,out,true),r,rows,prov,out,true,sec,k,nins))
res.sort(key=lambda x:x[0])
print("n_instructions parsed=0 for %d/%d queries"%(sum(1 for x in res if x[8]==0),len(res)))
print("score by nins bucket:")
for lo,hi in [(0,0),(1,2),(3,5),(6,100)]:
    sub=[x[0] for x in res if lo<=x[8]<=hi]
    if sub: print("  nins %d-%d n=%d mean %.3f"%(lo,hi,len(sub),np.mean(sub)))
print("score by #provisions:")
for lo,hi in [(1,1),(2,3),(4,8),(9,100)]:
    sub=[x[0] for x in res if lo<=len(x[2])<=hi]
    if sub: print("  nprov %d-%d n=%d mean %.3f"%(lo,hi,len(sub),np.mean(sub)))
for sc,r,rows,prov,out,true,sec,k,nins in res[:4]:
    print("="*110); print("score %.3f | %s %s @%s | nprov=%d nins=%d applied=%d"%(sc,r.act_citation,r.section_label,r.date,len(rows),nins,k))
    for p in prov[:4]:
        print("  PROV:",p[:420])
        print("    PARSED:",[{kk:(vv[:60] if isinstance(vv,str) else vv) for kk,vv in e.items()} for e in parse_instructions(p,sec)][:4])
    en=mtok(r.enacted_text); tgt=mtok(true)
    sm=difflib.SequenceMatcher(a=en,b=tgt,autojunk=False)
    print("  GOLD EDITS:")
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag!='equal': print("    %s en[%d:%d]=%r -> %r"%(tag,i1,i2,' '.join(en[i1:i2])[:110],' '.join(tgt[j1:j2])[:160]))
