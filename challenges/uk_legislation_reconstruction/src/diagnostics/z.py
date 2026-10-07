import sys,json,pickle,re,collections,numpy as np,pandas as pd
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
from lib import norm
from metric import mtok,edits
from apply import all_instructions,focus,split_clauses,other_section,parse_instructions,parse_table
C=pickle.load(open('working/corpus.pkl','rb'))
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
g={r.item_id:(json.loads(r.amending_ids),norm(r.text_at_date)) for r in tt.itertuples()}
zero=[];sigs=collections.Counter();ex=collections.defaultdict(list)
nprov_noins=0; nprov=0
for r in tr.itertuples():
    ids,true=g[r.item_id]; sec=norm(r.section_label).replace('s. ','').upper()
    rows=sorted((C.pos[u] for u in ids),key=lambda i:(C.date[i],C.label[i]))
    tot=0
    for i in rows:
        t=focus(C.text[i],sec,[pp for cc,pp in C.arefs[i] if cc==r.act_citation])
        tbl='Extent of repeal' in C.text[i] or 'Extent of revocation' in C.text[i]
        k=len(all_instructions(t,sec,tbl)); tot+=k; nprov+=1
        if k==0:
            nprov_noins+=1
            head,cls=split_clauses(t)
            skipped=sum(1 for cl in cls if other_section(cl,head,sec))
            sig=('TABLE' if tbl else 'plain')+('|allskipped' if skipped==len(cls) else '')
            sigs[sig]+=1
            if len(ex[sig])<3: ex[sig].append((sec,t[:260]))
    if tot==0: zero.append((r,rows,sec))
print("provisions with 0 instructions: %d/%d"%(nprov_noins,nprov))
for k,v in sigs.most_common(): print(" ",k,v)
for k in sigs:
    for sec,t in ex[k][:2]: print("  [%s] sec=%s %s"%(k,sec,t[:230]))
print("\nqueries with no instruction at all:",len(zero))
for r,rows,sec in zero[:5]:
    print("---",r.act_citation,r.section_label,"nprov",len(rows))
    for i in rows[:2]: print("    ",C.label[i],"|",C.text[i][:230])
