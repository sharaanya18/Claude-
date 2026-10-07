"""For each gold change block, did we produce it? Separates anchor errors from content errors."""
import sys,json,pickle,re,difflib,collections
import numpy as np,pandas as pd
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
from lib import norm
from metric import mtok,edits,f1
from apply import reconstruct,looks_tabular
C=pickle.load(open('working/corpus.pkl','rb'))
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
G={r.item_id:(json.loads(r.amending_ids),norm(r.text_at_date)) for r in tt.itertuples()}
def blocks(a,b):
    out=[]
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes():
        if tag!='equal': out.append((tag,i1,i2,tuple(b[j1:j2])))
    return out
cls=collections.Counter(); ntok=collections.Counter()
for r in tr.itertuples():
    ids,true=G[r.item_id]; sec=norm(r.section_label).replace('s. ','').upper()
    rws=sorted((C.pos[u] for u in ids),key=lambda i:(C.date[i],C.label[i]))
    plan=[(C.text[i],[p for c,p in C.arefs[i] if c==r.act_citation],looks_tabular(C.text[i])) for i in rws]
    out,_=reconstruct(norm(r.enacted_text),plan,sec)
    en=mtok(r.enacted_text)
    gb=blocks(en,mtok(true)); pb=blocks(en,mtok(out))
    pby=collections.defaultdict(list)
    for tag,i1,i2,nw in pb: pby[(i1,i2)].append(nw)
    anchors={i1 for _,i1,_,_ in pb}
    for tag,i1,i2,nw in gb:
        w=max(1,len(nw)+(i2-i1)); ntok['gold_blocks']+=1
        if (i1,i2) in pby:
            cand=pby[(i1,i2)]
            if nw in cand: cls['exact block']+=1; ntok['hit_tok']+=w
            else:
                best=max((len(set(nw)&set(c))/max(1,len(set(nw)|set(c))) for c in cand),default=0)
                cls['right anchor+span, content %s'%('close' if best>0.5 else 'wrong')]+=1
                ntok['part_tok']+=w*best
        elif i1 in anchors: cls['right anchor, wrong span']+=1
        elif any(abs(a-i1)<=3 for a in anchors): cls['anchor off by <=3 tokens']+=1
        else: cls['no edit near this position']+=1
print("gold change blocks:",ntok['gold_blocks'])
for k,v in cls.most_common(): print("  %-34s %5d  (%4.1f%%)"%(k,v,100*v/ntok['gold_blocks']))
