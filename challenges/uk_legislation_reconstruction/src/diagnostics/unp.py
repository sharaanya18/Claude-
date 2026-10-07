import sys,json,pickle,re,collections
import pandas as pd
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
from lib import norm
from apply import parse_instructions,split_clauses,focus
C=pickle.load(open('working/corpus.pkl','rb'))
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
g={r.item_id:json.loads(r.amending_ids) for r in tt.itertuples()}
unp=collections.Counter(); ex=collections.defaultdict(list)
seen=set()
for r in tr.itertuples():
    sec=norm(r.section_label).replace('s. ','').upper()
    for u in g[r.item_id]:
        i=C.pos[u]
        if (i,sec) in seen: continue
        seen.add((i,sec))
        t=focus(C.text[i],sec,[pp for cc,pp in C.arefs[i] if cc==r.act_citation])
        head,cls=split_clauses(t)
        for cl in cls:
            if parse_instructions(cl,sec) or parse_instructions(head+" "+cl,sec): continue
            if ('section '+sec) not in cl.lower() and not head: continue
            # signature: leading words + first verb
            sig=re.sub(r'"[^"]*"','"Q"',cl)[:70]
            sig=re.sub(r'\d+','N',sig)
            key=' '.join(sig.split()[:7])
            unp[key]+=1; ex[key].append(cl[:230])
print("unparsed clauses:",sum(unp.values()))
for k,v in unp.most_common(28):
    print("%4d | %s"%(v,k)); print("       e.g.",ex[k][0][:200])
