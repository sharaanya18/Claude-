import sys,json,pickle,re,collections
import pandas as pd
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
from lib import norm
from apply import all_instructions,split_clauses,other_section,parse_instructions,focus
C=pickle.load(open('working/corpus.pkl','rb'))
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
g={r.item_id:json.loads(r.amending_ids) for r in tt.itertuples()}
unp=collections.Counter(); ex=collections.defaultdict(list); seen=set(); nz=0; tot=0
for r in tr.itertuples():
    sec=norm(r.section_label).replace('s. ','').upper()
    for u in g[r.item_id]:
        i=C.pos[u]
        if (i,sec,r.act_citation) in seen: continue
        seen.add((i,sec,r.act_citation)); tot+=1
        t=C.text[i]; ap=[p for c,p in C.arefs[i] if c==r.act_citation]
        tbl=('Extent of repeal' in t) or ('Extent of revocation' in t)
        if all_instructions(t,sec,ap,tbl): continue
        nz+=1
        # which clauses mention our section but yielded nothing?
        head,cls=split_clauses(t[:60000])
        cand=[cl for cl in cls if re.search(r'\bsection\s+'+re.escape(sec)+r'\b',cl,re.I) or
              (head and re.search(r'\bsection\s+'+re.escape(sec)+r'\b',head,re.I))]
        if not cand:
            unp['no clause mentions the section (id-only target)']+=1; continue
        for cl in cand[:2]:
            sig=re.sub(r'"[^"]*"','"Q"',cl); sig=re.sub(r'\d+','N',sig)
            key=' '.join(sig.split()[:8])[:80]
            unp[key]+=1
            if len(ex[key])<2: ex[key].append(cl[:250])
print("gold provisions with no instruction: %d / %d"%(nz,tot))
for k,v in unp.most_common(22):
    print("%4d | %s"%(v,k))
    if ex[k]: print("       e.g.",ex[k][0][:210])
