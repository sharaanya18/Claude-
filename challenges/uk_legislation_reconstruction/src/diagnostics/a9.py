import sys,json,pickle,re,collections
import pandas as pd,numpy as np
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
from lib import norm
C=pickle.load(open('working/corpus.pkl','rb'))
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
g={r.item_id:json.loads(r.amending_ids) for r in tt.itertuples()}
# collect gold provision texts, split into instruction clauses, count operation verbs
ops=collections.Counter(); scopes=collections.Counter()
for r in tr.itertuples():
    for u in g[r.item_id]:
        t=C.text[C.pos[u]]
        for m in re.finditer(r'\b(substitute|there is substituted|insert|there is inserted|omit|repeal\w*|revoke\w*|is amended|are amended|cease\w* to have effect|add)\b',t,re.I):
            ops[m.group(1).lower()[:14]]+=1
        for m in re.finditer(r'\bin (subsection|paragraph|sub-paragraph|section|subsections|paragraphs|the definition|each)\b',t,re.I):
            scopes[m.group(1).lower()]+=1
print("OPS:",ops.most_common(14)); print("SCOPES:",scopes.most_common(10))
# gold text edit profile
from metric import mtok,edits
ins_tot=dele=0; nblocks=[]
import difflib
for r in tr.itertuples():
    en=mtok(r.enacted_text); at=mtok(tt[tt.item_id==r.item_id].text_at_date.iloc[0])
    sm=difflib.SequenceMatcher(a=en,b=at,autojunk=False)
    ops2=[o for o in sm.get_opcodes() if o[0]!='equal']
    nblocks.append(len(ops2))
    for tag,i1,i2,j1,j2 in ops2: dele+=i2-i1; ins_tot+=j2-j1
print("mean change blocks per query %.1f median %.0f  tokens deleted %d inserted %d"%(np.mean(nblocks),np.median(nblocks),dele,ins_tot))
