import sys,json,pickle,importlib,numpy as np,pandas as pd,re
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
from lib import norm
from metric import text_score
C=pickle.load(open('working/corpus.pkl','rb'))
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
G={r.item_id:(json.loads(r.amending_ids),norm(r.text_at_date)) for r in tt.itertuples()}
def run(mod):
    out={}
    for r in tr.itertuples():
        ids,true=G[r.item_id]; sec=norm(r.section_label).replace('s. ','').upper()
        rows=sorted((C.pos[u] for u in ids),key=lambda i:(C.date[i],C.label[i]))
        plan=[(C.text[i],[pp for cc,pp in C.arefs[i] if cc==r.act_citation],
               ('Extent of repeal' in C.text[i] or 'Extent of revocation' in C.text[i])) for i in rows]
        o,_=mod.reconstruct(norm(r.enacted_text),plan,sec)
        out[r.item_id]=(text_score(r.enacted_text,o,true),o)
    return out
import apply as A2; A2.TRUNC_SOFT=False
sys.path.insert(0,'/tmp/claude-0/-home-user-Claude-/99c30af6-2f07-5670-9d10-aa251fa1ba75/scratchpad')
import importlib.util
spec=importlib.util.spec_from_file_location("apply_v1","/tmp/claude-0/-home-user-Claude-/99c30af6-2f07-5670-9d10-aa251fa1ba75/scratchpad/apply_v1.py")
A1=importlib.util.module_from_spec(spec); spec.loader.exec_module(A1)
r1=run(A1); r2=run(A2)
d=sorted(((r2[k][0]-r1[k][0],k) for k in r1))
print("v1 %.4f  v2 %.4f"%(np.mean([v[0] for v in r1.values()]),np.mean([v[0] for v in r2.values()])))
print("regressed:",sum(1 for x,_ in d if x<-0.01),"improved:",sum(1 for x,_ in d if x>0.01))
q=tr.set_index('item_id')
for delta,k in d[:6]:
    print("  %+0.3f %s %s %s  v1=%.2f v2=%.2f"%(delta,k,q.loc[k,'act_citation'],q.loc[k,'section_label'],r1[k][0],r2[k][0]))
