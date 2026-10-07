import pickle,sys,json,collections,numpy as np,pandas as pd,textwrap
sys.path.insert(0,'src'); from lib import *
sys.modules["__main__"].Corpus=None
import index as _ix
C=pickle.load(open('working/corpus.pkl','rb'))
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
g=dict(zip(tt.item_id,tt.amending_ids.map(json.loads)))
miss=[]
stats=collections.Counter()
for r in tr.itertuples():
    sec=norm(r.section_label).replace('s. ','').upper()
    cands=set(C.index.get((r.act_citation,sec),np.array([],dtype=np.int32)).tolist())
    gold=set(C.pos[u] for u in g[r.item_id])
    for u in gold-cands: miss.append((r,u))
    # date filter stats
    cd=[i for i in cands if C.date[i]<=r.date]
    stats['cand']+=len(cands); stats['cand_dated']+=len(cd)
    stats['gold']+=len(gold); stats['gold_in_dated']+=len(gold&set(cd))
    stats['fp_dated']+=len(set(cd)-gold)
print(dict(stats))
print("precision if we take all date-filtered cands: %.3f  recall %.3f"%(stats['gold_in_dated']/stats['cand_dated'],stats['gold_in_dated']/stats['gold']))
print("\n== MISSES (%d) =="%len(miss))
for r,u in miss[:6]:
    print("="*100)
    print(f"Q {r.act_title} ({r.act_citation}) {r.section_label} @{r.date}")
    print(f"P {C.title[u]} | {C.cit[u]} | {C.date[u]} | {C.label[u]}")
    print("ctx:",C.ctx[u][:200]); print("scope_act",C.scope_act[u],"ctx_act",C.ctx_act[u])
    print("txt:",textwrap.shorten(C.text[u],700))
