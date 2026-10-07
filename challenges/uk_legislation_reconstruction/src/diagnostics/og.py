import sys,json,pickle,re,numpy as np,pandas as pd
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
from lib import norm
from metric import mtok,edits,f1,text_score
from apply import apply_one, looks_tabular
from pipe import make_query,build_plan
C=pickle.load(open('working/corpus.pkl','rb'))
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
G={r.item_id:(set(json.loads(r.amending_ids)),norm(r.text_at_date)) for r in tt.itertuples()}
rows=list(tr.itertuples())
base=[];greedy=[];ngr=[];nall=[]
for r in rows:
    Q=make_query(r); ids,true=G[r.item_id]
    prows=[C.pos[u] for u in ids]
    instr,ctxs=build_plan(C,Q,prows,[1.0]*len(prows))
    en=mtok(Q['en']); et,kt=edits(en,mtok(true))
    def sc(s):
        ep,kp=edits(en,mtok(s)); return 0.0 if kp<0.5*kt else f1(ep,et)
    s=Q['en']; 
    for e in instr: s,_=apply_one(s,e)
    base.append(sc(re.sub(r'\s+',' ',s).strip()))
    # greedy forward: keep an instruction only if it improves F1 at its turn
    s=Q['en']; cur=sc(s); k=0
    for e in instr:
        s2,_=apply_one(s,e); v=sc(re.sub(r'\s+',' ',s2).strip())
        if v>cur+1e-12: s,cur=s2,v; k+=1
    greedy.append(cur); ngr.append(k); nall.append(len(instr))
print("all-apply %.4f | oracle-greedy-gate %.4f  (kept %.0f%% of %.1f instr/query)"%(
    np.mean(base),np.mean(greedy),100*np.sum(ngr)/max(1,np.sum(nall)),np.mean(nall)))
