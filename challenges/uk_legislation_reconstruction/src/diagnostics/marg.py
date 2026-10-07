import sys,json,pickle,collections,numpy as np,pandas as pd,re
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
from lib import norm
from metric import mtok,edits,f1
from apply import all_instructions,apply_one,focus
C=pickle.load(open('working/corpus.pkl','rb'))
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
g={r.item_id:(json.loads(r.amending_ids),norm(r.text_at_date)) for r in tt.itertuples()}
stat=collections.defaultdict(lambda:[0,0,0,0.0])   # op -> [n, npos, nneg, sumgain]
base_sc=[]; full_sc=[]
for r in tr.itertuples():
    ids,true=g[r.item_id]; sec=norm(r.section_label).replace('s. ','').upper()
    en=mtok(r.enacted_text); et,kt=edits(en,mtok(true))
    rows=sorted((C.pos[u] for u in ids),key=lambda i:(C.date[i],C.label[i]))
    plan=[]
    for i in rows:
        t=focus(C.text[i],sec,[pp for cc,pp in C.arefs[i] if cc==r.act_citation])
        tbl='Extent of repeal' in C.text[i] or 'Extent of revocation' in C.text[i]
        for e in all_instructions(t,sec,tbl): plan.append(e)
    def run(sel):
        s=norm(r.enacted_text)
        for k,e in enumerate(plan):
            if sel[k]: s,_=apply_one(s,e)
        ep,kp=edits(en,mtok(re.sub(r'\s+',' ',s).strip()))
        return (0.0 if kp<0.5*kt else f1(ep,et))
    sel=[True]*len(plan); full=run(sel); full_sc.append(full)
    for k,e in enumerate(plan):
        sel[k]=False; without=run(sel); sel[k]=True
        gain=full-without
        st=stat[e['op']]; st[0]+=1; st[3]+=gain
        if gain>1e-9: st[1]+=1
        elif gain<-1e-9: st[2]+=1
print("full mean %.4f"%np.mean(full_sc))
print("%-10s %6s %6s %6s %9s %9s"%("op","n","pos","neg","sumgain","avg"))
for op,(n,p,q,sg) in sorted(stat.items(),key=lambda x:-x[1][3]):
    print("%-10s %6d %6d %6d %9.2f %9.4f"%(op,n,p,q,sg,sg/n))
