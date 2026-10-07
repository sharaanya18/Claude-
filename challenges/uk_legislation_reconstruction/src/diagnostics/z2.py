import sys,json,pickle,re,collections,numpy as np,pandas as pd
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
from lib import norm
from metric import mtok,edits,f1
from apply import apply_one
from pipe import make_query,build_plan
C=pickle.load(open('working/corpus.pkl','rb'))
D='dataset/public/'
tr=pd.read_csv(D+'train.csv',keep_default_na=False)
tt=pd.read_csv(D+'train_targets.csv',keep_default_na=False)
G={r.item_id:(set(json.loads(r.amending_ids)),norm(r.text_at_date)) for r in tt.itertuples()}
cls=collections.Counter(); low=[]
for r in tr.itertuples():
    Q=make_query(r); ids,true=G[r.item_id]
    prows=[C.pos[u] for u in ids]
    instr,ctxs=build_plan(C,Q,prows,[1.0]*len(prows))
    en=mtok(Q['en']); et,kt=edits(en,mtok(true))
    s=Q['en']; nap=0
    for e in instr:
        s2,ok=apply_one(s,e); nap+=ok and s2!=s; s=s2
    out=re.sub(r'\s+',' ',s).strip(); ep,kp=edits(en,mtok(out))
    sc=0.0 if kp<0.5*kt else f1(ep,et)
    if sc<0.08:
        if kp<0.5*kt: cls['GUARD_zero (deleted too much)']+=1
        elif not instr: cls['no_instructions_parsed']+=1
        elif nap==0: cls['instructions_but_none_applied']+=1
        elif len(ep)==0: cls['no_edits_produced']+=1
        else: cls['edits_wrong_position_or_content']+=1
        low.append((r,instr,out,true,Q,len(ep),len(et),nap))
print("queries<0.08:",sum(cls.values())); [print("  ",k,v) for k,v in cls.most_common()]
print()
for r,instr,out,true,Q,ne,nt,nap in [x for x in low if x[7]>0][:4]:
    print("="*100); print("%s %s | instr=%d applied=%d  predE=%d trueE=%d"%(r.act_citation,r.section_label,len(instr),nap,ne,nt))
    for e in instr[:5]: print("   ",{k:(v[:70] if isinstance(v,str) else v) for k,v in e.items()})
    import difflib
    a=mtok(out); b=mtok(true)
    n=0
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes():
        if tag!='equal' and n<5: print("    MINE %r != GOLD %r"%(' '.join(a[i1:i2])[:110],' '.join(b[j1:j2])[:110])); n+=1
