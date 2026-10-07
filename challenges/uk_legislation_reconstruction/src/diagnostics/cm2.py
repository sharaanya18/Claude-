import sys,pickle,re,collections
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
C=pickle.load(open('working/corpus.pkl','rb'))
titles={cc:C.title[idx[0]] for cc,idx in C.doc.items()}
si={cc:t for cc,t in titles.items() if cc.startswith('S.I.')}
print("S.I. docs:",len(si),"Act docs:",len(titles)-len(si))
for kw in ['Commencement','Appointed Day','Transitional','Order','Regulations','Rules']:
    n=sum(1 for t in si.values() if kw.lower() in t.lower()); print(" %-14s %d"%(kw,n))
print("\nsample commencement-ish titles:")
for cc,t in list(si.items()):
    if 'commencement' in t.lower() or 'appointed day' in t.lower(): print("  ",cc,"|",t[:130])
