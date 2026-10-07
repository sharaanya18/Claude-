import sys,pickle,re,collections,numpy as np,pandas as pd
sys.path.insert(0,'src'); import index as IX
sys.modules['__main__'].Corpus=IX.Corpus
C=pickle.load(open('working/corpus.pkl','rb'))
RE_CIF=re.compile(r'com(?:es?|ing)\s+into\s+force|shall\s+come\s+into\s+force|has\s+effect\s+from',re.I)
# how many units contain commencement language?
n=sum(1 for t in C.text if RE_CIF.search(t))
print("units with commencement language:",n)
docs=collections.Counter(C.cit[i] for i in range(C.n) if RE_CIF.search(C.text[i]))
print("docs:",len(docs))
# commencement orders by title
co=[cc for cc,idx in C.doc.items() if 'Commencement' in C.title[idx[0]]]
print("documents titled Commencement:",len(co))
# find ones referencing the Serious Crime Act 2007 and Schedule 8
hits=[i for i in range(C.n) if RE_CIF.search(C.text[i]) and ('2007 c. 27' in [c for c,_ in C.arefs[i]])]
print("units with cif language referencing 2007 c. 27:",len(hits))
for i in hits[:5]:
    print("---",C.title[i][:90],"|",C.cit[i],"|",C.date[i],"|",C.label[i])
    print("   ",C.text[i][:450])
