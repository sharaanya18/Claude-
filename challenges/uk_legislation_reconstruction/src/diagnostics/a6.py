import pandas as pd,re,collections
c=pd.read_csv('dataset/public/corpus.csv',keep_default_na=False)
# how are abbreviations defined?
pat=re.compile(r'(Act\s+(?:19|20)\d\d)\s*(?:\(c\. ?\d+\)\s*)?\(\s*"([^"]{2,30})"\s*\)')
hits=collections.Counter()
for t in c.text.values[:60000]:
    for m in pat.finditer(t): hits[m.group(2)]+=1
print(hits.most_common(30))
print('--- SSCBA defs ---')
d=c[c.text.str.contains('"SSCBA 1992"',regex=False)]
print(len(d))
for r in d.head(3).itertuples(): print(r.citation,r.label,'|',r.text[:300])
print('--- "the 1992 Act" style ---')
d=c[c.text.str.contains('("the 1992 Act")',regex=False)]
print(len(d))
for r in d.head(3).itertuples(): print(r.citation,r.label,'|',r.text[:260])
