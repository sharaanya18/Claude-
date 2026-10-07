import sys,pandas as pd
sys.path.insert(0,'src')
from lib import norm
from apply import markers,span_of
tr=pd.read_csv('dataset/public/train.csv',keep_default_na=False)
r=tr[(tr.act_citation=='1996 c. 18')&(tr.section_label=='s. 27')].iloc[0]
s=norm(r.enacted_text)
print(s[:700]); print()
mk=markers(s)
print("markers:",[(t,l) for t,l,a,b in mk][:40])
for ch in (['1'],['1','f'],['1','ca'],['2']):
    print(ch,'->',span_of(s,ch) and s[span_of(s,ch)[0]:span_of(s,ch)[0]+70])
