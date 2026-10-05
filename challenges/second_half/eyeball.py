import numpy as np
exec(open('exp1.py').read().split("rows=tr.copy()")[0])
inv_a={v:k for k,v in aid.items()}
r=tr.iloc[3]
sub=L.groupby('s')
for j,s in enumerate(r.pre[:3]):
    d=L[L.s==sid[s]]; h1=d[d.half==1]; h2=d[d.half==2]
    print('SESSION',s,'n',len(d),'h1',len(h1))
    print(' h1 artists (order):',[inv_a[a][-4:] for a in h1.a.values[:40]])
    print(' h2 first artists:',[inv_a[a][-4:] for a in h2.a.values[:15]])
    print(' true cont:',[inv_a[a][-4:] for a in cc[r.cand[r.y[j]]]], 'relpos h1 head',h1.relative_position.values[:3],'sec',h1.seconds.values[:5], h1.seconds.values[-3:])
