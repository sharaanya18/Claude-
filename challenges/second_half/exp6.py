import numpy as np, collections
exec(open('exp1.py').read().split("rows=tr.copy()")[0])
rid={r:i for i,r in enumerate(L.recording.unique())}; L['r']=L.recording.map(rid)
r_arr=L.r.values; s_arr=L.s.values; a_arr=L.a.values; h_arr=L.half.values
# position lists
starts=np.r_[0,np.where(s_arr[1:]!=s_arr[:-1])[0]+1,len(L)]
sess_range={s_arr[starts[i]]:(starts[i],starts[i+1]) for i in range(len(starts)-1)}
# index of recording->list of (global play index) in non-test sessions
idx=collections.defaultdict(list)
for i in np.where(~is_test[s_arr])[0]: idx[r_arr[i]].append(i)
tmap={}
for _,r in tr.iterrows():
    for s,m in zip(r.pre,r.y): tmap[sid[s]]=cc[r.cand[m]]
hit=tot=0; seqhit=0; wrong_hit=0
cov_any=0
for s,cont in list(tmap.items()):
    a,b=sess_range[s]; h1=[i for i in range(a,b) if h_arr[i]==1]
    last=h1[-1]; rec=r_arr[last]
    # other sessions (not this one) playing the same recording as the last prefix play
    matches=[i for i in idx[rec] if s_arr[i]!=s]
    tot+=1
    if matches: cov_any+=1
    found=False
    for i in matches:
        sa,sb=sess_range[s_arr[i]]
        nxt=a_arr[i+1:min(i+12,sb)]
        if set(nxt)&set(cont): found=True;break
    seqhit+=found
print('train prefixes',tot,'last recording seen in other fit sessions',cov_any/tot,'and a cont artist follows within 11 plays in some',seqhit/tot)
# baseline: random artist chance
