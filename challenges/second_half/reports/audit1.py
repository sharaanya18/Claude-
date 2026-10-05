import json, numpy as np, pandas as pd
P='dataset/public/'
L=pd.read_csv(P+'listens.csv'); C=pd.read_csv(P+'continuations.csv')
tr=pd.read_csv(P+'train.csv'); te=pd.read_csv(P+'test.csv')
for d in (tr,te):
    d['pre']=d.prefixes.map(json.loads); d['cand']=d.candidates.map(json.loads)
print(L.shape, L.session.nunique(), L.artist.nunique(), L.release.isna().mean())
trs=set(s for l in tr.pre for s in l); tes=set(s for l in te.pre for s in l)
print('train sess',len(trs),'test sess',len(tes),'overlap',len(trs&tes))
# sessions per row membership across rows
print('sessions reused across train rows', pd.Series([s for l in tr.pre for s in l]).value_counts().max(),
      'test', pd.Series([s for l in te.pre for s in l]).value_counts().max())
trc=set(c for l in tr.cand for c in l); tec=set(c for l in te.cand for c in l)
print('cont train',len(trc),'test',len(tec),'in C',C.continuation.nunique(), 'overlap',len(trc&tec))
# halves per session
g=L.groupby('session').agg(n=('half','size'),h2=('half',lambda x:(x==2).sum()))
g['is_test']=g.index.isin(tes); g['is_train']=g.index.isin(trs)
print(g.groupby(['is_test','is_train']).agg(cnt=('n','size'),n_med=('n','median'),h2_sum=('h2','sum')))
print('prefix len test', g[g.is_test].n.describe())
print('prefix len train(first half)', L[L.session.isin(trs)&(L.half==1)].groupby('session').size().describe())
# full sessions available
fs=g[~g.is_test]; print('full sessions',len(fs),'of which with half2',(fs.h2>0).sum())
# continuation structure verify on train: rebuild from second half
L=L.sort_values(['session','relative_position'])
first=L[L.half==1].groupby('session').artist.apply(set)
a1=L[L.half==1]
cnt=a1.drop_duplicates(['session','artist']).groupby('artist').size()
print('artists with >=3 first-half sessions', (cnt>=3).sum(), 'of', len(cnt))
Cm=C.merge(C.groupby('continuation').size().rename('k'),on='continuation'); print(Cm.k.value_counts())
# verify rebuild for a few train rows
ok=bad=0
tmap={}
for _,r in tr.iterrows():
    for s,m in zip(r.pre,[r.match_1,r.match_2,r.match_3,r.match_4,r.match_5,r.match_6]): tmap[s]=r.cand[m-1]
cc=C.sort_values(['continuation','rank']).groupby('continuation').artist.apply(list).to_dict()
for s,c in list(tmap.items())[:2000]:
    s2=L[(L.session==s)&(L.half==2)]
    seen=first[s]; out=[]
    for a in s2.artist:
        if a in seen or a in out or cnt.get(a,0)<3: continue
        out.append(a)
        if len(out)==2: break
    if out==cc[c]: ok+=1
    else: bad+=1
print('rebuild ok',ok,'bad',bad)
# continuation artist coverage in training-day first-halves (non-test sessions)
nontest=set(g.index[~g.is_test])
cnt_nt=L[L.session.isin(nontest)&(L.half==1)].drop_duplicates(['session','artist']).groupby('artist').size()
for name,cs in (('train',trc),('test',tec)):
    arts=C[C.continuation.isin(cs)].artist
    print(name,'cont artists cnt in nontest first halves: zero',(arts.map(cnt_nt).fillna(0)==0).mean(),'<3',(arts.map(cnt_nt).fillna(0)<3).mean(), 'median',arts.map(cnt_nt).fillna(0).median())
# do cont artists appear in the same-row other prefixes' published plays? (should be no)
