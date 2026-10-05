import numpy as np, time, scipy.sparse as sp, itertools
from feats import *
L,C,tr,te=load(); D=Data(L,C,set(s for l in te.pre for s in l))
fold=np.random.RandomState(0).randint(0,5,len(tr)); Y=np.array(list(tr.y))
# count matrices (plays) for tf weighting
d=pd.DataFrame({'s':D.s,'a':D.a,'h':D.h}); cnt_all=d.groupby(['s','a']).size().reset_index(name='c'); cnt1=d[d.h==1].groupby(['s','a']).size().reset_index(name='c')
mk=lambda x:sp.csr_matrix((x.c.values.astype(np.float32),(x.s.values,x.a.values)),shape=(D.NS,D.NA))
Ca,C1=mk(cnt_all),mk(cnt1)
def evalcfg(w,gam,tf,nb_repr,tgt,frac=1.0,topk=0,seed=0):
    rng=np.random.RandomState(seed); Ms={}
    for f in range(5):
        R=tr[fold==f]; held=np.zeros(D.NS,bool)
        for l in R.pre:
            for x in l: held[D.sid[x]]=True
        fi=np.where(~D.is_test&~held)[0]
        if frac<1: fi=rng.choice(fi,int(len(fi)*frac),replace=False)
        Pidx=[D.sid[x] for l in R.pre for x in l]
        cols=sorted({a for c in R.cand for ci in c for a in D.cc[ci]}); cm={a:i for i,a in enumerate(cols)}
        df=np.asarray(D.Xall[fi].sum(0)).ravel(); W=sp.diags((np.log((len(fi)+1)/(df+1))**w).astype(np.float32))
        tfun=(lambda X:X) if tf=='bin' else ((lambda X:X.sqrt()) if tf=='sqrt' else (lambda X:X.log1p()))
        Pm=normalize(tfun(C1[Pidx])@W)
        Nb=Ca if nb_repr=='all' else C1
        Fm=normalize(tfun(Nb[fi])@W)
        K=(Pm@Fm.T).toarray()
        if topk:
            thr=-np.partition(-K,topk,axis=1)[:,topk:topk+1]; K=np.where(K>=thr,K,0)
        K=K**gam
        T=(D.Xall if tgt=='all' else D.Xall)[fi][:,cols]
        S=np.asarray(T.T.dot(K.T)).T
        for ri,(idx,r) in enumerate(R.iterrows()):
            M=np.zeros((6,6))
            for j in range(6):
                for k,ci in enumerate(r.cand): M[j,k]=np.mean([S[ri*6+j,cm[a]] for a in D.cc[ci]])
            Ms[idx]=M
    out={}
    allv=np.concatenate([Ms[i].ravel() for i in tr.index]); sc=np.median(allv[allv>0])
    for e in (.003,.01,.03,.1,.3):
        pred=[hungarian(np.log(Ms[i]+e*sc)) for i in tr.index]; out[e]=round(score(np.array(pred),Y),4)
    return out
t0=time.time()
for kw in [dict(),dict(tf='log'),dict(tf='log',gam=.5),dict(tf='log',gam=.3),dict(tf='log',gam=.5,w=1),dict(tf='log',gam=.5,w=.25),dict(tf='sqrt',gam=.5)]:
    a=dict(w=.5,gam=1,tf='bin',nb_repr='all',tgt='all'); a.update(kw)
    print(kw,evalcfg(**a),flush=True)
