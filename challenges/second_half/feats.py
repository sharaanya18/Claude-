"""Dev feature builder: pair features for (prefix, candidate) with fit-set exclusion."""
import numpy as np, scipy.sparse as sp
from sklearn.preprocessing import normalize
from common import *

class Data:
    def __init__(s, L, C, te_sessions):
        s.L=L
        s.aid={a:i for i,a in enumerate(L.artist.unique())}; s.NA=len(s.aid)
        s.sid={x:i for i,x in enumerate(L.session.unique())}; s.NS=len(s.sid)
        s.a=L.artist.map(s.aid).values; s.s=L.session.map(s.sid).values; s.h=L.half.values
        rel=L.release.fillna('NA_'+L.artist); s.rel=pd.factorize(rel)[0]; s.rec=pd.factorize(L.recording)[0]
        s.is_test=np.zeros(s.NS,bool)
        for x in te_sessions: s.is_test[s.sid[x]]=True
        def mat(tok,half=None):
            d=pd.DataFrame({'s':s.s,'t':tok,'h':s.h})
            if half: d=d[d.h==half]
            d=d.drop_duplicates(['s','t']); return sp.csr_matrix((np.ones(len(d),np.float32),(d.s.values,d.t.values)),shape=(s.NS,tok.max()+1))
        s.X1,s.Xall=mat(s.a,1),mat(s.a); s.R1,s.Rall=mat(s.rel,1),mat(s.rel); s.Q1,s.Qall=mat(s.rec,1),mat(s.rec)
        s.cc=C.groupby('continuation').artist.apply(lambda x:[s.aid[a] for a in x]).to_dict()
        s.ccrel=C.groupby('continuation').release.apply(lambda x:list(x)).to_dict()
        s.n1=np.asarray(s.X1.sum(1)).ravel()
        s.nall=np.asarray(s.Xall.sum(1)).ravel()
        s.pop1=np.asarray(s.X1[~s.is_test].sum(0)).ravel(); s.popall=np.asarray(s.Xall[~s.is_test].sum(0)).ravel()
        def idfw(X,w):
            df=np.asarray(X.sum(0)).ravel(); return sp.diags((np.log((s.NS+1)/(df+1))**w).astype(np.float32))
        s.Wa,s.Wr=idfw(s.Xall,.5),idfw(s.Rall,.5)
    def trans(s,fit,W=10,decay=.8):
        A=None; ok=fit[s.s]
        for d in range(1,W+1):
            m=(s.s[:-d]==s.s[d:])&ok[:-d]
            M=sp.csr_matrix((np.full(m.sum(),decay**(d-1),np.float32),(s.a[:-d][m],s.a[d:][m])),shape=(s.NA,s.NA)); A=M if A is None else A+M
        return A.tocsr()

def row_features(D,R,fit_idx,bg_idx):
    """R: dataframe of rows (cols pre,cand). fit_idx: session indices allowed for fitting. Returns (n,6,6,F), names."""
    Pidx=[D.sid[x] for l in R.pre for x in l]
    cols=sorted({a for c in R.cand for ci in c for a in D.cc[ci]}); cm={a:i for i,a in enumerate(cols)}
    fit=np.zeros(D.NS,bool); fit[fit_idx]=True
    Tall=D.Xall[fit_idx][:,cols].tocsr()
    feats={}
    def S_of(Kfn,idx,excl=False):
        K=Kfn(idx)
        if excl:
            pos={x:i for i,x in enumerate(fit_idx)}
            for r_,x in enumerate(idx): K[r_,pos[x]]=0
        return np.asarray(Tall.T.dot(K.T)).T
    Fa=normalize(D.Xall[fit_idx]@D.Wa); Fr=normalize(D.Rall[fit_idx]@D.Wr)
    Ka=lambda idx:(normalize(D.X1[idx]@D.Wa)@Fa.T).toarray()
    Kr=lambda idx:(normalize(D.R1[idx]@D.Wr)@Fr.T).toarray()
    S={}
    S['a1']=S_of(Ka,Pidx); S['a2']=S_of(lambda i:Ka(i)**2,Pidx); S['r1']=S_of(Kr,Pidx)
    B=S_of(Ka,bg_idx,excl=True); S['bmu']=np.tile(B.mean(0),(len(Pidx),1)); S['bsd']=np.tile(B.std(0)+1e-6,(len(Pidx),1))
    # transitions
    A=D.trans(fit); colsum=np.asarray(A.sum(0)).ravel()[cols]+1e-9
    rr=[];cc_=[];vv=[]
    for pi,x in enumerate(Pidx):
        idx=np.where((D.s==x)&(D.h==1))[0]; rr+=[pi]*len(idx); cc_+=list(D.a[idx]); vv+=[1]*len(idx)
    Pm=sp.csr_matrix((vv,(rr,cc_)),shape=(len(Pidx),D.NA)); Pm=sp.diags(1/np.asarray(Pm.sum(1)).ravel())@Pm
    S['t10']=(Pm@A[:,cols]).toarray()/colsum
    # assemble per row
    n=len(R); names=[]; out=[]
    npf=len(Pidx)
    for ri,(_,r) in enumerate(R.iterrows()):
        sl=slice(ri*6,ri*6+6)
        cidx=np.array([[cm[a] for a in D.cc[c]] for c in r.cand])  # (6 cand, 2)
        M=[]; nm=[]
        for q in (0,1):
            for k in ('a1','a2','r1','t10','bmu','bsd'):
                M.append(S[k][sl][:,cidx[:,q]]); nm.append(f'{k}_{q}')
        M=np.stack(M,-1)   # (6p,6c,F0)
        out.append(M); names=nm
    out=np.stack(out)       # (n,6,6,12)
    return out,names,Pidx


TRAITS=['lrec','lrel','lart','hit','emp']
def trait_features(D,R,fit_idx,C):
    """Listener-trait features: obscurity of recordings/releases/artists, hit-ratio, missing release.
    Counts come from fit sessions only (so held-out/test sessions never count themselves)."""
    L=D.L; fit=np.zeros(D.NS,bool); fit[fit_idx]=True; m=fit[D.s]
    rec_c=np.bincount(D.rec[m],minlength=D.rec.max()+1); rel_c=np.bincount(D.rel[m],minlength=D.rel.max()+1); art_c=np.bincount(D.a[m],minlength=D.NA)
    emp=L.release.isna().values.astype(np.float32)
    pl={'lrec':np.log1p(rec_c[D.rec]),'lrel':np.log1p(rel_c[D.rel]),'lart':np.log1p(art_c[D.a])}
    pl['hit']=pl['lrec']-pl['lart']; pl['emp']=emp
    # continuation first plays: need rec/rel/art codes for candidates
    recmap={r:i for i,r in enumerate(pd.factorize(L.recording)[1])}; relmap={r:i for i,r in enumerate(pd.factorize(L.release.fillna('NA_'+L.artist))[1])}
    cp={}
    crec=C.recording.map(recmap).fillna(-1).astype(int).values; cart=C.artist.map(D.aid).values
    crel=np.array([relmap.get(r,-1) if isinstance(r,str) else -1 for r in C.release.values])
    ca={'lrec':np.log1p(np.where(crec>=0,rec_c[np.clip(crec,0,None)],0)),'lart':np.log1p(art_c[cart])}
    ca['lrel']=np.log1p(np.where(crel>=0,rel_c[np.clip(crel,0,None)],0)); ca['hit']=ca['lrec']-ca['lart']; ca['emp']=C.release.isna().values.astype(np.float32)
    cdf=pd.DataFrame(ca); cdf['continuation']=C.continuation.values; cdf['rank']=C['rank'].values
    cvals={}
    for c,g in cdf.groupby('continuation'): cvals[c]=g.sort_values('rank')[TRAITS].to_numpy()   # (2,5)
    # prefix trait arrays
    out=[]
    for _,r in R.iterrows():
        pf=[]
        for x in r.pre:
            idx=np.where((D.s==D.sid[x])&(D.h==1))[0]
            pf.append(np.stack([pl[t][idx] for t in TRAITS],0))   # (5,n)
        cv=np.stack([cvals[c] for c in r.cand])      # (6,2,5)
        feat=np.zeros((6,6,len(TRAITS)*6+0),np.float32)
        for j in range(6):
            P=pf[j]
            for ti in range(len(TRAITS)):
                pm=P[ti].mean(); pmed=np.median(P[ti]); ps=P[ti].std()+0.3
                c=cv[:,:,ti]; cm_=c.mean(1)
                feat[j,:,ti*6+0]=(cm_-pm)/ps; feat[j,:,ti*6+1]=np.abs(cm_-pm); feat[j,:,ti*6+2]=cm_-pmed
                feat[j,:,ti*6+3]=(P[ti][None,:]<c[:,0:1]).mean(1)       # percentile of rank-1 value in prefix
                feat[j,:,ti*6+4]=(P[ti][None,:]<c[:,1:2]).mean(1)
                feat[j,:,ti*6+5]=pm
        out.append(feat)
    names=[f'{t}_{k}' for t in TRAITS for k in ('zd','ad','md','pc1','pc2','pm')]
    return np.stack(out),names
